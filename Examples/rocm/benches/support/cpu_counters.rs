//! Optional Linux hardware-counter sampling for the current benchmark thread.
//!
//! The group reports raw user-mode hardware cycles and instructions plus the kernel's
//! enabled/running times. Counts are never scaled when the kernel multiplexes the events.

#[cfg(all(feature = "insights", target_os = "linux", target_arch = "x86_64"))]
mod linux_x86_64 {
    use std::{
        io,
        mem::size_of,
        os::fd::{
            AsRawFd,
            FromRawFd,
            OwnedFd,
        },
    };

    // Values and ABI layout follow /usr/include/linux/perf_event.h. Keep this at the
    // original 64-byte PERF_ATTR_SIZE_VER0 prefix for broad kernel compatibility.
    const PERF_TYPE_HARDWARE: u32 = 0;
    const PERF_COUNT_HW_CPU_CYCLES: u64 = 0;
    const PERF_COUNT_HW_INSTRUCTIONS: u64 = 1;
    const PERF_ATTR_SIZE_VER0: u32 = 64;
    const PERF_FORMAT_TOTAL_TIME_ENABLED: u64 = 1 << 0;
    const PERF_FORMAT_TOTAL_TIME_RUNNING: u64 = 1 << 1;
    const PERF_FORMAT_GROUP: u64 = 1 << 3;
    const PERF_FLAG_FD_CLOEXEC: libc::c_ulong = 1 << 3;
    const PERF_IOC_FLAG_GROUP: libc::c_ulong = 1;
    // Linux _IO('$', nr): _IOC_NONE | ('$' << 8) | nr.
    const PERF_EVENT_IOC_ENABLE: libc::c_ulong = 0x24 << 8;
    const PERF_EVENT_IOC_DISABLE: libc::c_ulong = (0x24 << 8) | 1;
    const PERF_EVENT_IOC_RESET: libc::c_ulong = (0x24 << 8) | 3;
    const ATTR_DISABLED: u64 = 1 << 0;
    const ATTR_EXCLUDE_KERNEL: u64 = 1 << 5;
    const ATTR_EXCLUDE_HV: u64 = 1 << 6;
    const EXPECTED_GROUP_EVENTS: u64 = 2;

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct PerfEventAttrV0 {
        type_: u32,
        size: u32,
        config: u64,
        sample_period: u64,
        sample_type: u64,
        read_format: u64,
        flags: u64,
        wakeup_events: u32,
        bp_type: u32,
        config1: u64,
    }

    /// Raw, unscaled counts and enabled/running time deltas for one measurement interval.
    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
    pub struct CpuCounterSample {
        pub cycles: u64,
        pub instructions: u64,
        pub time_enabled: u64,
        pub time_running: u64,
    }

    /// Current-thread grouped hardware counters. The group is opened disabled and counts only
    /// user mode; kernel and hypervisor execution are excluded for both members.
    pub struct CpuCounters {
        leader: OwnedFd,
        _instructions: OwnedFd,
        running: bool,
        previous_enabled: u64,
        previous_running: u64,
    }

    impl CpuCounters {
        /// Open a two-event group for the calling thread. Permissions and hardware support are
        /// determined by Linux and the PMU; no privilege or system setting is changed here.
        pub fn open() -> io::Result<Self> {
            if size_of::<PerfEventAttrV0>() != 64 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "unexpected perf_event_attr ABI size",
                ));
            }

            let mut leader_attr = event_attr(PERF_COUNT_HW_CPU_CYCLES);
            let leader = open_event(&mut leader_attr, -1)?;
            let mut instructions_attr = event_attr(PERF_COUNT_HW_INSTRUCTIONS);
            let instructions = open_event(&mut instructions_attr, leader.as_raw_fd())?;

            Ok(Self {
                leader,
                _instructions: instructions,
                running: false,
                previous_enabled: 0,
                previous_running: 0,
            })
        }

        /// Reset and enable both grouped counters. Calling this while already running is an
        /// error, which avoids silently discarding a measurement interval.
        pub fn start(&mut self) -> io::Result<()> {
            if self.running {
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    "CPU counter group is already running",
                ));
            }
            group_ioctl(&self.leader, PERF_EVENT_IOC_RESET)?;
            group_ioctl(&self.leader, PERF_EVENT_IOC_ENABLE)?;
            self.running = true;
            Ok(())
        }

        /// Disable the group and read its raw counts and enabled/running times.
        ///
        /// When `time_running < time_enabled`, the kernel multiplexed the group. Values remain
        /// raw; callers can reject or annotate that sample rather than receiving scaled counts.
        pub fn stop(&mut self) -> io::Result<CpuCounterSample> {
            if !self.running {
                return Err(io::Error::new(
                    io::ErrorKind::NotConnected,
                    "CPU counter group is not running",
                ));
            }
            group_ioctl(&self.leader, PERF_EVENT_IOC_DISABLE)?;
            self.running = false;

            // PERF_FORMAT_GROUP + both time fields: nr, enabled, running, cycles, instructions.
            let mut words = [0_u64; 5];
            // SAFETY: `words` is a writable, properly aligned buffer of the exact expected
            // five-u64 group-read payload; the kernel writes at most the supplied byte length.
            #[allow(unsafe_code)]
            let result = unsafe {
                libc::read(
                    self.leader.as_raw_fd(),
                    words.as_mut_ptr().cast(),
                    size_of::<[u64; 5]>(),
                )
            };
            if result < 0 {
                return Err(io::Error::last_os_error());
            }
            if usize::try_from(result).ok() != Some(size_of::<[u64; 5]>()) {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "short read from grouped CPU counters",
                ));
            }
            if words[0] != EXPECTED_GROUP_EVENTS {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "CPU counter group returned an unexpected event count",
                ));
            }

            let time_enabled = words[1]
                .checked_sub(self.previous_enabled)
                .ok_or_else(counter_time_error)?;
            let time_running = words[2]
                .checked_sub(self.previous_running)
                .ok_or_else(counter_time_error)?;
            self.previous_enabled = words[1];
            self.previous_running = words[2];

            Ok(CpuCounterSample {
                time_enabled,
                time_running,
                cycles: words[3],
                instructions: words[4],
            })
        }
    }

    fn event_attr(config: u64) -> PerfEventAttrV0 {
        PerfEventAttrV0 {
            type_: PERF_TYPE_HARDWARE,
            size: PERF_ATTR_SIZE_VER0,
            config,
            read_format: PERF_FORMAT_GROUP
                | PERF_FORMAT_TOTAL_TIME_ENABLED
                | PERF_FORMAT_TOTAL_TIME_RUNNING,
            // `exclude_user` stays clear; kernel and hypervisor execution are excluded.
            flags: ATTR_DISABLED | ATTR_EXCLUDE_KERNEL | ATTR_EXCLUDE_HV,
            ..PerfEventAttrV0::default()
        }
    }

    fn counter_time_error() -> io::Error {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "perf-event time moved backwards",
        )
    }

    fn open_event(attr: &mut PerfEventAttrV0, group_fd: libc::c_int) -> io::Result<OwnedFd> {
        // pid=0 attaches to the calling thread; cpu=-1 follows that thread across CPUs.
        // SAFETY: `attr` has the kernel-defined PERF_ATTR_SIZE_VER0 layout and remains live for
        // the syscall. The returned nonnegative descriptor is uniquely owned below.
        #[allow(unsafe_code)]
        let fd = unsafe {
            libc::syscall(
                libc::SYS_perf_event_open,
                std::ptr::from_mut(attr),
                0,
                -1,
                group_fd,
                PERF_FLAG_FD_CLOEXEC,
            )
        };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        let raw_fd = libc::c_int::try_from(fd)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "perf fd overflow"))?;
        // SAFETY: the successful syscall returned a new descriptor with CLOEXEC; ownership is
        // transferred exactly once into `OwnedFd`.
        #[allow(unsafe_code)]
        Ok(unsafe { OwnedFd::from_raw_fd(raw_fd) })
    }

    fn group_ioctl(fd: &OwnedFd, request: libc::c_ulong) -> io::Result<()> {
        // SAFETY: the fd is a live perf-event descriptor; group flag applies to its event group.
        #[allow(unsafe_code)]
        let result = unsafe { libc::ioctl(fd.as_raw_fd(), request, PERF_IOC_FLAG_GROUP) };
        if result < 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }
}

/// Whether grouped Linux hardware counters are available for this build target and feature set.
#[must_use]
pub const fn is_supported() -> bool {
    cfg!(all(
        feature = "insights",
        target_os = "linux",
        target_arch = "x86_64"
    ))
}

/// Explanation for builds where the Linux x86-64 counter helper is unavailable.
#[must_use]
pub const fn unsupported_reason() -> Option<&'static str> {
    if !cfg!(feature = "insights") {
        Some("enable the example crate's `insights` feature")
    } else if !cfg!(target_os = "linux") {
        Some("hardware perf counters are implemented only on Linux")
    } else if !cfg!(target_arch = "x86_64") {
        Some("hardware perf counters are implemented only on x86_64")
    } else {
        None
    }
}

#[cfg(all(feature = "insights", target_os = "linux", target_arch = "x86_64"))]
pub use linux_x86_64::{
    CpuCounterSample,
    CpuCounters,
};
