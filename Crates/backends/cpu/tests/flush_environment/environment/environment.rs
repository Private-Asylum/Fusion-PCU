//! Test-only thread-local floating state. No production execution changes the environment.
use std::{marker::PhantomData, rc::Rc};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct State {
    control: u64,
    status: u64,
}

pub struct Guard {
    original: State,
    _thread: PhantomData<Rc<()>>,
}

#[cfg(target_arch = "x86_64")]
pub fn state() -> State {
    let mut control = 0_u32;
    // SAFETY: x86_64 baseline provides SSE2; this stores exactly four bytes into live
    // aligned stack storage and only reads the current thread's MXCSR.
    unsafe {
        core::arch::asm!("stmxcsr [{pointer}]", pointer=in(reg) &raw mut control, options(nostack,preserves_flags));
    }
    State {
        control: u64::from(control),
        status: 0,
    }
}

#[cfg(target_arch = "x86_64")]
fn set(state: State) {
    let control = u32::try_from(state.control).unwrap();
    // SAFETY: The value is either an exact earlier snapshot or changes only mask-proved
    // DAZ/FZ bits. The four-byte operand remains live; only this thread is affected.
    unsafe {
        core::arch::asm!("ldmxcsr [{pointer}]", pointer=in(reg) &raw const control, options(nostack,preserves_flags));
    }
}

#[cfg(target_arch = "x86_64")]
fn supported() -> bool {
    #[repr(C, align(16))]
    struct Save([u8; 512]);
    let mut saved = Save([0; 512]);
    // SAFETY: x86_64 baseline supports FXSAVE; the operand has the required16-byte
    // alignment and complete512-byte writable storage. FXSAVE does not alter FP state.
    unsafe {
        core::arch::asm!("fxsave64 [{pointer}]", pointer=in(reg) saved.0.as_mut_ptr(), options(nostack,preserves_flags));
    }
    let mask = u32::from_le_bytes(saved.0[28..32].try_into().unwrap());
    // The architectural zero-mask fallback excludes DAZ, so do not attempt unsupported bits.
    mask & 0x8040 == 0x8040
}

#[cfg(target_arch = "aarch64")]
pub fn state() -> State {
    let control: u64;
    let status: u64;
    // SAFETY: User-mode AArch64 permits current-thread FPCR/FPSR reads; registers
    // are copied without memory access or modification of the floating environment.
    unsafe {
        core::arch::asm!("mrs {control}, fpcr", "mrs {status}, fpsr", control=out(reg) control,status=out(reg) status,options(nostack,preserves_flags));
    }
    State { control, status }
}

#[cfg(target_arch = "aarch64")]
fn set(state: State) {
    // SAFETY: Only the saved current-thread FPCR or its defined FZ-bit variant is written;
    // FPSR is restored exactly. ISB makes the local control change visible before tests run.
    unsafe {
        core::arch::asm!("msr fpcr, {control}", "msr fpsr, {status}", "isb", control=in(reg) state.control,status=in(reg) state.status,options(nostack,preserves_flags));
    }
}

impl Guard {
    pub fn enter(flush: bool) -> Self {
        #[cfg(target_arch = "x86_64")]
        assert!(
            supported(),
            "exact DAZ/FZ fixture requires MXCSR_MASK support"
        );
        let original = state();
        #[cfg(target_arch = "x86_64")]
        let mask = 0x8040;
        #[cfg(target_arch = "aarch64")]
        let mask = 1 << 24;
        let control = if flush {
            original.control | mask
        } else {
            original.control & !mask
        };
        set(State {
            control,
            status: original.status,
        });
        Self {
            original,
            _thread: PhantomData,
        }
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        set(self.original);
    }
}
