//! Optional host timestamp-counter clock for benchmark insight ledgers.
//!
//! This module deliberately exposes raw elapsed timestamp-counter ticks. They are not
//! necessarily CPU core cycles, and they are not GPU timings. The RDTSCP AUX value is
//! returned as an opaque context so consumers can reject samples that crossed contexts.

#[cfg(feature = "insights")]
use fusion_pcu::insights::{
    InsightClock,
    InsightStamp,
};

/// Whether this target can use the serialized x86 timestamp-counter clock.
#[must_use]
pub const fn is_supported() -> bool {
    cfg!(all(feature = "insights", target_arch = "x86_64"))
}

/// A clear explanation when the optional clock is unavailable on this target.
#[must_use]
pub const fn unsupported_reason() -> Option<&'static str> {
    if !cfg!(feature = "insights") {
        Some("enable the example crate's `insights` feature")
    } else if !cfg!(target_arch = "x86_64") {
        Some("serialized RDTSCP insights are implemented only for x86_64")
    } else {
        None
    }
}

/// Serialized x86-64 timestamp-counter clock.
///
/// The clock path performs no heap allocation. Its `ticks` are elapsed TSC ticks, not
/// calibrated or guaranteed to equal actual CPU cycles. Construction is only available when
/// both the feature and target architecture are enabled; the CPU must implement RDTSCP.
#[cfg(all(feature = "insights", target_arch = "x86_64"))]
#[derive(Clone, Copy, Debug, Default)]
pub struct SerializedTscClock;

#[cfg(all(feature = "insights", target_arch = "x86_64"))]
impl InsightClock for SerializedTscClock {
    #[inline]
    fn stamp(&mut self) -> InsightStamp {
        let (ticks, context) = read_serialized_rdtscp();
        InsightStamp { ticks, context }
    }
}

/// Read RDTSCP between LFENCEs to order the timestamp sample with surrounding work.
///
/// `RDTSCP` reports `TSC_AUX` in ECX. The assembler block intentionally omits `nomem`, making
/// it a compiler memory barrier as well as using the processor fences. The narrow unsafe
/// allowance is required because executing this instruction sequence is target-specific.
#[cfg(all(feature = "insights", target_arch = "x86_64"))]
#[inline]
fn read_serialized_rdtscp() -> (u64, u64) {
    let low: u32;
    let high: u32;
    let aux: u32;

    // SAFETY: This code is compiled only for x86_64 and uses RDTSCP's documented EAX/EDX/ECX
    // outputs. CPUs without RDTSCP are outside this clock's documented support.
    #[allow(unsafe_code)]
    unsafe {
        std::arch::asm!(
            "lfence",
            "rdtscp",
            "lfence",
            lateout("eax") low,
            lateout("edx") high,
            lateout("ecx") aux,
            options(nostack, preserves_flags),
        );
    }

    (u64::from(low) | (u64::from(high) << 32), u64::from(aux))
}
