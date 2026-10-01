//! Cold CPU instruction detection and guarded, bit-preserving SIMD execution.

/// CPU implementations of the bounded checked F32/F64 negation profiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuCpuImplementation {
    Scalar,
    Sse2,
    Avx2,
    Neon,
}

/// Runtime instruction facts; availability does not advertise broader IR support.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)] // Independent instruction facts, not policy states.
pub struct PcuCpuFeatures {
    pub sse2: bool,
    pub avx: bool,
    pub avx2: bool,
    pub neon: bool,
}

/// Explicit request for an unavailable CPU implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PcuCpuImplementationUnavailable(pub PcuCpuImplementation);

/// Unforgeable instruction facts obtained through runtime detection.
///
/// The `no_std` entry point intentionally promises only scalar execution. Hosted consumers
/// enable `std` to detect available SIMD. Detection belongs outside the warm kernel call.
#[derive(Debug, Clone, Copy)]
pub struct PcuCpuProcessor {
    features: PcuCpuFeatures,
}

impl PcuCpuProcessor {
    #[must_use]
    pub const fn scalar() -> Self {
        Self {
            features: PcuCpuFeatures {
                sse2: false,
                avx: false,
                avx2: false,
                neon: false,
            },
        }
    }

    /// Detects instructions through the standard library's platform and OS-aware probes.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn detect() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let features = PcuCpuFeatures {
            sse2: std::is_x86_feature_detected!("sse2"),
            avx: std::is_x86_feature_detected!("avx"),
            avx2: std::is_x86_feature_detected!("avx2"),
            neon: false,
        };
        #[cfg(target_arch = "aarch64")]
        let features = PcuCpuFeatures {
            neon: std::arch::is_aarch64_feature_detected!("neon"),
            ..PcuCpuFeatures::default()
        };
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        let features = PcuCpuFeatures::default();
        Self { features }
    }

    #[must_use]
    pub const fn features(self) -> PcuCpuFeatures {
        self.features
    }

    /// Chooses the widest implemented instruction family, without a throughput claim.
    #[must_use]
    pub const fn widest(self) -> PcuCpuImplementation {
        if self.features.avx2 {
            PcuCpuImplementation::Avx2
        } else if self.features.sse2 {
            PcuCpuImplementation::Sse2
        } else if self.features.neon {
            PcuCpuImplementation::Neon
        } else {
            PcuCpuImplementation::Scalar
        }
    }

    /// Checks an explicit instruction request. Unavailable SIMD is rejected at preparation.
    ///
    /// # Errors
    /// Returns the unavailable requested implementation; it never substitutes another one.
    pub const fn require(
        self,
        implementation: PcuCpuImplementation,
    ) -> Result<(), PcuCpuImplementationUnavailable> {
        let supported = match implementation {
            PcuCpuImplementation::Scalar => true,
            PcuCpuImplementation::Sse2 => self.features.sse2,
            PcuCpuImplementation::Avx2 => self.features.avx2,
            PcuCpuImplementation::Neon => self.features.neon,
        };
        if supported {
            Ok(())
        } else {
            Err(PcuCpuImplementationUnavailable(implementation))
        }
    }
}

pub fn negate_bytes(
    processor: PcuCpuProcessor,
    implementation: PcuCpuImplementation,
    input: &[u8],
    output: &mut [u8],
) {
    assert_eq!(input.len(), output.len());
    assert_eq!(input.len() % 4, 0);
    processor
        .require(implementation)
        .expect("prepared instruction facts");
    // SAFETY: Detection facts cannot be constructed by callers. Each target-specific entry is
    // guarded by require above and compile-time architecture gating. The slices are disjoint
    // Rust borrows with equal lengths, and every unaligned load/store stays within those lengths.
    let processed = match implementation {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        PcuCpuImplementation::Sse2 => unsafe { x86::sse2(input, output) },
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        PcuCpuImplementation::Avx2 => unsafe { x86::avx2(input, output) },
        #[cfg(target_arch = "aarch64")]
        PcuCpuImplementation::Neon => unsafe { neon::negate(input, output) },
        _ => 0,
    };
    for (source, destination) in input[processed..]
        .as_chunks::<4>()
        .0
        .iter()
        .zip(output[processed..].as_chunks_mut::<4>().0.iter_mut())
    {
        let bits = u32::from_ne_bytes(*source);
        destination.copy_from_slice(&(bits ^ 0x8000_0000).to_ne_bytes());
    }
}

pub fn negate_f64_bytes(
    processor: PcuCpuProcessor,
    implementation: PcuCpuImplementation,
    input: &[u8],
    output: &mut [u8],
) {
    assert_eq!(input.len(), output.len());
    assert_eq!(input.len() % 8, 0);
    processor
        .require(implementation)
        .expect("prepared instruction facts");
    // SAFETY: Private detection tokens guard each target-feature entry. Equal, disjoint
    // byte slices bound every unaligned vector access; only complete vectors are processed.
    let processed = match implementation {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        PcuCpuImplementation::Sse2 => unsafe { x86::sse2_f64(input, output) },
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        PcuCpuImplementation::Avx2 => unsafe { x86::avx2_f64(input, output) },
        #[cfg(target_arch = "aarch64")]
        PcuCpuImplementation::Neon => unsafe { neon::negate_f64(input, output) },
        _ => 0,
    };
    for (source, destination) in input[processed..]
        .as_chunks::<8>()
        .0
        .iter()
        .zip(output[processed..].as_chunks_mut::<8>().0.iter_mut())
    {
        let bits = u64::from_ne_bytes(*source);
        destination.copy_from_slice(&(bits ^ 0x8000_0000_0000_0000).to_ne_bytes());
    }
}

#[cfg(target_arch = "aarch64")]
#[path = "neon/neon.rs"]
mod neon;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[path = "x86/x86.rs"]
mod x86;
