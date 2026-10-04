//! Cold CPU instruction detection and guarded, bit-preserving SIMD execution.

/// CPU instruction families; each operation separately checks its required extensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuCpuImplementation {
    Scalar,
    Sse2,
    Avx2,
    Neon,
    /// AVX sign-bit operations; integer arithmetic needs a separate family.
    Avx,
    /// AVX-512 foundation. Individual arithmetic profiles may require BW/DQ/VL.
    Avx512,
}

#[path = "features/features.rs"]
mod features;
pub use features::PcuCpuFeatures;

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
            features: PcuCpuFeatures::NONE,
        }
    }

    /// Detects instructions through the standard library's platform and OS-aware probes.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn detect() -> Self {
        let features = PcuCpuFeatures::detect();
        Self { features }
    }

    #[must_use]
    pub const fn features(self) -> PcuCpuFeatures {
        self.features
    }

    /// Chooses the widest implemented instruction family, without a throughput claim.
    #[must_use]
    pub const fn widest(self) -> PcuCpuImplementation {
        if self.features.avx512f {
            PcuCpuImplementation::Avx512
        } else if self.features.avx2 {
            PcuCpuImplementation::Avx2
        } else if self.features.avx {
            PcuCpuImplementation::Avx
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
            PcuCpuImplementation::Avx => self.features.avx,
            PcuCpuImplementation::Avx512 => self.features.avx512f,
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
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        PcuCpuImplementation::Avx => unsafe { x86::avx(input, output) },
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        PcuCpuImplementation::Avx512 => unsafe { x86::avx512(input, output) },
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
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        PcuCpuImplementation::Avx => unsafe { x86::avx_f64(input, output) },
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        PcuCpuImplementation::Avx512 => unsafe { x86::avx512_f64(input, output) },
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
