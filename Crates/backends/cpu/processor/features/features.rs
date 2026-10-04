//! OS-aware instruction facts, independent from qualified operation implementations.
//! Rust detection follows CPU feature bits and required OS register-state support:
//! <https://doc.rust-lang.org/std/macro.is_x86_feature_detected.html>
//! <https://doc.rust-lang.org/std/arch/macro.is_aarch64_feature_detected.html>
//! Detection alone never admits a numeric profile or permits changing IEEE rounding.

/// Runtime instruction facts; availability does not advertise broader IR support.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)] // Each bit is an independent hardware/OS fact.
pub struct PcuCpuFeatures {
    pub sse: bool,
    pub sse2: bool,
    pub sse3: bool,
    pub ssse3: bool,
    pub sse41: bool,
    pub sse42: bool,
    pub avx: bool,
    pub avx2: bool,
    pub fma: bool,
    pub f16c: bool,
    pub avxvnni: bool,
    pub avxvnniint8: bool,
    pub avxvnniint16: bool,
    pub avxifma: bool,
    pub avxneconvert: bool,
    pub avx512f: bool,
    pub avx512cd: bool,
    pub avx512bw: bool,
    pub avx512dq: bool,
    pub avx512vl: bool,
    pub avx512ifma: bool,
    pub avx512vbmi: bool,
    pub avx512vbmi2: bool,
    pub avx512vnni: bool,
    pub avx512bitalg: bool,
    pub avx512vpopcntdq: bool,
    pub avx512bf16: bool,
    pub avx512fp16: bool,
    pub avx512vp2intersect: bool,
    pub neon: bool,
    pub arm_fp16: bool,
    pub arm_bf16: bool,
    pub dotprod: bool,
    pub i8mm: bool,
    pub sve: bool,
    pub sve2: bool,
    pub f32mm: bool,
    pub f64mm: bool,
}

impl PcuCpuFeatures {
    pub(super) const NONE: Self = Self {
        sse: false,
        sse2: false,
        sse3: false,
        ssse3: false,
        sse41: false,
        sse42: false,
        avx: false,
        avx2: false,
        fma: false,
        f16c: false,
        avxvnni: false,
        avxvnniint8: false,
        avxvnniint16: false,
        avxifma: false,
        avxneconvert: false,
        avx512f: false,
        avx512cd: false,
        avx512bw: false,
        avx512dq: false,
        avx512vl: false,
        avx512ifma: false,
        avx512vbmi: false,
        avx512vbmi2: false,
        avx512vnni: false,
        avx512bitalg: false,
        avx512vpopcntdq: false,
        avx512bf16: false,
        avx512fp16: false,
        avx512vp2intersect: false,
        neon: false,
        arm_fp16: false,
        arm_bf16: false,
        dotprod: false,
        i8mm: false,
        sve: false,
        sve2: false,
        f32mm: false,
        f64mm: false,
    };

    #[cfg(feature = "std")]
    pub(super) fn detect() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            Self {
                sse: std::is_x86_feature_detected!("sse"),
                sse2: std::is_x86_feature_detected!("sse2"),
                sse3: std::is_x86_feature_detected!("sse3"),
                ssse3: std::is_x86_feature_detected!("ssse3"),
                sse41: std::is_x86_feature_detected!("sse4.1"),
                sse42: std::is_x86_feature_detected!("sse4.2"),
                avx: std::is_x86_feature_detected!("avx"),
                avx2: std::is_x86_feature_detected!("avx2"),
                fma: std::is_x86_feature_detected!("fma"),
                f16c: std::is_x86_feature_detected!("f16c"),
                avxvnni: std::is_x86_feature_detected!("avxvnni"),
                avxvnniint8: std::is_x86_feature_detected!("avxvnniint8"),
                avxvnniint16: std::is_x86_feature_detected!("avxvnniint16"),
                avxifma: std::is_x86_feature_detected!("avxifma"),
                avxneconvert: std::is_x86_feature_detected!("avxneconvert"),
                avx512f: std::is_x86_feature_detected!("avx512f"),
                avx512cd: std::is_x86_feature_detected!("avx512cd"),
                avx512bw: std::is_x86_feature_detected!("avx512bw"),
                avx512dq: std::is_x86_feature_detected!("avx512dq"),
                avx512vl: std::is_x86_feature_detected!("avx512vl"),
                avx512ifma: std::is_x86_feature_detected!("avx512ifma"),
                avx512vbmi: std::is_x86_feature_detected!("avx512vbmi"),
                avx512vbmi2: std::is_x86_feature_detected!("avx512vbmi2"),
                avx512vnni: std::is_x86_feature_detected!("avx512vnni"),
                avx512bitalg: std::is_x86_feature_detected!("avx512bitalg"),
                avx512vpopcntdq: std::is_x86_feature_detected!("avx512vpopcntdq"),
                avx512bf16: std::is_x86_feature_detected!("avx512bf16"),
                avx512fp16: std::is_x86_feature_detected!("avx512fp16"),
                avx512vp2intersect: std::is_x86_feature_detected!("avx512vp2intersect"),
                ..Self::NONE
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self {
                neon: std::arch::is_aarch64_feature_detected!("neon"),
                arm_fp16: std::arch::is_aarch64_feature_detected!("fp16"),
                arm_bf16: std::arch::is_aarch64_feature_detected!("bf16"),
                dotprod: std::arch::is_aarch64_feature_detected!("dotprod"),
                i8mm: std::arch::is_aarch64_feature_detected!("i8mm"),
                sve: std::arch::is_aarch64_feature_detected!("sve"),
                sve2: std::arch::is_aarch64_feature_detected!("sve2"),
                f32mm: std::arch::is_aarch64_feature_detected!("f32mm"),
                f64mm: std::arch::is_aarch64_feature_detected!("f64mm"),
                ..Self::NONE
            }
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self::NONE
        }
    }
}
