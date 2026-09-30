//! Cold cuBLAS configuration identity for explicitly native compound arithmetic.
//!
//! NVIDIA's CUDA Toolkit 13.4 cuBLAS documentation describes real SGEMM/DGEMM
//! `DEFAULT_MATH` with compute and
//! intermediate storage of at least the requested mantissa and exponent precision.
//! This does not promise checked exceptions, strict operation order or portable bits.
//! Explicit optimized F32 permits the documented TF32 tensor mode and its input rounding.
//! Environment snapshots guard precision overrides; getters only verify programmed modes.

#[rustfmt::skip]
use std::ffi::{
    OsStr,
    OsString,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuPrecisionPolicy,
    PcuScalarType,
};
use crate::CublasError;

const ENVIRONMENT_NAMES: [&str; 6] = [
    "CUBLAS_EMULATION_STRATEGY",
    "CUBLAS_EMULATION_SPECIAL_VALUES_SUPPORT_MASK",
    "CUBLAS_EMULATE_SINGLE_PRECISION",
    "CUBLAS_EMULATE_DOUBLE_PRECISION",
    "CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT",
    "NVIDIA_TF32_OVERRIDE",
];

/// The relevant process configuration captured before native compound preparation.
///
/// Values remain attached to the prepared handle. PCU does not change process environment or
/// retune a warm handle. Applications must finish process configuration before preparation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CublasEnvironmentSnapshot {
    values: [Option<OsString>; 6],
}

impl CublasEnvironmentSnapshot {
    /// Captures the documented precision and emulation override variables without SDK calls.
    #[must_use]
    pub fn capture() -> Self {
        Self {
            values: ENVIRONMENT_NAMES.map(std::env::var_os),
        }
    }

    /// Returns a captured value for one documented override.
    #[must_use]
    pub fn value(&self, name: &str) -> Option<&OsStr> {
        ENVIRONMENT_NAMES
            .iter()
            .position(|candidate| *candidate == name)
            .and_then(|index| self.values[index].as_deref())
    }

    fn validate_preserve(&self) -> Result<(), CublasError> {
        for (name, value) in ENVIRONMENT_NAMES.iter().zip(&self.values) {
            if let Some(value) = value {
                // Explicit disable is documented. Any other present value requires an
                // implementation offer with proved override semantics, including non-UTF8.
                if matches!(
                    *name,
                    "CUBLAS_EMULATE_SINGLE_PRECISION"
                        | "CUBLAS_EMULATE_DOUBLE_PRECISION"
                        | "NVIDIA_TF32_OVERRIDE"
                ) && value == "0"
                {
                    continue;
                }
                return Err(CublasError::UnsupportedNumericalConfiguration(name));
            }
        }
        Ok(())
    }
}

/// Immutable native compound precision choice and its process configuration identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CublasNumericalConfig {
    scalar_type: PcuScalarType,
    precision: PcuPrecisionPolicy,
    environment: CublasEnvironmentSnapshot,
}

impl CublasNumericalConfig {
    /// Assesses a native real GEMM precision choice without touching CUDA or loading cuBLAS.
    ///
    /// # Errors
    ///
    /// Rejects unsupported scalar types and unproved precision overrides under Preserve.
    pub fn new(
        scalar_type: PcuScalarType,
        precision: PcuPrecisionPolicy,
        environment: CublasEnvironmentSnapshot,
    ) -> Result<Self, CublasError> {
        if !matches!(scalar_type, PcuScalarType::F32 | PcuScalarType::F64) {
            return Err(CublasError::UnsupportedNumericalConfiguration(
                "native GEMM requires F32 or F64",
            ));
        }
        if precision == PcuPrecisionPolicy::Preserve {
            environment.validate_preserve()?;
        }
        Ok(Self {
            scalar_type,
            precision,
            environment,
        })
    }

    /// Logical input/output scalar precision.
    #[must_use]
    pub const fn scalar_type(&self) -> PcuScalarType {
        self.scalar_type
    }

    /// Explicit compound arithmetic precision permission.
    #[must_use]
    pub const fn precision(&self) -> PcuPrecisionPolicy {
        self.precision
    }

    /// Captured precision and emulation overrides.
    #[must_use]
    pub const fn environment(&self) -> &CublasEnvironmentSnapshot {
        &self.environment
    }

    pub(crate) const fn math_mode(&self) -> i32 {
        if matches!(self.scalar_type, PcuScalarType::F32)
            && matches!(self.precision, PcuPrecisionPolicy::BackendOptimized)
        {
            crate::ffi::cublas::CUBLAS_TF32_TENSOR_OP_MATH
        } else {
            crate::ffi::cublas::CUBLAS_DEFAULT_MATH
        }
    }
}

/// Observed library properties and programmed handle modes after cold setup.
///
/// This reports configuration, not the effective internal algorithm or a portable bit claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CublasConfiguredModes {
    pub library_major: i32,
    pub library_minor: i32,
    pub library_patch: i32,
    pub math_mode: i32,
    pub atomics_mode: i32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn environment(value: Option<(&str, &OsStr)>) -> CublasEnvironmentSnapshot {
        CublasEnvironmentSnapshot {
            values: ENVIRONMENT_NAMES.map(|name| {
                value
                    .as_ref()
                    .filter(|(selected, _)| *selected == name)
                    .map(|(_, value)| (*value).to_owned())
            }),
        }
    }

    #[test]
    fn native_precision_modes_are_independent_of_strict_order_and_reproduction() {
        for scalar in [PcuScalarType::F32, PcuScalarType::F64] {
            let preserve =
                CublasNumericalConfig::new(scalar, PcuPrecisionPolicy::Preserve, environment(None))
                    .unwrap();
            assert_eq!(
                preserve.math_mode(),
                crate::ffi::cublas::CUBLAS_DEFAULT_MATH
            );
            let optimized = CublasNumericalConfig::new(
                scalar,
                PcuPrecisionPolicy::BackendOptimized,
                environment(None),
            )
            .unwrap();
            assert_eq!(
                optimized.math_mode(),
                if scalar == PcuScalarType::F32 { 3 } else { 0 }
            );
        }
    }

    #[test]
    fn preserve_rejects_enabled_and_unknown_overrides_without_sdk_work() {
        for name in ENVIRONMENT_NAMES {
            for value in ["1", "eager", "", "unknown"] {
                let snapshot = environment(Some((name, OsStr::new(value))));
                assert!(
                    CublasNumericalConfig::new(
                        PcuScalarType::F64,
                        PcuPrecisionPolicy::Preserve,
                        snapshot.clone()
                    )
                    .is_err()
                );
                assert!(
                    CublasNumericalConfig::new(
                        PcuScalarType::F64,
                        PcuPrecisionPolicy::BackendOptimized,
                        snapshot
                    )
                    .is_ok()
                );
            }
        }
        for name in [
            "CUBLAS_EMULATE_SINGLE_PRECISION",
            "CUBLAS_EMULATE_DOUBLE_PRECISION",
            "NVIDIA_TF32_OVERRIDE",
        ] {
            assert!(
                CublasNumericalConfig::new(
                    PcuScalarType::F32,
                    PcuPrecisionPolicy::Preserve,
                    environment(Some((name, OsStr::new("0"))))
                )
                .is_ok()
            );
        }
        assert!(
            CublasNumericalConfig::new(
                PcuScalarType::F16,
                PcuPrecisionPolicy::BackendOptimized,
                environment(None)
            )
            .is_err()
        );
    }

    #[cfg(unix)]
    #[test]
    fn preserve_rejects_non_utf8_overrides_without_sdk_work() {
        use std::os::unix::ffi::OsStrExt;

        let value = OsStr::from_bytes(&[0xff]);
        for name in ENVIRONMENT_NAMES {
            assert!(
                CublasNumericalConfig::new(
                    PcuScalarType::F32,
                    PcuPrecisionPolicy::Preserve,
                    environment(Some((name, value)))
                )
                .is_err()
            );
        }
    }
}
