//! Structural admission for the bounded scalar `i32` arithmetic map profile.

use crate::{
    PcuBindingRef,
    PcuDispatchKernelIr,
    PcuDispatchValueId,
    PcuValueType,
    PcuValueTypeCaps,
};
use crate::map_validation::integer_map_validation::{
    validate_integer_checked_div_rem_kernel,
    validate_integer_map_kernel,
    IntegerMapValidationError,
};

/// Structural failure within the typed scalar `i32` arithmetic map profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuI32MapValidationError {
    UnsupportedInterface,
    UnsupportedRequirements,
    InvalidBinding(PcuBindingRef),
    DuplicateBinding(PcuBindingRef),
    UnsupportedOperation(usize),
    InvalidIndex(usize),
    InvalidValue(PcuDispatchValueId),
    MissingReturn,
}

const fn map_error(error: IntegerMapValidationError) -> PcuI32MapValidationError {
    match error {
        IntegerMapValidationError::UnsupportedInterface => {
            PcuI32MapValidationError::UnsupportedInterface
        }
        IntegerMapValidationError::UnsupportedRequirements => {
            PcuI32MapValidationError::UnsupportedRequirements
        }
        IntegerMapValidationError::InvalidBinding(value) => {
            PcuI32MapValidationError::InvalidBinding(value)
        }
        IntegerMapValidationError::DuplicateBinding(value) => {
            PcuI32MapValidationError::DuplicateBinding(value)
        }
        IntegerMapValidationError::UnsupportedOperation(value) => {
            PcuI32MapValidationError::UnsupportedOperation(value)
        }
        IntegerMapValidationError::InvalidIndex(value) => {
            PcuI32MapValidationError::InvalidIndex(value)
        }
        IntegerMapValidationError::InvalidValue(value) => {
            PcuI32MapValidationError::InvalidValue(value)
        }
        IntegerMapValidationError::MissingReturn => PcuI32MapValidationError::MissingReturn,
    }
}

/// Validates indexed `i32` loads, one or more wrapping Add/Sub/Mul operations, one store and Return.
///
/// The exact operation sequence is required. It may be direct (`InvocationId`) or one bounded
/// `GridStrideLoop` (`GridStrideId`) followed by Return. Binding coverage and extents are checked
/// by the dispatch ABI. `i32` division and remainder are outside this profile.
///
/// # Errors
///
/// Returns the first structural violation of the typed `i32` map profile.
pub fn validate_i32_map_kernel(
    kernel: &PcuDispatchKernelIr<'_>,
) -> Result<(), PcuI32MapValidationError> {
    validate_integer_map_kernel(kernel, PcuValueType::i32(), PcuValueTypeCaps::INT32)
        .map_err(map_error)
}

/// Validates the exact two-input/two-output checked `i32` quotient/remainder profile.
///
/// Direct indexing or one canonical grid-stride loop is accepted. Zero divisors and
/// `i32::MIN / -1` are execution faults; both output buffers are unusable after a fault.
///
/// # Errors
///
/// Returns the first structural violation of the checked `DivRem` profile.
pub fn validate_i32_checked_div_rem_kernel(
    kernel: &PcuDispatchKernelIr<'_>,
) -> Result<(), PcuI32MapValidationError> {
    validate_integer_checked_div_rem_kernel(kernel, PcuValueType::i32(), PcuValueTypeCaps::INT32)
        .map_err(map_error)
}

#[cfg(test)]
mod tests {
    use std::boxed::Box;
    use super::{
        validate_i32_checked_div_rem_kernel,
        validate_i32_map_kernel,
        PcuI32MapValidationError,
    };
    use crate::{
        PcuBinding,
        PcuBindingAccess,
        PcuBindingRef,
        PcuBindingStorageClass,
        PcuDispatchAluOp,
        PcuDispatchControlOp,
        PcuDispatchDataOp,
        PcuDispatchEntryPoint,
        PcuDispatchIndex,
        PcuDispatchKernelIr,
        PcuDispatchOp,
        PcuDispatchValueId,
        PcuKernelId,
        PcuScalarType,
        PcuValueType,
        PcuValueTypeCaps,
    };

    fn kernel(extent: Option<u32>, alu: PcuDispatchAluOp) -> PcuDispatchKernelIr<'static> {
        let bindings = Box::leak(Box::new([
            PcuBinding::scalar::<i32>(
                Some("a"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<i32>(
                Some("b"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<i32>(
                Some("out"),
                0,
                2,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
            ),
        ]));
        let index = if extent.is_some() {
            PcuDispatchIndex::GridStrideId
        } else {
            PcuDispatchIndex::InvocationId
        };
        let body = Box::leak(Box::new([
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(2),
                binding: PcuBindingRef::new(0, 1),
                index,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: crate::PcuValueType::i32(),
                result: PcuDispatchValueId(3),
                op: alu,
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index,
                value: PcuDispatchValueId(3),
            }),
        ]));
        let ops = match extent {
            Some(extent) => Box::leak(Box::new([
                PcuDispatchOp::GridStrideLoop { extent, body },
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ])) as &'static [PcuDispatchOp<'static>],
            None => Box::leak(Box::new([
                body[0],
                body[1],
                body[2],
                body[3],
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ])),
        };
        PcuDispatchKernelIr {
            id: PcuKernelId(1),
            entry: PcuDispatchEntryPoint {
                name: "map",
                logical_shape: [1, 1, 1],
            },
            bindings,
            ports: &[],
            parameters: &[],
            ops,
            type_caps: PcuValueTypeCaps::for_scalar(PcuScalarType::I32),
            feature_caps: crate::PcuDispatchFeatureCaps::default(),
        }
    }

    #[test]
    fn admits_wrapping_arithmetic_and_large_grid_extent_but_not_division() {
        for op in [
            PcuDispatchAluOp::Add,
            PcuDispatchAluOp::Sub,
            PcuDispatchAluOp::Mul,
        ] {
            assert_eq!(validate_i32_map_kernel(&kernel(None, op)), Ok(()));
            assert_eq!(validate_i32_map_kernel(&kernel(Some(u32::MAX), op)), Ok(()));
        }
        assert!(matches!(
            validate_i32_map_kernel(&kernel(None, PcuDispatchAluOp::Div)),
            Err(PcuI32MapValidationError::UnsupportedOperation(2))
        ));
    }

    fn chained_kernel(extent: Option<u32>) -> PcuDispatchKernelIr<'static> {
        let mut kernel = kernel(extent, PcuDispatchAluOp::Add);
        let index = if extent.is_some() {
            PcuDispatchIndex::GridStrideId
        } else {
            PcuDispatchIndex::InvocationId
        };
        let body = Box::leak(Box::new([
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(2),
                binding: PcuBindingRef::new(0, 1),
                index,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: crate::PcuValueType::i32(),
                result: PcuDispatchValueId(3),
                op: PcuDispatchAluOp::Add,
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: crate::PcuValueType::i32(),
                result: PcuDispatchValueId(4),
                op: PcuDispatchAluOp::Mul,
                lhs: PcuDispatchValueId(3),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index,
                value: PcuDispatchValueId(4),
            }),
        ]));
        kernel.ops = match extent {
            Some(extent) => Box::leak(Box::new([
                PcuDispatchOp::GridStrideLoop { extent, body },
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ])),
            None => Box::leak(Box::new([
                body[0],
                body[1],
                body[2],
                body[3],
                body[4],
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ])),
        };
        kernel
    }

    #[test]
    fn admits_chained_wrapping_alu_direct_and_grid_stride() {
        assert_eq!(validate_i32_map_kernel(&chained_kernel(None)), Ok(()));
        assert_eq!(validate_i32_map_kernel(&chained_kernel(Some(100))), Ok(()));
    }

    fn checked_div_rem_kernel(extent: Option<u32>) -> PcuDispatchKernelIr<'static> {
        let bindings = Box::leak(Box::new([
            PcuBinding::scalar::<i32>(
                Some("a"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<i32>(
                Some("b"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<i32>(
                Some("q"),
                0,
                2,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
            ),
            PcuBinding::scalar::<i32>(
                Some("r"),
                0,
                3,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
            ),
        ]));
        let index = if extent.is_some() {
            PcuDispatchIndex::GridStrideId
        } else {
            PcuDispatchIndex::InvocationId
        };
        let body = Box::leak(Box::new([
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(2),
                binding: PcuBindingRef::new(0, 1),
                index,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem {
                value_type: PcuValueType::i32(),
                flags: crate::model::PcuIntegerDivFlags::CHECKED,
                quotient: PcuDispatchValueId(3),
                remainder: PcuDispatchValueId(4),
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index,
                value: PcuDispatchValueId(3),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 3),
                index,
                value: PcuDispatchValueId(4),
            }),
        ]));
        let ops = match extent {
            Some(extent) => Box::leak(Box::new([
                PcuDispatchOp::GridStrideLoop { extent, body },
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ])) as &'static [PcuDispatchOp<'static>],
            None => Box::leak(Box::new([
                body[0],
                body[1],
                body[2],
                body[3],
                body[4],
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ])),
        };
        PcuDispatchKernelIr {
            id: PcuKernelId(1),
            entry: PcuDispatchEntryPoint {
                name: "divrem_i32",
                logical_shape: [1, 1, 1],
            },
            bindings,
            ports: &[],
            parameters: &[],
            ops,
            type_caps: PcuValueTypeCaps::for_scalar(PcuScalarType::I32),
            feature_caps: crate::PcuDispatchFeatureCaps::default(),
        }
    }

    #[test]
    fn admits_checked_i32_divrem_direct_and_grid_stride_only() {
        assert_eq!(
            validate_i32_checked_div_rem_kernel(&checked_div_rem_kernel(None)),
            Ok(())
        );
        assert_eq!(
            validate_i32_checked_div_rem_kernel(&checked_div_rem_kernel(Some(19))),
            Ok(())
        );
        assert!(matches!(
            validate_i32_checked_div_rem_kernel(&checked_div_rem_kernel(Some(0))),
            Err(PcuI32MapValidationError::UnsupportedOperation(0))
        ));
    }
}
