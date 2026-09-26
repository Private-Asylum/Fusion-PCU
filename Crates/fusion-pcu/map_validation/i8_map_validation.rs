//! Structural admission for the bounded scalar `i8` arithmetic map profile.

use crate::{
    PcuBindingRef,
    PcuDispatchKernelIr,
    PcuDispatchValueId,
    PcuValueType,
    PcuValueTypeCaps,
};
use crate::map_validation::integer_map_validation::{
    validate_integer_map_kernel,
    IntegerMapValidationError,
};

/// Structural failure within the typed scalar `i8` arithmetic map profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuI8MapValidationError {
    UnsupportedInterface,
    UnsupportedRequirements,
    InvalidBinding(PcuBindingRef),
    DuplicateBinding(PcuBindingRef),
    UnsupportedOperation(usize),
    InvalidIndex(usize),
    InvalidValue(PcuDispatchValueId),
    MissingReturn,
}

const fn map_error(error: IntegerMapValidationError) -> PcuI8MapValidationError {
    match error {
        IntegerMapValidationError::UnsupportedInterface => {
            PcuI8MapValidationError::UnsupportedInterface
        }
        IntegerMapValidationError::UnsupportedRequirements => {
            PcuI8MapValidationError::UnsupportedRequirements
        }
        IntegerMapValidationError::InvalidBinding(value) => {
            PcuI8MapValidationError::InvalidBinding(value)
        }
        IntegerMapValidationError::DuplicateBinding(value) => {
            PcuI8MapValidationError::DuplicateBinding(value)
        }
        IntegerMapValidationError::UnsupportedOperation(value) => {
            PcuI8MapValidationError::UnsupportedOperation(value)
        }
        IntegerMapValidationError::InvalidIndex(value) => {
            PcuI8MapValidationError::InvalidIndex(value)
        }
        IntegerMapValidationError::InvalidValue(value) => {
            PcuI8MapValidationError::InvalidValue(value)
        }
        IntegerMapValidationError::MissingReturn => PcuI8MapValidationError::MissingReturn,
    }
}

/// Validates indexed `i8` loads, one or more wrapping Add/Sub/Mul operations, one store and Return.
///
/// The exact operation sequence is required. It may be direct (`InvocationId`) or one bounded
/// `GridStrideLoop` (`GridStrideId`) followed by Return. Binding coverage and extents are checked
/// by the dispatch ABI. `i8` division and remainder are outside this profile.
///
/// # Errors
///
/// Returns the first structural violation of the typed `i8` map profile.
pub fn validate_i8_map_kernel(
    kernel: &PcuDispatchKernelIr<'_>,
) -> Result<(), PcuI8MapValidationError> {
    validate_integer_map_kernel(kernel, PcuValueType::i8(), PcuValueTypeCaps::INT8)
        .map_err(map_error)
}

#[cfg(test)]
mod tests {
    use std::boxed::Box;
    use super::{
        validate_i8_map_kernel,
        PcuI8MapValidationError,
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
        PcuValueTypeCaps,
    };

    fn kernel(extent: Option<u32>, alu: PcuDispatchAluOp) -> PcuDispatchKernelIr<'static> {
        let bindings = Box::leak(Box::new([
            PcuBinding::scalar::<i8>(
                Some("a"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<i8>(
                Some("b"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<i8>(
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
                value_type: crate::PcuValueType::i8(),
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
            type_caps: PcuValueTypeCaps::for_scalar(PcuScalarType::I8),
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
            assert_eq!(validate_i8_map_kernel(&kernel(None, op)), Ok(()));
            assert_eq!(validate_i8_map_kernel(&kernel(Some(u32::MAX), op)), Ok(()));
        }
        assert!(matches!(
            validate_i8_map_kernel(&kernel(None, PcuDispatchAluOp::Div)),
            Err(PcuI8MapValidationError::UnsupportedOperation(2))
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
                value_type: crate::PcuValueType::i8(),
                result: PcuDispatchValueId(3),
                op: PcuDispatchAluOp::Add,
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: crate::PcuValueType::i8(),
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
        assert_eq!(validate_i8_map_kernel(&chained_kernel(None)), Ok(()));
        assert_eq!(validate_i8_map_kernel(&chained_kernel(Some(100))), Ok(()));
    }
}
