//! Bounded type checking for scalar dispatch data-flow regions.

#[rustfmt::skip]
use crate::{
    PcuBindingRef,
    PcuBindingType,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchValueId,
    PcuValueType,
};

const VALUE_LIMIT: usize = 256;

/// First type or value-flow error found in the bounded scalar Dispatch profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuTypedDispatchValidationError {
    UnsupportedOperation(usize),
    ValueOutOfRange(PcuDispatchValueId),
    DuplicateValue(PcuDispatchValueId),
    UndefinedValue(PcuDispatchValueId),
    MissingBinding(PcuBindingRef),
    NonValueBinding(PcuBindingRef),
    TypeMismatch {
        expected: PcuValueType,
        actual: PcuValueType,
    },
}

/// Type-check scalar value flow for direct instructions and one bounded grid-stride region.
///
/// The accepted subset is `BindingLoad`, scalar `Alu`, `CheckedDivRem`, the explicitly defined
/// integer widening and f32/half conversions, checked F64-to-F32 conversion, `BindingStore` and `Return`. Values are SSA scoped
/// to their region. The verifier deliberately rejects vector/matrix values and unlisted casts.
///
/// # Errors
///
/// Returns the first unsupported instruction, out-of-range/duplicate/undefined value, invalid
/// binding, or disagreement between an inferred and required value type.
pub fn validate_typed_dispatch_value_flow(
    kernel: &PcuDispatchKernelIr<'_>,
) -> Result<(), PcuTypedDispatchValidationError> {
    if let [
        PcuDispatchOp::GridStrideLoop { body, extent },
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ] = kernel.ops
    {
        if *extent == 0 {
            return Err(PcuTypedDispatchValidationError::UnsupportedOperation(0));
        }
        validate_region(kernel, body, true)
    } else {
        validate_region(kernel, kernel.ops, false)
    }
}

fn validate_region(
    kernel: &PcuDispatchKernelIr<'_>,
    ops: &[PcuDispatchOp<'_>],
    in_grid: bool,
) -> Result<(), PcuTypedDispatchValidationError> {
    let mut types: [Option<PcuValueType>; VALUE_LIMIT] = [None; VALUE_LIMIT];
    for (position, op) in ops.iter().copied().enumerate() {
        match op {
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad { .. }) => {
                validate_load(kernel, &mut types, op, position, in_grid)?;
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore { .. }) => {
                validate_store(kernel, &types, op)?;
            }
            PcuDispatchOp::Data(data) => validate_computation(&mut types, data, position)?,
            PcuDispatchOp::Control(PcuDispatchControlOp::Return) if !in_grid => {}
            _ => {
                return Err(PcuTypedDispatchValidationError::UnsupportedOperation(
                    position,
                ));
            }
        }
    }
    Ok(())
}

fn validate_load(
    kernel: &PcuDispatchKernelIr<'_>,
    types: &mut [Option<PcuValueType>; VALUE_LIMIT],
    op: PcuDispatchOp<'_>,
    position: usize,
    in_grid: bool,
) -> Result<(), PcuTypedDispatchValidationError> {
    let PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
        result,
        binding,
        index,
    }) = op
    else {
        return Err(PcuTypedDispatchValidationError::UnsupportedOperation(
            position,
        ));
    };
    if !in_grid && matches!(index, PcuDispatchIndex::GridStrideId) {
        return Err(PcuTypedDispatchValidationError::UnsupportedOperation(
            position,
        ));
    }
    let binding = kernel
        .bindings
        .iter()
        .find(|item| item.reference() == binding)
        .ok_or(PcuTypedDispatchValidationError::MissingBinding(binding))?;
    let PcuBindingType::Value(value_type) = binding.binding_type else {
        return Err(PcuTypedDispatchValidationError::NonValueBinding(
            binding.reference(),
        ));
    };
    if !matches!(value_type, PcuValueType::Scalar(_)) {
        return Err(PcuTypedDispatchValidationError::UnsupportedOperation(
            position,
        ));
    }
    define(types, result, value_type)
}

fn validate_store(
    kernel: &PcuDispatchKernelIr<'_>,
    types: &[Option<PcuValueType>; VALUE_LIMIT],
    op: PcuDispatchOp<'_>,
) -> Result<(), PcuTypedDispatchValidationError> {
    let PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore { binding, value, .. }) = op else {
        unreachable!();
    };
    let target = kernel
        .bindings
        .iter()
        .find(|item| item.reference() == binding)
        .ok_or(PcuTypedDispatchValidationError::MissingBinding(binding))?;
    let PcuBindingType::Value(expected) = target.binding_type else {
        return Err(PcuTypedDispatchValidationError::NonValueBinding(binding));
    };
    check_type(expected, require(types, value)?)
}

fn validate_computation(
    types: &mut [Option<PcuValueType>; VALUE_LIMIT],
    data: PcuDispatchDataOp,
    position: usize,
) -> Result<(), PcuTypedDispatchValidationError> {
    match data {
        PcuDispatchDataOp::Constant { result, value } => define(types, result, value.value_type()),
        PcuDispatchDataOp::Convert {
            result,
            value,
            conversion,
        } => {
            check_type(conversion.source_type(), require(types, value)?)?;
            define(types, result, conversion.target_type())
        }
        PcuDispatchDataOp::CheckedFloatConvert {
            conversion,
            result,
            value,
            ..
        } => {
            check_type(conversion.source_type(), require(types, value)?)?;
            define(types, result, conversion.target_type())
        }
        PcuDispatchDataOp::Alu {
            value_type,
            result,
            lhs,
            rhs,
            ..
        } => {
            if !matches!(value_type, PcuValueType::Scalar(_)) {
                return Err(PcuTypedDispatchValidationError::UnsupportedOperation(
                    position,
                ));
            }
            check_type(value_type, require(types, lhs)?)?;
            check_type(value_type, require(types, rhs)?)?;
            define(types, result, value_type)
        }
        PcuDispatchDataOp::CheckedDivRem {
            value_type,
            quotient,
            remainder,
            lhs,
            rhs,
            ..
        } => {
            check_type(value_type, require(types, lhs)?)?;
            check_type(value_type, require(types, rhs)?)?;
            define(types, quotient, value_type)?;
            define(types, remainder, value_type)
        }
        PcuDispatchDataOp::CheckedIntegerBinary {
            value_type,
            result,
            lhs,
            rhs,
            ..
        } => {
            if !is_supported_checked_integer(value_type) {
                return Err(PcuTypedDispatchValidationError::UnsupportedOperation(
                    position,
                ));
            }
            check_type(value_type, require(types, lhs)?)?;
            check_type(value_type, require(types, rhs)?)?;
            define(types, result, value_type)
        }
        PcuDispatchDataOp::CheckedFloatBinary {
            value_type,
            result,
            lhs,
            rhs,
            ..
        } => {
            if !is_supported_checked_float(value_type) {
                return Err(PcuTypedDispatchValidationError::UnsupportedOperation(
                    position,
                ));
            }
            check_type(value_type, require(types, lhs)?)?;
            check_type(value_type, require(types, rhs)?)?;
            define(types, result, value_type)
        }
        PcuDispatchDataOp::CheckedFloatUnary {
            value_type,
            result,
            value,
            ..
        } => {
            if !is_supported_checked_float(value_type) {
                return Err(PcuTypedDispatchValidationError::UnsupportedOperation(
                    position,
                ));
            }
            check_type(value_type, require(types, value)?)?;
            define(types, result, value_type)
        }
        _ => Err(PcuTypedDispatchValidationError::UnsupportedOperation(
            position,
        )),
    }
}

const fn is_supported_checked_float(value_type: PcuValueType) -> bool {
    matches!(
        value_type,
        PcuValueType::Scalar(
            crate::PcuScalarType::F16
                | crate::PcuScalarType::BF16
                | crate::PcuScalarType::F32
                | crate::PcuScalarType::F64
        )
    )
}

const fn is_supported_checked_integer(value_type: PcuValueType) -> bool {
    matches!(
        value_type,
        PcuValueType::Scalar(
            crate::PcuScalarType::I8
                | crate::PcuScalarType::U8
                | crate::PcuScalarType::I16
                | crate::PcuScalarType::U16
                | crate::PcuScalarType::I32
                | crate::PcuScalarType::U32
                | crate::PcuScalarType::I64
                | crate::PcuScalarType::U64
        )
    )
}

fn slot(value: PcuDispatchValueId) -> Result<usize, PcuTypedDispatchValidationError> {
    let index = usize::from(value.0);
    if index < VALUE_LIMIT {
        Ok(index)
    } else {
        Err(PcuTypedDispatchValidationError::ValueOutOfRange(value))
    }
}

fn define(
    types: &mut [Option<PcuValueType>; VALUE_LIMIT],
    value: PcuDispatchValueId,
    value_type: PcuValueType,
) -> Result<(), PcuTypedDispatchValidationError> {
    let index = slot(value)?;
    if types[index].is_some() {
        return Err(PcuTypedDispatchValidationError::DuplicateValue(value));
    }
    types[index] = Some(value_type);
    Ok(())
}

fn require(
    types: &[Option<PcuValueType>; VALUE_LIMIT],
    value: PcuDispatchValueId,
) -> Result<PcuValueType, PcuTypedDispatchValidationError> {
    types[slot(value)?].ok_or(PcuTypedDispatchValidationError::UndefinedValue(value))
}

fn check_type(
    expected: PcuValueType,
    actual: PcuValueType,
) -> Result<(), PcuTypedDispatchValidationError> {
    if expected == actual {
        Ok(())
    } else {
        Err(PcuTypedDispatchValidationError::TypeMismatch { expected, actual })
    }
}

#[cfg(test)]
mod tests {
    #[rustfmt::skip]
    use super::{
        validate_typed_dispatch_value_flow,
        PcuTypedDispatchValidationError as Error,
    };
    #[rustfmt::skip]
    use crate::{
        PcuBinding,
        PcuBindingAccess,
        PcuBindingRef,
        PcuBindingStorageClass,
        PcuDispatchConversion,
        PcuDispatchCheckedFloatConversion,
        PcuDispatchDataOp as Data,
        PcuDispatchEntryPoint,
        PcuDispatchFeatureCaps,
        PcuDispatchIndex,
        PcuDispatchKernelIr,
        PcuDispatchOp as Op,
        PcuDispatchValueId as Id,
        PcuKernelId,
        PcuValueType,
        PcuValueTypeCaps,
    };

    fn kernel<'a>(bindings: &'a [PcuBinding<'a>], ops: &'a [Op<'a>]) -> PcuDispatchKernelIr<'a> {
        PcuDispatchKernelIr {
            id: PcuKernelId(1),
            entry: PcuDispatchEntryPoint {
                name: "typed",
                logical_shape: [1, 1, 1],
            },
            bindings,
            ports: &[],
            parameters: &[],
            ops,
            type_caps: PcuValueTypeCaps::empty(),
            feature_caps: PcuDispatchFeatureCaps::empty(),
        }
    }

    fn binding(
        slot: u32,
        scalar: crate::PcuScalarType,
        access: PcuBindingAccess,
    ) -> PcuBinding<'static> {
        PcuBinding::value(
            Some("buffer"),
            0,
            slot,
            PcuBindingStorageClass::Storage,
            access,
            PcuValueType::Scalar(scalar),
        )
    }

    #[test]
    fn admits_only_exact_signed_and_unsigned_widening_value_flow() {
        for (source, target, conversion) in [
            (
                crate::PcuScalarType::I8,
                crate::PcuScalarType::I16,
                PcuDispatchConversion::I8ToI16,
            ),
            (
                crate::PcuScalarType::U8,
                crate::PcuScalarType::U16,
                PcuDispatchConversion::U8ToU16,
            ),
            (
                crate::PcuScalarType::I16,
                crate::PcuScalarType::I32,
                PcuDispatchConversion::I16ToI32,
            ),
            (
                crate::PcuScalarType::U16,
                crate::PcuScalarType::U32,
                PcuDispatchConversion::U16ToU32,
            ),
            (
                crate::PcuScalarType::I32,
                crate::PcuScalarType::I64,
                PcuDispatchConversion::I32ToI64,
            ),
            (
                crate::PcuScalarType::U32,
                crate::PcuScalarType::U64,
                PcuDispatchConversion::U32ToU64,
            ),
        ] {
            let bindings = [
                binding(0, source, PcuBindingAccess::ReadOnly),
                binding(1, target, PcuBindingAccess::WriteOnly),
            ];
            let ops = [
                Op::Data(Data::BindingLoad {
                    result: Id(0),
                    binding: PcuBindingRef::new(0, 0),
                    index: PcuDispatchIndex::InvocationId,
                }),
                Op::Data(Data::Convert {
                    result: Id(1),
                    value: Id(0),
                    conversion,
                }),
                Op::Data(Data::BindingStore {
                    binding: PcuBindingRef::new(0, 1),
                    index: PcuDispatchIndex::InvocationId,
                    value: Id(1),
                }),
                Op::Control(crate::PcuDispatchControlOp::Return),
            ];
            assert_eq!(
                validate_typed_dispatch_value_flow(&kernel(&bindings, &ops)),
                Ok(())
            );
        }
    }

    #[test]
    fn half_conversion_signatures_are_exact_and_admitted_by_value_flow() {
        for (conversion, source, target) in [
            (
                PcuDispatchConversion::F32ToF16Bits,
                crate::PcuScalarType::F32,
                crate::PcuScalarType::F16,
            ),
            (
                PcuDispatchConversion::F16BitsToF32,
                crate::PcuScalarType::F16,
                crate::PcuScalarType::F32,
            ),
            (
                PcuDispatchConversion::F32ToBf16Bits,
                crate::PcuScalarType::F32,
                crate::PcuScalarType::BF16,
            ),
            (
                PcuDispatchConversion::Bf16BitsToF32,
                crate::PcuScalarType::BF16,
                crate::PcuScalarType::F32,
            ),
        ] {
            assert_eq!(conversion.source_type(), PcuValueType::Scalar(source));
            assert_eq!(conversion.target_type(), PcuValueType::Scalar(target));
            let bindings = [
                binding(0, source, PcuBindingAccess::ReadOnly),
                binding(1, target, PcuBindingAccess::WriteOnly),
            ];
            let ops = [
                Op::Data(Data::BindingLoad {
                    result: Id(0),
                    binding: PcuBindingRef::new(0, 0),
                    index: PcuDispatchIndex::InvocationId,
                }),
                Op::Data(Data::Convert {
                    result: Id(1),
                    value: Id(0),
                    conversion,
                }),
                Op::Data(Data::BindingStore {
                    binding: PcuBindingRef::new(0, 1),
                    index: PcuDispatchIndex::InvocationId,
                    value: Id(1),
                }),
                Op::Control(crate::PcuDispatchControlOp::Return),
            ];
            assert_eq!(
                validate_typed_dispatch_value_flow(&kernel(&bindings, &ops)),
                Ok(())
            );
        }
    }

    #[test]
    fn rejects_conversion_source_and_store_type_mismatches() {
        let bindings = [
            binding(0, crate::PcuScalarType::U8, PcuBindingAccess::ReadOnly),
            binding(1, crate::PcuScalarType::I16, PcuBindingAccess::WriteOnly),
        ];
        let ops = [
            Op::Data(Data::BindingLoad {
                result: Id(0),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::InvocationId,
            }),
            Op::Data(Data::Convert {
                result: Id(1),
                value: Id(0),
                conversion: PcuDispatchConversion::I8ToI16,
            }),
        ];
        assert!(matches!(
            validate_typed_dispatch_value_flow(&kernel(&bindings, &ops)),
            Err(Error::TypeMismatch { .. })
        ));

        let ops = [
            Op::Data(Data::BindingLoad {
                result: Id(0),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::InvocationId,
            }),
            Op::Data(Data::Convert {
                result: Id(1),
                value: Id(0),
                conversion: PcuDispatchConversion::U8ToU16,
            }),
            Op::Data(Data::BindingStore {
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::InvocationId,
                value: Id(1),
            }),
        ];
        assert!(matches!(
            validate_typed_dispatch_value_flow(&kernel(&bindings, &ops)),
            Err(Error::TypeMismatch { .. })
        ));
    }

    #[test]
    fn rejects_use_before_definition_and_values_beyond_bounded_ssa_table() {
        let ops = [Op::Data(Data::Convert {
            result: Id(1),
            value: Id(0),
            conversion: PcuDispatchConversion::I8ToI16,
        })];
        assert_eq!(
            validate_typed_dispatch_value_flow(&kernel(&[], &ops)),
            Err(Error::UndefinedValue(Id(0)))
        );
        let ops = [Op::Data(Data::Convert {
            result: Id(1),
            value: Id(256),
            conversion: PcuDispatchConversion::I8ToI16,
        })];
        assert_eq!(
            validate_typed_dispatch_value_flow(&kernel(&[], &ops)),
            Err(Error::ValueOutOfRange(Id(256)))
        );
    }

    #[test]
    fn checked_integer_binary_requires_exact_integer_types_and_valid_ssa() {
        let bindings = [
            binding(0, crate::PcuScalarType::I32, PcuBindingAccess::ReadOnly),
            binding(1, crate::PcuScalarType::I32, PcuBindingAccess::ReadOnly),
            binding(2, crate::PcuScalarType::I32, PcuBindingAccess::WriteOnly),
        ];
        let load_a = Op::Data(Data::BindingLoad {
            result: Id(1),
            binding: PcuBindingRef::new(0, 0),
            index: PcuDispatchIndex::InvocationId,
        });
        let load_b = Op::Data(Data::BindingLoad {
            result: Id(2),
            binding: PcuBindingRef::new(0, 1),
            index: PcuDispatchIndex::InvocationId,
        });
        let checked = Op::Data(Data::CheckedIntegerBinary {
            value_type: PcuValueType::i32(),
            op: crate::model::PcuDispatchIntegerBinaryOp::Add,
            result: Id(3),
            lhs: Id(1),
            rhs: Id(2),
        });
        assert_eq!(
            validate_typed_dispatch_value_flow(&kernel(&bindings, &[load_a, load_b, checked])),
            Ok(())
        );

        let bad_type = Op::Data(Data::CheckedIntegerBinary {
            value_type: PcuValueType::f32(),
            op: crate::model::PcuDispatchIntegerBinaryOp::Mul,
            result: Id(3),
            lhs: Id(1),
            rhs: Id(2),
        });
        assert_eq!(
            validate_typed_dispatch_value_flow(&kernel(&bindings, &[load_a, load_b, bad_type])),
            Err(Error::UnsupportedOperation(2))
        );
        let duplicate = Op::Data(Data::CheckedIntegerBinary {
            value_type: PcuValueType::i32(),
            op: crate::model::PcuDispatchIntegerBinaryOp::Sub,
            result: Id(1),
            lhs: Id(1),
            rhs: Id(2),
        });
        assert_eq!(
            validate_typed_dispatch_value_flow(&kernel(&bindings, &[load_a, load_b, duplicate])),
            Err(Error::DuplicateValue(Id(1)))
        );
        let undefined = Op::Data(Data::CheckedIntegerBinary {
            value_type: PcuValueType::i32(),
            op: crate::model::PcuDispatchIntegerBinaryOp::Sub,
            result: Id(3),
            lhs: Id(99),
            rhs: Id(2),
        });
        assert_eq!(
            validate_typed_dispatch_value_flow(&kernel(&bindings, &[load_a, load_b, undefined])),
            Err(Error::UndefinedValue(Id(99)))
        );
    }

    #[test]
    fn checked_float_binary_ssa_preserves_homogeneous_f64_type() {
        let bindings = [
            binding(0, crate::PcuScalarType::F64, PcuBindingAccess::ReadOnly),
            binding(1, crate::PcuScalarType::F64, PcuBindingAccess::ReadOnly),
            binding(2, crate::PcuScalarType::F64, PcuBindingAccess::WriteOnly),
        ];
        let ops = [
            Op::Data(Data::BindingLoad {
                result: Id(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::InvocationId,
            }),
            Op::Data(Data::BindingLoad {
                result: Id(2),
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::InvocationId,
            }),
            Op::Data(Data::CheckedFloatBinary {
                value_type: PcuValueType::f64(),
                op: crate::PcuDispatchFloatBinaryOp::Mul,
                underflow_policy: crate::PcuFloatUnderflowPolicy::IeeeAfterRounding,
                range_policy: crate::PcuRangePolicy::Reject,
                result: Id(3),
                lhs: Id(1),
                rhs: Id(2),
            }),
            Op::Data(Data::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index: PcuDispatchIndex::InvocationId,
                value: Id(3),
            }),
        ];
        assert_eq!(
            validate_typed_dispatch_value_flow(&kernel(&bindings, &ops)),
            Ok(())
        );
    }

    #[test]
    fn checked_f64_to_f32_conversion_checks_input_output_types_and_ssa() {
        let bindings = [
            binding(0, crate::PcuScalarType::F64, PcuBindingAccess::ReadOnly),
            binding(1, crate::PcuScalarType::F32, PcuBindingAccess::WriteOnly),
        ];
        let load = Op::Data(Data::BindingLoad {
            result: Id(1),
            binding: PcuBindingRef::new(0, 0),
            index: PcuDispatchIndex::InvocationId,
        });
        let convert = Op::Data(Data::CheckedFloatConvert {
            conversion: PcuDispatchCheckedFloatConversion::F64ToF32,
            underflow_policy: crate::PcuFloatUnderflowPolicy::IeeeAfterRounding,
            range_policy: crate::PcuRangePolicy::Reject,
            result: Id(2),
            value: Id(1),
        });
        let store = Op::Data(Data::BindingStore {
            binding: PcuBindingRef::new(0, 1),
            index: PcuDispatchIndex::InvocationId,
            value: Id(2),
        });
        assert_eq!(
            validate_typed_dispatch_value_flow(&kernel(&bindings, &[load, convert, store])),
            Ok(())
        );

        let missing = Op::Data(Data::CheckedFloatConvert {
            conversion: PcuDispatchCheckedFloatConversion::F64ToF32,
            underflow_policy: crate::PcuFloatUnderflowPolicy::IeeeAfterRounding,
            range_policy: crate::PcuRangePolicy::Reject,
            result: Id(2),
            value: Id(99),
        });
        assert_eq!(
            validate_typed_dispatch_value_flow(&kernel(&[], &[missing])),
            Err(Error::UndefinedValue(Id(99)))
        );

        let wrong_input = [
            Op::Data(Data::BindingLoad {
                result: Id(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::InvocationId,
            }),
            convert,
        ];
        let wrong_bindings = [binding(
            0,
            crate::PcuScalarType::F32,
            PcuBindingAccess::ReadOnly,
        )];
        assert!(matches!(
            validate_typed_dispatch_value_flow(&kernel(&wrong_bindings, &wrong_input)),
            Err(Error::TypeMismatch { .. })
        ));

        let wrong_output = [
            binding(0, crate::PcuScalarType::F64, PcuBindingAccess::ReadOnly),
            binding(1, crate::PcuScalarType::F64, PcuBindingAccess::WriteOnly),
        ];
        assert!(matches!(
            validate_typed_dispatch_value_flow(&kernel(&wrong_output, &[load, convert, store])),
            Err(Error::TypeMismatch { .. })
        ));
    }

    #[test]
    fn checked_f32_to_f64_widening_checks_mixed_width_ssa() {
        let bindings = [
            binding(0, crate::PcuScalarType::F32, PcuBindingAccess::ReadOnly),
            binding(1, crate::PcuScalarType::F64, PcuBindingAccess::WriteOnly),
        ];
        let ops = [
            Op::Data(Data::BindingLoad {
                result: Id(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::InvocationId,
            }),
            Op::Data(Data::CheckedFloatConvert {
                conversion: PcuDispatchCheckedFloatConversion::F32ToF64,
                underflow_policy: crate::PcuFloatUnderflowPolicy::RejectSubnormalResult,
                range_policy: crate::PcuRangePolicy::Reject,
                result: Id(2),
                value: Id(1),
            }),
            Op::Data(Data::BindingStore {
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::InvocationId,
                value: Id(2),
            }),
        ];
        assert_eq!(
            validate_typed_dispatch_value_flow(&kernel(&bindings, &ops)),
            Ok(())
        );
        let bad = [
            ops[0],
            Op::Data(Data::CheckedFloatConvert {
                conversion: PcuDispatchCheckedFloatConversion::F32ToF64,
                underflow_policy: crate::PcuFloatUnderflowPolicy::IeeeAfterRounding,
                range_policy: crate::PcuRangePolicy::Reject,
                result: Id(2),
                value: Id(99),
            }),
        ];
        assert_eq!(
            validate_typed_dispatch_value_flow(&kernel(&bindings, &bad)),
            Err(Error::UndefinedValue(Id(99)))
        );
    }
}
