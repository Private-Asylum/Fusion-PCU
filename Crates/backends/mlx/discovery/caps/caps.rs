//! Coarse capability floor; exact bounded structure is admitted during preparation.

#[rustfmt::skip]
use fusion_pcu::{
    PcuCaps,
    PcuDispatchFeatureCaps,
    PcuDispatchOpCaps,
    PcuDispatchPolicyCaps,
    PcuDispatchScalarAluSupport,
    PcuDispatchSupport,
    PcuExecutorClass,
    PcuExecutorDescriptor,
    PcuExecutorOrigin,
    PcuExecutorSupport,
    PcuFeatureSupport,
    PcuImplementationKind,
    PcuPrimitiveCaps,
    PcuPrimitiveSupport,
    PcuScalarType,
    PcuSupport,
    PcuValueTypeCaps,
};
// Value transport is broader than the independently admitted arithmetic table.
const TYPES: PcuValueTypeCaps = PcuValueTypeCaps::for_scalar(PcuScalarType::U8)
    .union(PcuValueTypeCaps::for_scalar(PcuScalarType::I8))
    .union(PcuValueTypeCaps::for_scalar(PcuScalarType::U16))
    .union(PcuValueTypeCaps::for_scalar(PcuScalarType::I16))
    .union(PcuValueTypeCaps::for_scalar(PcuScalarType::U32))
    .union(PcuValueTypeCaps::for_scalar(PcuScalarType::I32))
    .union(PcuValueTypeCaps::for_scalar(PcuScalarType::U64))
    .union(PcuValueTypeCaps::for_scalar(PcuScalarType::I64))
    .union(PcuValueTypeCaps::for_scalar(PcuScalarType::U128))
    .union(PcuValueTypeCaps::for_scalar(PcuScalarType::I128))
    .union(PcuValueTypeCaps::for_scalar(PcuScalarType::U256))
    .union(PcuValueTypeCaps::for_scalar(PcuScalarType::I256))
    .union(PcuValueTypeCaps::for_scalar(PcuScalarType::U512))
    .union(PcuValueTypeCaps::for_scalar(PcuScalarType::I512))
    .union(PcuValueTypeCaps::for_scalar(PcuScalarType::F16))
    .union(PcuValueTypeCaps::for_scalar(PcuScalarType::BF16))
    .union(PcuValueTypeCaps::for_scalar(PcuScalarType::F8E4M3FN))
    .union(PcuValueTypeCaps::for_scalar(PcuScalarType::F8E5M2))
    .union(PcuValueTypeCaps::for_scalar(PcuScalarType::F32))
    .union(PcuValueTypeCaps::for_scalar(PcuScalarType::F64))
    .union(PcuValueTypeCaps::for_scalar(PcuScalarType::F128))
    .union(PcuValueTypeCaps::for_scalar(PcuScalarType::F256))
    .union(PcuValueTypeCaps::SCALAR_VALUES);
const INSTRUCTIONS: PcuDispatchOpCaps = PcuDispatchOpCaps::ALU_CHECKED_FLOAT_UNARY
    .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_CONVERT)
    .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY)
    .union(PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY)
    .union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM)
    .union(PcuDispatchOpCaps::BINDING_LOAD)
    .union(PcuDispatchOpCaps::BINDING_LOAD_ELEMENT_ZERO)
    .union(PcuDispatchOpCaps::BINDING_STORE)
    .union(PcuDispatchOpCaps::CONTROL_RETURN)
    .union(PcuDispatchOpCaps::CONTROL_LOOP);
const SCALAR: PcuDispatchScalarAluSupport = PcuDispatchScalarAluSupport::empty()
    .with(
        PcuScalarType::U8,
        PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY.union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM),
    )
    .with(
        PcuScalarType::I8,
        PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY.union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM),
    )
    .with(
        PcuScalarType::U16,
        PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY.union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM),
    )
    .with(
        PcuScalarType::I16,
        PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY.union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM),
    )
    .with(
        PcuScalarType::U32,
        PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY.union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM),
    )
    .with(
        PcuScalarType::I32,
        PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY.union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM),
    )
    .with(
        PcuScalarType::U64,
        PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY.union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM),
    )
    .with(
        PcuScalarType::I64,
        PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY.union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM),
    )
    .with(
        PcuScalarType::U128,
        PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY.union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM),
    )
    .with(
        PcuScalarType::I128,
        PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY.union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM),
    )
    .with(
        PcuScalarType::U256,
        PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY.union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM),
    )
    .with(
        PcuScalarType::I256,
        PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY.union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM),
    )
    .with(
        PcuScalarType::U512,
        PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY.union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM),
    )
    .with(
        PcuScalarType::I512,
        PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY.union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM),
    )
    .with(
        PcuScalarType::F16,
        PcuDispatchOpCaps::ALU_CHECKED_FLOAT_UNARY
            .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY),
    )
    .with(
        PcuScalarType::BF16,
        PcuDispatchOpCaps::ALU_CHECKED_FLOAT_UNARY
            .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY),
    )
    .with(
        PcuScalarType::F8E4M3FN,
        PcuDispatchOpCaps::ALU_CHECKED_FLOAT_UNARY
            .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY),
    )
    .with(
        PcuScalarType::F8E5M2,
        PcuDispatchOpCaps::ALU_CHECKED_FLOAT_UNARY
            .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY),
    )
    .with(
        PcuScalarType::F32,
        PcuDispatchOpCaps::ALU_CHECKED_FLOAT_UNARY
            .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY)
            .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_CONVERT),
    )
    .with(
        PcuScalarType::F64,
        PcuDispatchOpCaps::ALU_CHECKED_FLOAT_UNARY
            .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY)
            .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_CONVERT),
    );
const POLICY: PcuDispatchPolicyCaps =
    PcuDispatchPolicyCaps::SERIAL.union(PcuDispatchPolicyCaps::ORDERED_SUBMISSION);
const FEATURES: PcuDispatchFeatureCaps = PcuDispatchFeatureCaps::MUTABLE_RESOURCES
    .union(PcuDispatchFeatureCaps::READ_ONLY_RESOURCES)
    .union(PcuDispatchFeatureCaps::RANGE_CLAMP);
pub(super) const fn support() -> PcuSupport {
    let mut support = PcuSupport::unsupported();
    support.caps = PcuCaps::ENUMERATE_EXECUTORS
        .union(PcuCaps::DISPATCH)
        .union(PcuCaps::COMPUTE_DISPATCH);
    support.implementation = PcuImplementationKind::Native;
    support.executor_count = 1;
    support.primitive_support = PcuPrimitiveSupport {
        primitives: PcuFeatureSupport::new(PcuPrimitiveCaps::DISPATCH, PcuPrimitiveCaps::empty()),
    };
    support.value_type_support = PcuFeatureSupport::new(TYPES, PcuValueTypeCaps::empty());
    let mut dispatch = PcuDispatchSupport::unsupported();
    dispatch.flags = POLICY;
    dispatch.instructions = PcuFeatureSupport::new(INSTRUCTIONS, PcuDispatchOpCaps::empty());
    dispatch.scalar_alu = PcuFeatureSupport::new(SCALAR, PcuDispatchScalarAluSupport::empty());
    dispatch.features = PcuFeatureSupport::new(FEATURES, PcuDispatchFeatureCaps::empty());
    support.dispatch_support = dispatch;
    support
}
pub(super) const EXECUTORS: [PcuExecutorDescriptor; 1] = [PcuExecutorDescriptor {
    id: super::EXECUTOR,
    name: "MLX-owned exact carriers, checked floating/integer arithmetic and delegated tensor executor",
    class: PcuExecutorClass::Compute,
    origin: PcuExecutorOrigin::TopologyBound,
    support: PcuExecutorSupport {
        primitives: PcuPrimitiveCaps::DISPATCH,
        dispatch_policy: POLICY,
        value_types: TYPES,
        dispatch_instructions: INSTRUCTIONS,
        dispatch_scalar_alu: SCALAR,
        dispatch_features: FEATURES,
        stream_instructions: fusion_pcu::PcuStreamCapabilities::empty(),
        command_instructions: fusion_pcu::PcuCommandOpCaps::empty(),
        transaction_features: fusion_pcu::PcuTransactionFeatureCaps::empty(),
        signal_instructions: fusion_pcu::PcuSignalOpCaps::empty(),
    },
}];

#[cfg(test)]
mod tests {
    #[rustfmt::skip]
    use super::{
        TYPES,
        SCALAR,
        INSTRUCTIONS,
    };
    #[rustfmt::skip]
    use fusion_pcu::{
        PcuScalarType,
        PcuValueTypeCaps,
        PcuDispatchOpCaps,
    };
    #[test]
    fn carrier_type_floor_does_not_invent_arithmetic() {
        for scalar in [
            PcuScalarType::U8,
            PcuScalarType::I8,
            PcuScalarType::U16,
            PcuScalarType::I16,
            PcuScalarType::U32,
            PcuScalarType::I32,
            PcuScalarType::U64,
            PcuScalarType::I64,
            PcuScalarType::U128,
            PcuScalarType::I128,
            PcuScalarType::U256,
            PcuScalarType::I256,
            PcuScalarType::U512,
            PcuScalarType::I512,
            PcuScalarType::F16,
            PcuScalarType::BF16,
            PcuScalarType::F8E4M3FN,
            PcuScalarType::F8E5M2,
            PcuScalarType::F32,
            PcuScalarType::F64,
            PcuScalarType::F128,
            PcuScalarType::F256,
        ] {
            assert!(TYPES.contains(PcuValueTypeCaps::for_scalar(scalar)));
            let arithmetic = SCALAR.for_scalar(scalar);
            if matches!(
                scalar,
                PcuScalarType::F16
                    | PcuScalarType::BF16
                    | PcuScalarType::F8E4M3FN
                    | PcuScalarType::F8E5M2
            ) {
                assert!(arithmetic.contains(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY));
            } else if matches!(scalar, PcuScalarType::F32 | PcuScalarType::F64) {
                assert_eq!(
                    arithmetic,
                    PcuDispatchOpCaps::ALU_CHECKED_FLOAT_UNARY
                        .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY)
                        .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_CONVERT)
                );
            } else if matches!(scalar, PcuScalarType::F128 | PcuScalarType::F256) {
                assert_eq!(arithmetic, PcuDispatchOpCaps::empty());
            } else if matches!(
                scalar,
                PcuScalarType::U8
                    | PcuScalarType::I8
                    | PcuScalarType::U16
                    | PcuScalarType::I16
                    | PcuScalarType::U32
                    | PcuScalarType::I32
                    | PcuScalarType::U64
                    | PcuScalarType::I64
                    | PcuScalarType::U128
                    | PcuScalarType::I128
                    | PcuScalarType::U256
                    | PcuScalarType::I256
                    | PcuScalarType::U512
                    | PcuScalarType::I512
            ) {
                assert_eq!(
                    arithmetic,
                    PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY
                        .union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM)
                );
            } else {
                assert_eq!(arithmetic, PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY);
            }
        }
        for scalar in [PcuScalarType::Bool, PcuScalarType::I4, PcuScalarType::U4] {
            assert!(!TYPES.contains(PcuValueTypeCaps::for_scalar(scalar)));
        }
        assert!(INSTRUCTIONS.contains(PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY));
        assert!(INSTRUCTIONS.contains(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM));
    }
}
