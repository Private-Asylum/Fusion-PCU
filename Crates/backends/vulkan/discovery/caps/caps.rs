//! Coarse bounded bit-map floor; exact structure/policies are admitted cold.

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
    PcuExecutorId,
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
const TYPES: PcuValueTypeCaps = PcuValueTypeCaps::FLOAT32
    .union(PcuValueTypeCaps::FLOAT64)
    .union(PcuValueTypeCaps::for_scalar(PcuScalarType::F16))
    .union(PcuValueTypeCaps::for_scalar(PcuScalarType::BF16))
    .union(PcuValueTypeCaps::for_scalar(PcuScalarType::F8E4M3FN))
    .union(PcuValueTypeCaps::for_scalar(PcuScalarType::F8E5M2))
    .union(PcuValueTypeCaps::for_scalar(PcuScalarType::I8))
    .union(PcuValueTypeCaps::for_scalar(PcuScalarType::U8))
    .union(PcuValueTypeCaps::for_scalar(PcuScalarType::I16))
    .union(PcuValueTypeCaps::for_scalar(PcuScalarType::U16))
    .union(PcuValueTypeCaps::for_scalar(PcuScalarType::I32))
    .union(PcuValueTypeCaps::for_scalar(PcuScalarType::U32))
    .union(PcuValueTypeCaps::for_scalar(PcuScalarType::I64))
    .union(PcuValueTypeCaps::for_scalar(PcuScalarType::U64))
    .union(PcuValueTypeCaps::for_scalar(PcuScalarType::I128))
    .union(PcuValueTypeCaps::for_scalar(PcuScalarType::U128))
    .union(PcuValueTypeCaps::for_scalar(PcuScalarType::I256))
    .union(PcuValueTypeCaps::for_scalar(PcuScalarType::U256))
    .union(PcuValueTypeCaps::for_scalar(PcuScalarType::I512))
    .union(PcuValueTypeCaps::for_scalar(PcuScalarType::U512))
    .union(PcuValueTypeCaps::for_scalar(PcuScalarType::F128))
    .union(PcuValueTypeCaps::for_scalar(PcuScalarType::F256))
    .union(PcuValueTypeCaps::SCALAR_VALUES);
const INSTRUCTIONS: PcuDispatchOpCaps = PcuDispatchOpCaps::ALU_CHECKED_FLOAT_UNARY
    .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY)
    .union(PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY)
    .union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM)
    .union(PcuDispatchOpCaps::BINDING_LOAD)
    .union(PcuDispatchOpCaps::BINDING_STORE)
    .union(PcuDispatchOpCaps::CONTROL_RETURN)
    .union(PcuDispatchOpCaps::CONTROL_LOOP);
const SCALAR: PcuDispatchScalarAluSupport = PcuDispatchScalarAluSupport::empty()
    .with(
        PcuScalarType::F32,
        PcuDispatchOpCaps::ALU_CHECKED_FLOAT_UNARY
            .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY),
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
        PcuScalarType::F64,
        PcuDispatchOpCaps::ALU_CHECKED_FLOAT_UNARY
            .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY),
    )
    .with(
        PcuScalarType::I8,
        PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY.union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM),
    )
    .with(
        PcuScalarType::U8,
        PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY.union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM),
    )
    .with(
        PcuScalarType::I16,
        PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY.union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM),
    )
    .with(
        PcuScalarType::U16,
        PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY.union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM),
    )
    .with(
        PcuScalarType::I32,
        PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY.union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM),
    )
    .with(
        PcuScalarType::U32,
        PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY.union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM),
    )
    .with(
        PcuScalarType::I64,
        PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY.union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM),
    )
    .with(
        PcuScalarType::U64,
        PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY.union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM),
    )
    .with(
        PcuScalarType::I128,
        PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY,
    )
    .with(
        PcuScalarType::U128,
        PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY,
    )
    .with(
        PcuScalarType::I256,
        PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY,
    )
    .with(
        PcuScalarType::U256,
        PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY,
    )
    .with(
        PcuScalarType::I512,
        PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY,
    )
    .with(
        PcuScalarType::U512,
        PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY,
    );
const POLICY: PcuDispatchPolicyCaps =
    PcuDispatchPolicyCaps::SERIAL.union(PcuDispatchPolicyCaps::ORDERED_SUBMISSION);
const FEATURES: PcuDispatchFeatureCaps = PcuDispatchFeatureCaps::MUTABLE_RESOURCES
    .union(PcuDispatchFeatureCaps::READ_ONLY_RESOURCES)
    .union(PcuDispatchFeatureCaps::RANGE_CLAMP);

pub(super) const fn support(_float64: bool) -> PcuSupport {
    let types = TYPES;
    let scalar = SCALAR;
    let mut support = PcuSupport::unsupported();
    support.caps = PcuCaps::ENUMERATE_EXECUTORS
        .union(PcuCaps::DISPATCH)
        .union(PcuCaps::COMPUTE_DISPATCH);
    support.implementation = PcuImplementationKind::Native;
    support.executor_count = 1;
    support.primitive_support = PcuPrimitiveSupport {
        primitives: PcuFeatureSupport::new(PcuPrimitiveCaps::DISPATCH, PcuPrimitiveCaps::empty()),
    };
    support.value_type_support = PcuFeatureSupport::new(types, PcuValueTypeCaps::empty());
    let mut dispatch = PcuDispatchSupport::unsupported();
    dispatch.flags = POLICY;
    dispatch.instructions = PcuFeatureSupport::new(INSTRUCTIONS, PcuDispatchOpCaps::empty());
    dispatch.scalar_alu = PcuFeatureSupport::new(scalar, PcuDispatchScalarAluSupport::empty());
    dispatch.features = PcuFeatureSupport::new(FEATURES, PcuDispatchFeatureCaps::empty());
    support.dispatch_support = dispatch;
    support
}

pub(super) const fn executor(_float64: bool) -> PcuExecutorDescriptor {
    let types = TYPES;
    let scalar = SCALAR;
    PcuExecutorDescriptor {
        id: PcuExecutorId(0),
        name: "vulkan-prepared-scalar-map",
        class: PcuExecutorClass::Compute,
        origin: PcuExecutorOrigin::TopologyBound,
        support: PcuExecutorSupport {
            primitives: PcuPrimitiveCaps::DISPATCH,
            dispatch_policy: POLICY,
            value_types: types,
            dispatch_instructions: INSTRUCTIONS,
            dispatch_scalar_alu: scalar,
            dispatch_features: FEATURES,
            ..PcuExecutorSupport::unsupported()
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_integer_floor_needs_no_optional_native_width() {
        for scalar in [
            PcuScalarType::I8,
            PcuScalarType::U8,
            PcuScalarType::I16,
            PcuScalarType::U16,
            PcuScalarType::I32,
            PcuScalarType::U32,
            PcuScalarType::I64,
            PcuScalarType::U64,
            PcuScalarType::I128,
            PcuScalarType::U128,
            PcuScalarType::I256,
            PcuScalarType::U256,
            PcuScalarType::I512,
            PcuScalarType::U512,
        ] {
            assert!(
                executor(false)
                    .support
                    .dispatch_scalar_alu
                    .for_scalar(scalar)
                    .contains(PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY)
            );
        }
        for scalar in [
            PcuScalarType::Bool,
            PcuScalarType::I4,
            PcuScalarType::U4,
            PcuScalarType::F128,
            PcuScalarType::F256,
        ] {
            assert!(
                !executor(false)
                    .support
                    .dispatch_scalar_alu
                    .for_scalar(scalar)
                    .contains(PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY)
            );
        }
    }

    #[test]
    fn checked_f64_binary_and_unary_need_only_u32() {
        assert!(
            executor(false)
                .support
                .dispatch_scalar_alu
                .for_scalar(PcuScalarType::F64)
                .contains(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_UNARY)
        );
        assert!(
            executor(false)
                .support
                .dispatch_scalar_alu
                .for_scalar(PcuScalarType::F64)
                .contains(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY)
        );
        assert!(
            executor(false)
                .support
                .value_types
                .contains(PcuValueTypeCaps::FLOAT64)
        );
        assert!(
            executor(true)
                .support
                .value_types
                .contains(PcuValueTypeCaps::FLOAT64)
        );
        assert!(
            executor(true)
                .support
                .dispatch_instructions
                .contains(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY)
        );
        assert!(
            executor(true)
                .support
                .dispatch_scalar_alu
                .for_scalar(PcuScalarType::F64)
                .contains(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY)
        );
        assert!(
            support(true)
                .value_type_support
                .direct
                .contains(PcuValueTypeCaps::FLOAT64)
        );
        assert!(
            support(false)
                .value_type_support
                .direct
                .contains(PcuValueTypeCaps::FLOAT64)
        );
    }
}
