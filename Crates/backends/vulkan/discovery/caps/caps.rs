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
const TYPES: PcuValueTypeCaps = PcuValueTypeCaps::FLOAT32.union(PcuValueTypeCaps::SCALAR_VALUES);
const INSTRUCTIONS: PcuDispatchOpCaps = PcuDispatchOpCaps::ALU_CHECKED_FLOAT_UNARY
    .union(PcuDispatchOpCaps::BINDING_LOAD)
    .union(PcuDispatchOpCaps::BINDING_STORE)
    .union(PcuDispatchOpCaps::CONTROL_RETURN)
    .union(PcuDispatchOpCaps::CONTROL_LOOP);
const SCALAR: PcuDispatchScalarAluSupport = PcuDispatchScalarAluSupport::empty().with(
    PcuScalarType::F32,
    PcuDispatchOpCaps::ALU_CHECKED_FLOAT_UNARY,
);
const POLICY: PcuDispatchPolicyCaps =
    PcuDispatchPolicyCaps::SERIAL.union(PcuDispatchPolicyCaps::ORDERED_SUBMISSION);
const FEATURES: PcuDispatchFeatureCaps =
    PcuDispatchFeatureCaps::MUTABLE_RESOURCES.union(PcuDispatchFeatureCaps::READ_ONLY_RESOURCES);

pub(super) const fn support(float64: bool) -> PcuSupport {
    let types = if float64 {
        TYPES.union(PcuValueTypeCaps::FLOAT64)
    } else {
        TYPES
    };
    let scalar = if float64 {
        SCALAR.with(
            PcuScalarType::F64,
            PcuDispatchOpCaps::ALU_CHECKED_FLOAT_UNARY,
        )
    } else {
        SCALAR
    };
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

pub(super) const fn executor(float64: bool) -> PcuExecutorDescriptor {
    let types = if float64 {
        TYPES.union(PcuValueTypeCaps::FLOAT64)
    } else {
        TYPES
    };
    let scalar = if float64 {
        SCALAR.with(
            PcuScalarType::F64,
            PcuDispatchOpCaps::ALU_CHECKED_FLOAT_UNARY,
        )
    } else {
        SCALAR
    };
    PcuExecutorDescriptor {
        id: PcuExecutorId(0),
        name: "vulkan-prepared-bit-map",
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
    fn f64_support_is_conditional_and_does_not_imply_other_arithmetic() {
        assert!(
            !executor(false)
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
            !executor(true)
                .support
                .dispatch_instructions
                .contains(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY)
        );
        assert!(
            support(true)
                .value_type_support
                .direct
                .contains(PcuValueTypeCaps::FLOAT64)
        );
        assert!(
            !support(false)
                .value_type_support
                .direct
                .contains(PcuValueTypeCaps::FLOAT64)
        );
    }
}
