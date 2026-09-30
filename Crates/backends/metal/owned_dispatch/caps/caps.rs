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
const TYPES: PcuValueTypeCaps = PcuValueTypeCaps::UINT32
    .union(PcuValueTypeCaps::FLOAT32)
    .union(PcuValueTypeCaps::SCALAR_VALUES);
const INSTRUCTIONS: PcuDispatchOpCaps = PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY
    .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_UNARY)
    .union(PcuDispatchOpCaps::BINDING_LOAD)
    .union(PcuDispatchOpCaps::BINDING_STORE)
    .union(PcuDispatchOpCaps::CONTROL_RETURN)
    .union(PcuDispatchOpCaps::CONTROL_LOOP);
const SCALAR: PcuDispatchScalarAluSupport = PcuDispatchScalarAluSupport::empty()
    .with(
        PcuScalarType::U32,
        PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY,
    )
    .with(
        PcuScalarType::F32,
        PcuDispatchOpCaps::ALU_CHECKED_FLOAT_UNARY,
    );
const POLICY: PcuDispatchPolicyCaps =
    PcuDispatchPolicyCaps::SERIAL.union(PcuDispatchPolicyCaps::ORDERED_SUBMISSION);
const FEATURES: PcuDispatchFeatureCaps =
    PcuDispatchFeatureCaps::MUTABLE_RESOURCES.union(PcuDispatchFeatureCaps::READ_ONLY_RESOURCES);
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
    id: PcuExecutorId(0),
    name: "metal-owned-checked-map",
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
