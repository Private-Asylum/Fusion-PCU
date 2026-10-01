//! Coarse CPU checked-map capability floor; exact IR and policies require cold admission.

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
const TYPES: PcuValueTypeCaps = PcuValueTypeCaps::INT8
    .union(PcuValueTypeCaps::UINT8)
    .union(PcuValueTypeCaps::INT16)
    .union(PcuValueTypeCaps::UINT16)
    .union(PcuValueTypeCaps::INT32)
    .union(PcuValueTypeCaps::UINT32)
    .union(PcuValueTypeCaps::INT64)
    .union(PcuValueTypeCaps::UINT64)
    .union(PcuValueTypeCaps::FLOAT32)
    .union(PcuValueTypeCaps::FLOAT64)
    .union(PcuValueTypeCaps::SCALAR_VALUES);
const INSTRUCTIONS: PcuDispatchOpCaps = PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY
    .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_UNARY)
    .union(PcuDispatchOpCaps::BINDING_LOAD)
    .union(PcuDispatchOpCaps::BINDING_LOAD_ELEMENT_ZERO)
    .union(PcuDispatchOpCaps::BINDING_STORE)
    .union(PcuDispatchOpCaps::CONTROL_RETURN)
    .union(PcuDispatchOpCaps::CONTROL_LOOP);
const SCALAR: PcuDispatchScalarAluSupport = PcuDispatchScalarAluSupport::empty()
    .with(
        PcuScalarType::I8,
        PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY,
    )
    .with(
        PcuScalarType::U8,
        PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY,
    )
    .with(
        PcuScalarType::I16,
        PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY,
    )
    .with(
        PcuScalarType::U16,
        PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY,
    )
    .with(
        PcuScalarType::I32,
        PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY,
    )
    .with(
        PcuScalarType::U32,
        PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY,
    )
    .with(
        PcuScalarType::I64,
        PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY,
    )
    .with(
        PcuScalarType::U64,
        PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY,
    )
    .with(
        PcuScalarType::F32,
        PcuDispatchOpCaps::ALU_CHECKED_FLOAT_UNARY,
    )
    .with(
        PcuScalarType::F64,
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
    name: "cpu-host-checked-map",
    class: PcuExecutorClass::Cpu,
    origin: PcuExecutorOrigin::Synthetic,
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
