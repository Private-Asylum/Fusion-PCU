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
const fn carrier_types() -> PcuValueTypeCaps {
    let mut types = PcuValueTypeCaps::SCALAR_VALUES;
    let mut index = 0;
    while index < PcuScalarType::COUNT {
        let scalar = PcuScalarType::ALL[index];
        if scalar.bit_width() >= 8 {
            types = types.union(PcuValueTypeCaps::for_scalar(scalar));
        }
        index += 1;
    }
    types
}
const TYPES: PcuValueTypeCaps = carrier_types();
const INSTRUCTIONS: PcuDispatchOpCaps = PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY
    .union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM)
    .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY)
    .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_UNARY)
    .union(PcuDispatchOpCaps::BINDING_LOAD)
    .union(PcuDispatchOpCaps::BINDING_LOAD_ELEMENT_ZERO)
    .union(PcuDispatchOpCaps::BINDING_STORE)
    .union(PcuDispatchOpCaps::CONTROL_RETURN)
    .union(PcuDispatchOpCaps::CONTROL_LOOP);
const SCALAR: PcuDispatchScalarAluSupport = PcuDispatchScalarAluSupport::empty()
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
        PcuScalarType::I128,
        PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY.union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM),
    )
    .with(
        PcuScalarType::U128,
        PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY.union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM),
    )
    .with(
        PcuScalarType::I256,
        PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY.union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM),
    )
    .with(
        PcuScalarType::U256,
        PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY.union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM),
    )
    .with(
        PcuScalarType::I512,
        PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY.union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM),
    )
    .with(
        PcuScalarType::U512,
        PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY.union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM),
    )
    .with(
        PcuScalarType::F16,
        PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY
            .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_UNARY),
    )
    .with(
        PcuScalarType::BF16,
        PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY
            .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_UNARY),
    )
    .with(
        PcuScalarType::F8E4M3FN,
        PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY
            .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_UNARY),
    )
    .with(
        PcuScalarType::F8E5M2,
        PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY
            .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_UNARY),
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
        PcuScalarType::F64,
        PcuDispatchOpCaps::ALU_CHECKED_FLOAT_UNARY
            .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY),
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
        PcuScalarType::F32,
        PcuDispatchOpCaps::ALU_CHECKED_FLOAT_UNARY
            .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY),
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
