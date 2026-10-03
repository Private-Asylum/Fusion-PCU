//! Separate byte-preserving load/store projection; this is not checked arithmetic.
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingRef,
    PcuBindingType,
    PcuDispatchKernelIr,
    PcuScalarTransportDescription,
    PcuScalarType,
    PcuValueType,
};
const CAPACITY: usize = 4;

/// Detached original-resource roles, retaining declared access and full requirements.
#[derive(Clone, Copy)]
pub struct TransportProjection {
    description: PcuScalarTransportDescription<CAPACITY>,
    inputs: [PcuBindingRef; CAPACITY],
    input_count: usize,
}
impl TransportProjection {
    pub(super) fn input_bindings(&self) -> &[PcuBindingRef] {
        &self.inputs[..self.input_count]
    }
    pub(super) fn contains_output(self, binding: PcuBindingRef) -> bool {
        self.description
            .resource(binding)
            .is_some_and(|resource| resource.minimum_write_elements != 0)
    }
}

pub(super) fn project(kernel: &PcuDispatchKernelIr<'_>) -> Option<TransportProjection> {
    let PcuBindingType::Value(PcuValueType::Scalar(scalar)) = kernel.bindings.first()?.binding_type
    else {
        return None;
    };
    // Neutral structural identities are broader than executable host carriers.
    // This opt-in keeps exactly the existing 22 byte-transport representations.
    if !matches!(
        scalar,
        PcuScalarType::I8
            | PcuScalarType::U8
            | PcuScalarType::I16
            | PcuScalarType::U16
            | PcuScalarType::I32
            | PcuScalarType::U32
            | PcuScalarType::I64
            | PcuScalarType::U64
            | PcuScalarType::I128
            | PcuScalarType::U128
            | PcuScalarType::I256
            | PcuScalarType::U256
            | PcuScalarType::I512
            | PcuScalarType::U512
            | PcuScalarType::F16
            | PcuScalarType::BF16
            | PcuScalarType::F8E4M3FN
            | PcuScalarType::F8E5M2
            | PcuScalarType::F32
            | PcuScalarType::F64
            | PcuScalarType::F128
            | PcuScalarType::F256
    ) {
        return None;
    }
    let description = fusion_pcu::describe_scalar_transport_map::<CAPACITY>(kernel, scalar).ok()?;
    if description
        .resources()
        .iter()
        .any(|resource| resource.has_cross_index_read_write())
    {
        // Required view capacity is not a snapshot proof. Existing alias checks
        // independently cover distinct logical references with overlapping spans.
        return None;
    }
    let mut projection = TransportProjection {
        description,
        inputs: [PcuBindingRef::new(0, 0); CAPACITY],
        input_count: 0,
    };
    for resource in description.resources() {
        if resource.minimum_read_elements != 0 {
            projection.inputs[projection.input_count] = resource.binding;
            projection.input_count += 1;
        }
    }
    Some(projection)
}
