//! Cold opt-in for at most four actual resources in a composed checked scalar map.
//!
//! This provider profile does not impose a core binding limit. Existing broader
//! F32/F64 admission remains independent. First access is not device ABI order.

#[rustfmt::skip]
use fusion_pcu::{
    CheckedScalarMapResourceSchema,
    PcuBindingRef,
    PcuDispatchDataOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuScalarType,
    PcuValueTypeCaps,
};

const CAPACITY: usize = 4;

/// Detached cold projection; original declaration permissions remain authoritative.
#[derive(Clone, Copy)]
pub struct ComposedProjection {
    resources: CheckedScalarMapResourceSchema<CAPACITY>,
    inputs: [PcuBindingRef; CAPACITY],
    input_count: usize,
}
impl ComposedProjection {
    pub(super) fn input_bindings(&self) -> &[PcuBindingRef] {
        &self.inputs[..self.input_count]
    }

    pub(super) fn contains_output(self, binding: PcuBindingRef) -> bool {
        self.resources
            .resource(binding)
            .is_some_and(|resource| resource.minimum_write_elements != 0)
    }
}

pub(super) fn project(kernel: &PcuDispatchKernelIr<'_>) -> Option<ComposedProjection> {
    // Unary maps have the same validated resource law. Keep the separate
    // arithmetic profile detectors rather than routing unary through binary admission.
    let value_type = super::validation::checked_float_binary_profile(kernel)
        .or_else(|| super::validation::checked_float_unary_profile(kernel))
        .or_else(|| {
            super::validation::checked_integer_binary_profile(kernel).map(|profile| profile.0)
        })?;
    let caps = PcuValueTypeCaps::for_scalar(value_type.scalar_type());
    let resources = if super::admitted_integer(value_type) {
        if matches!(
            value_type.scalar_type(),
            PcuScalarType::U256 | PcuScalarType::I256 | PcuScalarType::U512 | PcuScalarType::I512
        ) {
            // Four wide carrier identities remain structural-only for composition.
            // Existing single-binary and joint integer profiles remain independent.
            return None;
        }
        // Integer helpers reuse the same homogeneous SSA/resource law. The physical
        // packed status retains the exact union of constituent integer fault kinds.
        fusion_pcu::assess_checked_integer_map_resources::<CAPACITY>(kernel, value_type, caps)
            .ok()?
    } else {
        fusion_pcu::assess_checked_float_map_resources::<CAPACITY>(kernel, value_type, caps).ok()?
    };
    if resources
        .resources()
        .iter()
        .any(|resource| resource.has_cross_index_read_write())
    {
        // The detached fact includes zero reads even when indexed reads merge
        // the minimum input span to N. Spans alone do not prove a snapshot.
        return None;
    }
    let body = match kernel.ops.first() {
        Some(PcuDispatchOp::GridStrideLoop { body, .. }) => *body,
        _ => kernel.ops,
    };
    for instruction in body {
        match *instruction {
            PcuDispatchOp::Data(
                PcuDispatchDataOp::CheckedIntegerBinary { range_policy, .. }
                | PcuDispatchDataOp::CheckedFloatBinary { range_policy, .. }
                | PcuDispatchDataOp::CheckedFloatUnary { range_policy, .. },
            ) if range_policy != kernel.numerical_requirements.range_policy => {
                // The retained word's disposition must implement the requested header.
                // Exact lexical underflow policies are still kept on each instruction.
                return None;
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::Constant { .. })
                if super::admitted_integer(value_type) =>
            {
                // This provider's existing integer emitter has no constant producer.
                // Neutral structural eligibility does not create executable support.
                return None;
            }
            _ => (),
        }
    }
    let mut projection = ComposedProjection {
        resources,
        inputs: [PcuBindingRef::new(0, 0); CAPACITY],
        input_count: 0,
    };
    for resource in resources.resources() {
        if resource.minimum_read_elements != 0 {
            projection.inputs[projection.input_count] = resource.binding;
            projection.input_count += 1;
        }
    }
    Some(projection)
}

/// Cold rejection only: zero-read/full-write on the same resource is not a snapshot.
/// This also covers the preexisting broader F32/F64 fallback, without a capacity cap.
pub(super) fn has_broadcast_write_hazard(kernel: &PcuDispatchKernelIr<'_>) -> bool {
    let (body, extent) = match kernel.ops.first() {
        Some(PcuDispatchOp::GridStrideLoop { body, extent }) => (*body, *extent),
        _ => (kernel.ops, kernel.entry.logical_shape[0]),
    };
    if extent <= 1 {
        return false;
    }
    body.iter().any(|operation| {
        let PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            binding,
            index: PcuDispatchIndex::BindingElementZero,
            ..
        }) = operation
        else {
            return false;
        };
        body.iter().any(|candidate| {
            matches!(candidate,
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                    binding: output,
                    index: PcuDispatchIndex::InvocationId | PcuDispatchIndex::GridStrideId,
                    ..
                }) if output == binding
            )
        })
    })
}
