//! Exact resource assessment and dense register indexing, performed once cold.
#[rustfmt::skip]
use fusion_pcu::{
    assess_checked_float_map_resources,
    validate_typed_dispatch_value_flow,
    PcuCheckedScalarFaultLaw,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchFloatBinaryOp,
    PcuDispatchFloatUnaryOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchValueId,
    PcuFloatUnderflowPolicy,
    PcuParameterValue,
    PcuRangePolicy,
    PcuReproducibility,
    PcuScalarType,
    PcuValueType,
    PcuValueTypeCaps,
};
#[rustfmt::skip]
use super::{
    BINDINGS,
    SOURCE_REGISTERS,
    STEPS,
    Step,
    PcuSpirvComposedResource as Resource,
    PcuSpirvComposedFloatProfile as Profile,
    PcuSpirvError as Error,
};

pub(super) fn prepare(
    kernel: &PcuDispatchKernelIr<'_>,
    one_effect: bool,
) -> Result<Profile, Error> {
    if kernel
        .numerical_requirements
        .numerical_options
        .reproducibility
        != PcuReproducibility::Unspecified
    {
        return Err(Error::UnsupportedNumericalRequirements);
    }
    let scalar = scalar(kernel)?;
    let schema = assess_checked_float_map_resources::<BINDINGS>(
        kernel,
        PcuValueType::Scalar(scalar),
        PcuValueTypeCaps::for_scalar(scalar),
    )
    .map_err(|_| Error::InvalidKernelSignature)?;
    validate_typed_dispatch_value_flow(kernel).map_err(|_| Error::InvalidKernelSignature)?;
    let body = match kernel.ops {
        [
            PcuDispatchOp::GridStrideLoop { body, .. },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ] => *body,
        [
            body @ ..,
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ] => body,
        _ => return Err(Error::InvalidKernelSignature),
    };
    let checked = body
        .iter()
        .filter(|op| {
            matches!(
                op,
                PcuDispatchOp::Data(
                    PcuDispatchDataOp::CheckedFloatBinary { .. }
                        | PcuDispatchDataOp::CheckedFloatUnary { .. }
                )
            )
        })
        .count();
    if body.len() > STEPS
        || if one_effect {
            checked != 1
        } else {
            checked < 2
        }
        || schema.logical_extent == 0
        || schema.resources().is_empty()
        || schema.logical_extent.checked_mul(2).is_none()
    {
        return Err(Error::InvalidKernelSignature);
    }
    let mut plan = Profile {
        requirements: kernel.numerical_requirements,
        scalar,
        extent: schema.logical_extent,
        declarations: [fusion_pcu::PcuBindingRef::new(0, 0); BINDINGS],
        declaration_access: [fusion_pcu::PcuBindingAccess::ReadOnly; BINDINGS],
        declaration_count: kernel.bindings.len(),
        resources: [Resource::EMPTY; BINDINGS],
        resource_count: schema.resources().len(),
        steps: [Step::EMPTY; STEPS],
        step_count: body.len(),
        one_effect,
    };
    for (slot, binding) in kernel.bindings.iter().enumerate() {
        plan.declarations[slot] = binding.reference();
        plan.declaration_access[slot] = binding.access;
    }
    for (slot, resource) in schema.resources().iter().enumerate() {
        if resource.has_cross_index_read_write() {
            return Err(Error::InvalidKernelSignature);
        }
        let declaration = kernel
            .bindings
            .iter()
            .position(|binding| binding.reference() == resource.binding)
            .ok_or(Error::InvalidBinding)?;
        plan.resources[slot] = Resource {
            binding: resource.binding,
            declaration,
            access: resource.declared_access,
            // Deliberately retain the conservative span. New initial-content metadata
            // is a separate optimization and gives this realization no extra rights.
            read_elements: resource.minimum_read_elements,
            write_elements: resource.minimum_write_elements,
        };
        let bytes = u64::from(resource.minimum_read_elements) * u64::from(scalar.bit_width() / 8);
        u32::try_from(bytes.div_ceil(4)).map_err(|_| Error::InvalidKernelSignature)?;
    }
    let mut registers = [u32::MAX; SOURCE_REGISTERS];
    let mut count = 0;
    for (ordinal, operation) in body.iter().enumerate() {
        plan.steps[ordinal] = step(*operation, &plan, &mut registers, &mut count)?;
        plan.steps[ordinal].arguments[7] =
            u32::try_from(ordinal).map_err(|_| Error::InvalidKernelSignature)?;
    }
    Ok(plan)
}

fn scalar(kernel: &PcuDispatchKernelIr<'_>) -> Result<PcuScalarType, Error> {
    if !(2..=BINDINGS).contains(&kernel.bindings.len()) {
        return Err(Error::InvalidKernelSignature);
    }
    match kernel.bindings[0].value_type() {
        Some(PcuValueType::Scalar(
            scalar @ (PcuScalarType::F16
            | PcuScalarType::BF16
            | PcuScalarType::F8E4M3FN
            | PcuScalarType::F8E5M2
            | PcuScalarType::F32
            | PcuScalarType::F64),
        )) => Ok(scalar),
        Some(other) => Err(Error::UnsupportedValueType(other)),
        None => Err(Error::InvalidBinding),
    }
}

fn source(value: PcuDispatchValueId, registers: &[u32; SOURCE_REGISTERS]) -> Result<u32, Error> {
    registers
        .get(usize::from(value.0))
        .copied()
        .filter(|slot| *slot != u32::MAX)
        .ok_or(Error::InvalidKernelSignature)
}

fn destination(
    value: PcuDispatchValueId,
    registers: &mut [u32; SOURCE_REGISTERS],
    count: &mut u32,
) -> Result<u32, Error> {
    if *count >= 64 {
        return Err(Error::InvalidKernelSignature);
    }
    let slot = registers
        .get_mut(usize::from(value.0))
        .ok_or(Error::InvalidKernelSignature)?;
    if *slot != u32::MAX {
        return Err(Error::InvalidKernelSignature);
    }
    *slot = *count;
    *count += 1;
    Ok(*slot)
}

fn resource(binding: fusion_pcu::PcuBindingRef, plan: &Profile) -> Result<u32, Error> {
    plan.resources()
        .iter()
        .position(|resource| resource.binding == binding)
        .and_then(|index| u32::try_from(index).ok())
        .ok_or(Error::InvalidBinding)
}

fn policies(range: PcuRangePolicy, uf: PcuFloatUnderflowPolicy) -> (u32, u32) {
    (
        match uf {
            PcuFloatUnderflowPolicy::IeeeAfterRounding => 0,
            PcuFloatUnderflowPolicy::RejectSubnormalResult => 1,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow => 2,
        },
        u32::from(range == PcuRangePolicy::Clamp),
    )
}

fn step(
    operation: PcuDispatchOp<'_>,
    plan: &Profile,
    registers: &mut [u32; SOURCE_REGISTERS],
    count: &mut u32,
) -> Result<Step, Error> {
    let mut step = Step::EMPTY;
    match operation {
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result,
            binding,
            index,
        }) => {
            step.function = 1;
            step.arguments[0] = destination(result, registers, count)?;
            step.arguments[1] = resource(binding, plan)?;
            // With one logical lane, element zero is the lane's private word. A
            // later RW load must observe earlier ordered stores rather than original SSBO bytes.
            step.arguments[2] =
                u32::from(index == PcuDispatchIndex::BindingElementZero && plan.extent > 1);
        }
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore { binding, value, .. }) => {
            step.function = 2;
            step.arguments[1] = resource(binding, plan)?;
            step.arguments[2] = source(value, registers)?;
        }
        PcuDispatchOp::Data(PcuDispatchDataOp::Constant { result, value }) => {
            step.function = 3;
            step.arguments[0] = destination(result, registers, count)?;
            match value {
                PcuParameterValue::F32(bits) => step.arguments[5] = bits,
                PcuParameterValue::F64(bits) => {
                    step.arguments[5] = u32::try_from(bits & u64::from(u32::MAX))
                        .map_err(|_| Error::InvalidKernelSignature)?;
                    step.arguments[6] =
                        u32::try_from(bits >> 32).map_err(|_| Error::InvalidKernelSignature)?;
                }
                _ => return Err(Error::InvalidKernelSignature),
            }
        }
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
            result,
            lhs,
            rhs,
            op,
            range_policy,
            underflow_policy,
            ..
        }) => {
            step.arguments[1] = source(lhs, registers)?;
            step.arguments[2] = source(rhs, registers)?;
            step.arguments[0] = destination(result, registers, count)?;
            step.function = match op {
                PcuDispatchFloatBinaryOp::Add => 4,
                PcuDispatchFloatBinaryOp::Sub => 5,
                PcuDispatchFloatBinaryOp::Mul => 6,
                PcuDispatchFloatBinaryOp::Div => 7,
            };
            (step.arguments[3], step.arguments[4]) = policies(range_policy, underflow_policy);
            step.law = PcuCheckedScalarFaultLaw::float_binary(
                plan.scalar,
                op,
                range_policy,
                underflow_policy,
            );
        }
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
            result,
            value,
            op,
            range_policy,
            underflow_policy,
            ..
        }) => {
            step.arguments[1] = source(value, registers)?;
            step.arguments[0] = destination(result, registers, count)?;
            step.function = match op {
                PcuDispatchFloatUnaryOp::Neg => 8,
                PcuDispatchFloatUnaryOp::Relu => 9,
            };
            (step.arguments[3], step.arguments[4]) = policies(range_policy, underflow_policy);
            step.law = PcuCheckedScalarFaultLaw::float_unary(
                plan.scalar,
                op,
                range_policy,
                underflow_policy,
            );
        }
        _ => return Err(Error::InvalidKernelSignature),
    }
    Ok(step)
}
