//! Cold resource projection and dense SSA register mapping; original headers remain immutable.
#[rustfmt::skip]
use fusion_pcu::{
    describe_scalar_transport_map,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingType,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuReproducibility,
    PcuScalarType,
    PcuValueType,
};
#[rustfmt::skip]
use super::{
    PcuSpirvComposedResource,
    PcuSpirvOrderedTransportProfile as Profile,
    PcuSpirvError as Error,
    RESOURCES,
    SOURCE_REGISTERS,
    STEPS,
    Step,
};
pub(super) fn prepare(kernel: &PcuDispatchKernelIr<'_>) -> Result<Profile, Error> {
    if kernel
        .numerical_requirements
        .numerical_options
        .reproducibility
        != PcuReproducibility::Unspecified
    {
        return Err(Error::UnsupportedNumericalRequirements);
    }
    let PcuBindingType::Value(PcuValueType::Scalar(scalar)) = kernel
        .bindings
        .first()
        .ok_or(Error::InvalidBinding)?
        .binding_type
    else {
        return Err(Error::InvalidBinding);
    };
    if matches!(
        scalar,
        PcuScalarType::Bool | PcuScalarType::I4 | PcuScalarType::U4
    ) {
        return Err(Error::UnsupportedValueType(PcuValueType::Scalar(scalar)));
    }
    let description = describe_scalar_transport_map::<RESOURCES>(kernel, scalar)
        .map_err(|_| Error::InvalidKernelSignature)?;
    let (body, extent) = match kernel.ops {
        [
            PcuDispatchOp::GridStrideLoop { extent, body },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ] => (*body, *extent),
        [
            body @ ..,
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ] => (body, kernel.entry.logical_shape[0]),
        _ => return Err(Error::InvalidKernelSignature),
    };
    if body.len() > STEPS || extent == 0 {
        return Err(Error::InvalidKernelSignature);
    }
    extent
        .checked_mul(u32::from(scalar.bit_width() / 8))
        .ok_or(Error::InvalidKernelSignature)?;
    let mut resources = [PcuSpirvComposedResource {
        binding: PcuBindingRef::new(0, 0),
        declaration: 0,
        access: PcuBindingAccess::ReadOnly,
        read_elements: 0,
        write_elements: 0,
    }; RESOURCES];
    for (index, resource) in description.resources().iter().enumerate() {
        if resource.has_cross_index_read_write() {
            return Err(Error::InvalidKernelSignature);
        }
        let declaration = kernel
            .bindings
            .iter()
            .position(|b| b.reference() == resource.binding)
            .ok_or(Error::InvalidBinding)?;
        resources[index] = PcuSpirvComposedResource {
            binding: resource.binding,
            declaration,
            access: kernel.bindings[declaration].access,
            read_elements: resource.minimum_read_elements,
            write_elements: resource.minimum_write_elements,
        };
    }
    let steps = compile_steps(body, &resources[..description.resources().len()])?;
    Ok(Profile {
        requirements: kernel.numerical_requirements,
        scalar,
        extent,
        resources,
        resource_count: description.resources().len(),
        steps,
        step_count: body.len(),
    })
}

fn compile_steps(
    body: &[PcuDispatchOp<'_>],
    resources: &[PcuSpirvComposedResource],
) -> Result<[Step; STEPS], Error> {
    let mut steps = [Step::EMPTY; STEPS];
    let mut registers = [None; SOURCE_REGISTERS];
    let mut register_count = 0u32;
    for (index, operation) in body.iter().enumerate() {
        steps[index] = match *operation {
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result,
                binding,
                index,
            }) => {
                let resource = resources
                    .iter()
                    .position(|r| r.binding == binding)
                    .ok_or(Error::InvalidBinding)?;
                let slot = registers
                    .get_mut(usize::from(result.0))
                    .ok_or(Error::InvalidKernelSignature)?;
                *slot = Some(register_count);
                let function = if index == PcuDispatchIndex::BindingElementZero {
                    if resources[resource].write_elements == 0 {
                        1
                    } else {
                        2
                    }
                } else {
                    0
                };
                let step = Step {
                    function,
                    arguments: [
                        register_count,
                        u32::try_from(resource).map_err(|_| Error::InvalidBinding)?,
                        0,
                    ],
                };
                register_count += 1;
                step
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore { value, binding, .. }) => {
                let register = registers
                    .get(usize::from(value.0))
                    .copied()
                    .flatten()
                    .ok_or(Error::InvalidKernelSignature)?;
                let resource = resources
                    .iter()
                    .position(|r| r.binding == binding)
                    .ok_or(Error::InvalidBinding)?;
                Step {
                    function: 3,
                    arguments: [
                        0,
                        u32::try_from(resource).map_err(|_| Error::InvalidBinding)?,
                        register,
                    ],
                }
            }
            _ => return Err(Error::InvalidKernelSignature),
        };
    }
    Ok(steps)
}
