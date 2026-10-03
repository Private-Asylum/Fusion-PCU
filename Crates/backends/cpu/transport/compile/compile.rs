//! Cold typed transport admission and indexing, preserving the original request and access roles.
use alloc::vec::Vec;
#[rustfmt::skip]
use fusion_pcu::{
    describe_scalar_transport_map,
    PcuBindingAccess,
    PcuBindingType,
    PcuDispatchDataOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuReproducibility,
    PcuScalarType,
    PcuScalarTransportDescription,
    PcuValueType,
};
#[rustfmt::skip]
use super::{
    CARRIER_BYTES,
    REGISTERS,
    RESOURCES,
    STEPS,
    PcuCpuPreparedScalarTransport as Plan,
    PcuCpuScalarTransportError as Error,
    Resource,
    Step,
};

pub(super) fn prepare(kernel: &PcuDispatchKernelIr<'_>) -> Result<Plan, Error> {
    if kernel
        .numerical_requirements
        .numerical_options
        .reproducibility
        != PcuReproducibility::Unspecified
    {
        return Err(Error::UnsupportedProfile);
    }
    let Some(first) = kernel.bindings.first() else {
        return Err(Error::UnsupportedProfile);
    };
    let PcuBindingType::Value(PcuValueType::Scalar(scalar)) = first.binding_type else {
        return Err(Error::UnsupportedProfile);
    };
    let size = carrier_size(scalar)?;
    let description = describe_scalar_transport_map::<RESOURCES>(kernel, scalar)
        .map_err(Error::InvalidTransport)?;
    let body = crate::host::validated_region(kernel).map_err(|_| Error::UnsupportedProfile)?;
    if body.len() > STEPS
        || kernel
            .bindings
            .iter()
            .any(|b| b.access == PcuBindingAccess::WriteOnly)
    {
        // Public host arguments retain an actual mutable Rust borrow, represented as ReadWrite.
        return Err(Error::UnsupportedProfile);
    }
    let mut resources = Vec::new();
    resources
        .try_reserve_exact(description.resources().len())
        .map_err(|_| Error::AllocationFailed)?;
    let mut bytes = 0_usize;
    for resource in description.resources() {
        if resource.has_cross_index_read_write() {
            return Err(Error::UnsupportedProfile);
        }
        let declaration = kernel
            .bindings
            .iter()
            .position(|b| b.reference() == resource.binding)
            .ok_or(Error::InvalidProgram)?;
        let read_bytes = byte_extent(resource.minimum_read_elements, size)?;
        let write_bytes = byte_extent(resource.minimum_write_elements, size)?;
        let shadow = if write_bytes == 0 {
            None
        } else {
            let start = bytes;
            bytes = bytes
                .checked_add(read_bytes.max(write_bytes))
                .ok_or(Error::ExtentOverflow)?;
            Some(start)
        };
        // Conservative original-content copies remain explicit; the separate initial-read
        // optimization is not consumed by this first executable transport profile.
        resources.push(Resource {
            declaration,
            read_bytes,
            write_bytes,
            shadow,
        });
    }
    let mut schema = Vec::new();
    schema
        .try_reserve_exact(kernel.bindings.len())
        .map_err(|_| Error::AllocationFailed)?;
    for binding in kernel.bindings {
        let elements = description
            .resource(binding.reference())
            .map_or(0, fusion_pcu::PcuScalarTransportResource::minimum_elements);
        schema.push((
            binding.reference(),
            usize::try_from(elements).map_err(|_| Error::ExtentOverflow)?,
            binding.access,
        ));
    }
    let steps = compile_steps(body, &description)?;
    let mut scratch = Vec::new();
    scratch
        .try_reserve_exact(bytes)
        .map_err(|_| Error::AllocationFailed)?;
    scratch.resize(bytes, 0);
    let mut registers = Vec::new();
    registers
        .try_reserve_exact(REGISTERS)
        .map_err(|_| Error::AllocationFailed)?;
    registers.resize(REGISTERS, [0; CARRIER_BYTES]);
    let ordinal = PcuScalarType::ALL
        .iter()
        .position(|t| *t == scalar)
        .ok_or(Error::UnsupportedProfile)?;
    Ok(Plan {
        scalar,
        requirements: kernel.numerical_requirements,
        size,
        extent: usize::try_from(description.logical_extent).map_err(|_| Error::ExtentOverflow)?,
        local_id: 19456 + u32::try_from(ordinal).map_err(|_| Error::UnsupportedProfile)?,
        schema,
        resources,
        steps,
        scratch,
        registers,
    })
}
fn byte_extent(elements: u32, size: usize) -> Result<usize, Error> {
    usize::try_from(elements)
        .ok()
        .and_then(|n| n.checked_mul(size))
        .ok_or(Error::ExtentOverflow)
}
const fn carrier_size(scalar: PcuScalarType) -> Result<usize, Error> {
    Ok(match scalar {
        PcuScalarType::I8 | PcuScalarType::U8 | PcuScalarType::F8E4M3FN | PcuScalarType::F8E5M2 => {
            1
        }
        PcuScalarType::I16 | PcuScalarType::U16 | PcuScalarType::F16 | PcuScalarType::BF16 => 2,
        PcuScalarType::I32 | PcuScalarType::U32 | PcuScalarType::F32 => 4,
        PcuScalarType::I64 | PcuScalarType::U64 | PcuScalarType::F64 => 8,
        PcuScalarType::I128 | PcuScalarType::U128 | PcuScalarType::F128 => 16,
        PcuScalarType::I256 | PcuScalarType::U256 | PcuScalarType::F256 => 32,
        PcuScalarType::I512 | PcuScalarType::U512 => 64,
        PcuScalarType::Bool | PcuScalarType::I4 | PcuScalarType::U4 => {
            return Err(Error::UnsupportedProfile);
        }
    })
}

fn compile_steps(
    body: &[PcuDispatchOp<'_>],
    description: &PcuScalarTransportDescription<RESOURCES>,
) -> Result<Vec<Step>, Error> {
    let mut steps = Vec::new();
    steps
        .try_reserve_exact(body.len())
        .map_err(|_| Error::AllocationFailed)?;
    for operation in body {
        steps.push(match *operation {
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result,
                binding,
                index,
            }) => Step::Load {
                result: usize::from(result.0),
                resource: description
                    .resources()
                    .iter()
                    .position(|r| r.binding == binding)
                    .ok_or(Error::InvalidProgram)?,
                zero: index == PcuDispatchIndex::BindingElementZero,
            },
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore { binding, value, .. }) => {
                Step::Store {
                    value: usize::from(value.0),
                    resource: description
                        .resources()
                        .iter()
                        .position(|r| r.binding == binding)
                        .ok_or(Error::InvalidProgram)?,
                }
            }
            _ => return Err(Error::InvalidProgram),
        });
    }
    Ok(steps)
}
