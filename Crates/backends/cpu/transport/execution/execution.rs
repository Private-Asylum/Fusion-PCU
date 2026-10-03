//! Byte-only detached execution; preflight every borrow before any public write.
use fusion_pcu::PcuHostArgument;
#[rustfmt::skip]
use super::{
    RESOURCES,
    PcuCpuPreparedScalarTransport as Plan,
    PcuCpuScalarTransportError as Error,
    Step,
};

fn positions(plan: &Plan, arguments: &[PcuHostArgument<'_>]) -> Result<[usize; RESOURCES], Error> {
    let mut positions = [0; RESOURCES];
    for (index, resource) in plan.resources.iter().enumerate() {
        positions[index] = arguments
            .iter()
            .position(|arg| arg.target() == plan.schema[resource.declaration].0)
            .ok_or(Error::InvalidProgram)?;
        let length = resource.read_bytes.max(resource.write_bytes);
        let start = arguments[positions[index]].bytes().as_ptr() as usize;
        let end = start.checked_add(length).ok_or(Error::ExtentOverflow)?;
        for (prior, other) in plan.resources[..index].iter().enumerate() {
            if resource.write_bytes == 0 && other.write_bytes == 0 {
                continue;
            }
            let other_length = other.read_bytes.max(other.write_bytes);
            let other_start = arguments[positions[prior]].bytes().as_ptr() as usize;
            let other_end = other_start
                .checked_add(other_length)
                .ok_or(Error::ExtentOverflow)?;
            if length != 0 && other_length != 0 && start < other_end && other_start < end {
                return Err(Error::PhysicalAlias);
            }
        }
    }
    Ok(positions)
}
pub(super) fn execute(plan: &mut Plan, arguments: &mut [PcuHostArgument<'_>]) -> Result<(), Error> {
    let positions = positions(plan, arguments)?;
    for (index, resource) in plan.resources.iter().enumerate() {
        if let Some(start) = resource.shadow {
            plan.scratch[start..start + resource.read_bytes]
                .copy_from_slice(&arguments[positions[index]].bytes()[..resource.read_bytes]);
        }
    }
    for lane in 0..plan.extent {
        for step in &plan.steps {
            match *step {
                Step::Load {
                    result,
                    resource,
                    zero,
                } => {
                    let offset = if zero { 0 } else { lane * plan.size };
                    let resource_info = plan.resources[resource];
                    let bytes = if let Some(start) = resource_info.shadow {
                        &plan.scratch[start + offset..start + offset + plan.size]
                    } else {
                        &arguments[positions[resource]].bytes()[offset..offset + plan.size]
                    };
                    plan.registers[result][..plan.size].copy_from_slice(bytes);
                }
                Step::Store { value, resource } => {
                    let start = plan.resources[resource]
                        .shadow
                        .ok_or(Error::InvalidProgram)?;
                    let offset = start + lane * plan.size;
                    plan.scratch[offset..offset + plan.size]
                        .copy_from_slice(&plan.registers[value][..plan.size]);
                }
            }
        }
    }
    // No checked arithmetic or allocation can fail during publication. Every writable view was
    // validated before execution; only the proved exact prefix is copied and tails are retained.
    for (index, resource) in plan.resources.iter().enumerate() {
        if let Some(start) = resource.shadow {
            arguments[positions[index]]
                .bytes_mut()
                .expect("complete typed preflight proved every writable transport resource")
                [..resource.write_bytes]
                .copy_from_slice(&plan.scratch[start..start + resource.write_bytes]);
        }
    }
    Ok(())
}
