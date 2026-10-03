//! Exact cold original declarations and unique native operand/output roles.
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuScalarType,
};
#[rustfmt::skip]
use fusion_pcu_spirv::{
    PcuSpirvBitMapProfile,
    PcuSpirvCheckedUnaryProfile,
};
#[rustfmt::skip]
use crate::{
    PcuVulkanPreparedHost,
    PcuVulkanError,
};
#[derive(Clone, Copy)]
pub(super) enum Role {
    Input(usize),
    Output(usize),
    Unused,
}
pub(super) struct Binding {
    pub(super) target: PcuBindingRef,
    pub(super) scalar: PcuScalarType,
    pub(super) access: PcuBindingAccess,
    pub(super) bytes: usize,
    pub(super) role: Role,
}
pub(super) struct Schema {
    pub(super) bindings: Vec<Binding>,
    pub(super) input_count: usize,
    pub(super) output_count: usize,
}
fn binding(
    target: PcuBindingRef,
    scalar: PcuScalarType,
    count: u32,
    role: Role,
) -> Result<Binding, PcuVulkanError> {
    let bytes = usize::try_from(count)
        .ok()
        .and_then(|count| count.checked_mul(usize::from(scalar.bit_width() / 8)))
        .ok_or(PcuVulkanError::BufferTooLarge)?;
    let access = if matches!(role, Role::Output(_)) {
        PcuBindingAccess::ReadWrite
    } else {
        PcuBindingAccess::ReadOnly
    };
    Ok(Binding {
        target,
        scalar,
        access,
        bytes,
        role,
    })
}
fn unary(
    input: PcuBindingRef,
    output: PcuBindingRef,
    source: PcuScalarType,
    destination: PcuScalarType,
    input_count: u32,
    output_count: u32,
) -> Result<Schema, PcuVulkanError> {
    Ok(Schema {
        bindings: vec![
            binding(input, source, input_count, Role::Input(0))?,
            binding(output, destination, output_count, Role::Output(0))?,
        ],
        input_count: 1,
        output_count: 1,
    })
}
pub(super) fn compile(plan: &PcuVulkanPreparedHost) -> Result<Schema, PcuVulkanError> {
    match plan {
        PcuVulkanPreparedHost::ScalarTransport(plan) => {
            let p = plan.profile();
            unary(
                p.input,
                p.output,
                p.scalar,
                p.scalar,
                p.input_extent(),
                p.extent,
            )
        }
        PcuVulkanPreparedHost::Unary(plan) => {
            let p = plan.profile();
            unary(
                PcuSpirvCheckedUnaryProfile::INPUT,
                PcuSpirvCheckedUnaryProfile::OUTPUT,
                p.scalar,
                p.scalar,
                p.input_extent(),
                p.extent,
            )
        }
        PcuVulkanPreparedHost::UnaryRoles(plan) => {
            let profile = plan.profile();
            let description = profile.description;
            let bindings = plan
                .declaration_schema()
                .iter()
                .map(|&(target, access, bytes)| {
                    let role = if target == description.input_binding {
                        Role::Input(0)
                    } else if target == description.output_binding {
                        Role::Output(0)
                    } else {
                        Role::Unused
                    };
                    Binding {
                        target,
                        scalar: profile.native.scalar,
                        access,
                        bytes,
                        role,
                    }
                })
                .collect();
            Ok(Schema {
                bindings,
                input_count: 1,
                output_count: 1,
            })
        }
        PcuVulkanPreparedHost::Conversion(plan) => {
            let p = plan.profile();
            unary(
                p.input,
                p.output,
                p.source_scalar(),
                p.output_scalar(),
                p.input_extent(),
                p.extent,
            )
        }
        PcuVulkanPreparedHost::BitMap(plan) => {
            let p = plan.profile();
            unary(
                PcuSpirvBitMapProfile::INPUT,
                PcuSpirvBitMapProfile::OUTPUT,
                p.scalar,
                p.scalar,
                p.extent,
                p.extent,
            )
        }
        PcuVulkanPreparedHost::Binary(plan) => binary(plan),
        PcuVulkanPreparedHost::Integer(plan) => integer(plan),
        PcuVulkanPreparedHost::DivRem(plan) => div_rem(plan),
        PcuVulkanPreparedHost::Composed(_) | PcuVulkanPreparedHost::OrderedTransport(_) => {
            Err(PcuVulkanError::UnsupportedPreparedProfile)
        }
    }
}
fn binary(plan: &crate::PcuVulkanPreparedBinary) -> Result<Schema, PcuVulkanError> {
    let p = plan.profile();
    let mut bindings = Vec::with_capacity(p.declaration_count);
    for &target in &p.declarations[..p.declaration_count] {
        let (count, role) = if target == p.output {
            (p.extent, Role::Output(0))
        } else if let Some(index) = p.inputs[..p.input_count]
            .iter()
            .position(|input| *input == target)
        {
            (p.input_extent(index), Role::Input(index))
        } else {
            (0, Role::Unused)
        };
        bindings.push(binding(target, p.scalar, count, role)?);
    }
    Ok(Schema {
        bindings,
        input_count: p.input_count,
        output_count: 1,
    })
}

fn integer(plan: &crate::PcuVulkanPreparedInteger) -> Result<Schema, PcuVulkanError> {
    let p = plan.profile();
    let mut bindings = Vec::with_capacity(p.declaration_count);
    for &target in &p.declarations[..p.declaration_count] {
        let (count, role) = if target == p.output {
            (p.extent, Role::Output(0))
        } else if let Some(index) = p.inputs[..p.input_count]
            .iter()
            .position(|input| *input == target)
        {
            (p.input_extent(index), Role::Input(index))
        } else {
            (0, Role::Unused)
        };
        bindings.push(binding(target, p.scalar, count, role)?);
    }
    Ok(Schema {
        bindings,
        input_count: p.input_count,
        output_count: 1,
    })
}

fn div_rem(plan: &crate::PcuVulkanPreparedDivRem) -> Result<Schema, PcuVulkanError> {
    let p = plan.profile();
    let mut bindings = Vec::with_capacity(p.declaration_count);
    for &target in &p.declarations[..p.declaration_count] {
        let (count, role) = p
            .outputs
            .iter()
            .position(|output| *output == target)
            .map_or_else(
                || {
                    p.inputs[..p.input_count]
                        .iter()
                        .position(|input| *input == target)
                        .map_or((0, Role::Unused), |index| {
                            (p.input_extents[index], Role::Input(index))
                        })
                },
                |index| (p.extent, Role::Output(index)),
            );
        bindings.push(binding(target, p.scalar, count, role)?);
    }
    Ok(Schema {
        bindings,
        input_count: p.input_count,
        output_count: 2,
    })
}
