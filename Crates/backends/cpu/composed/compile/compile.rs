//! One cold pass retains actual resource/SSA routing and lexical arithmetic policies.
use alloc::vec::Vec;
#[rustfmt::skip]
use fusion_pcu::{
    assess_checked_float_map_resources,
    CheckedScalarMapResourceSchema,
    PcuBindingAccess,
    PcuBindingRef,
    PcuCheckedFloat,
    PcuDispatchDataOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuParameterValue,
    PcuReproducibility,
    PcuScalarType,
    PcuValueType,
    PcuValueTypeCaps,
};
#[rustfmt::skip]
use super::{
    execution,
    BINDINGS,
    REGISTERS,
    STEPS,
    Executable,
    PcuCpuComposedMapError as Error,
    PcuCpuPreparedComposedMap as Plan,
    Resource,
    Step,
};

pub(super) fn prepare<T: PcuCheckedFloat>(kernel: &PcuDispatchKernelIr<'_>) -> Result<Plan, Error> {
    if kernel
        .numerical_requirements
        .numerical_options
        .reproducibility
        != PcuReproducibility::Unspecified
        || !(2..=BINDINGS).contains(&kernel.bindings.len())
    {
        return Err(Error::UnsupportedProfile);
    }
    let (mut local_id, size, execute) = realization(T::TYPE)?;
    let descriptor = assess_checked_float_map_resources::<BINDINGS>(
        kernel,
        PcuValueType::Scalar(T::TYPE),
        PcuValueTypeCaps::for_scalar(T::TYPE),
    )
    .map_err(Error::InvalidResources)?;
    let body = crate::host::validated_region(kernel).map_err(|_| Error::UnsupportedProfile)?;
    let checked_operations = body
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
    if body.len() > STEPS || checked_operations == 0 {
        return Err(Error::UnsupportedProfile);
    }
    if checked_operations == 1 {
        // Separately identified one-effect profile; old multi-operation IDs stay exact.
        local_id = 18688 + (local_id - 17408);
    }
    build(
        kernel,
        descriptor,
        body,
        (T::TYPE, local_id, size, execute),
        step,
    )
}

pub(super) fn build(
    kernel: &PcuDispatchKernelIr<'_>,
    descriptor: CheckedScalarMapResourceSchema<BINDINGS>,
    body: &[PcuDispatchOp<'_>],
    realization: (PcuScalarType, u32, usize, Executable),
    lower: fn(PcuDispatchOp<'_>, &[Resource]) -> Result<Step, Error>,
) -> Result<Plan, Error> {
    let (scalar, local_id, size, _execute) = realization;
    let mut resources = Vec::new();
    resources
        .try_reserve_exact(descriptor.resources().len())
        .map_err(|_| Error::AllocationFailed)?;
    let mut bytes = 0_usize;
    for resource in descriptor.resources() {
        // The first profile does not invent snapshot semantics for a cross-index dependency.
        if resource.has_cross_index_read_write() {
            return Err(Error::UnsupportedProfile);
        }
        let declaration = kernel
            .bindings
            .iter()
            .position(|binding| binding.reference() == resource.binding)
            .ok_or(Error::InvalidProgram)?;
        let read_bytes = extent_bytes(resource.minimum_read_elements, size)?;
        let write_bytes = extent_bytes(resource.minimum_write_elements, size)?;
        let shadow = if write_bytes == 0 {
            None
        } else {
            let start = bytes;
            bytes = bytes
                .checked_add(read_bytes.max(write_bytes))
                .ok_or(Error::ExtentOverflow)?;
            Some(start)
        };
        resources.push(Resource {
            binding: resource.binding,
            declaration,
            read_bytes,
            write_bytes,
            shadow,
        });
    }
    let schema = schema(kernel, &resources, size)?;
    let mut steps = Vec::new();
    steps
        .try_reserve_exact(body.len())
        .map_err(|_| Error::AllocationFailed)?;
    for operation in body {
        steps.push(lower(*operation, &resources)?);
    }
    densify(&mut steps)?;
    let invariant = |step: &Step| match *step {
        Step::Constant { .. } | Step::IntegerConstant { .. } => true,
        Step::Load {
            resource,
            zero: true,
            ..
        } => resources[resource].write_bytes == 0,
        _ => false,
    };
    let mut initializers = Vec::new();
    initializers
        .try_reserve_exact(steps.iter().filter(|step| invariant(step)).count())
        .map_err(|_| Error::AllocationFailed)?;
    // Only literals and loads from immutable, disjoint resources may leave the hot
    // stream. Runtime values are refreshed after every call's argument preflight.
    steps.retain(|step| {
        if invariant(step) {
            initializers.push(*step);
            false
        } else {
            true
        }
    });
    let mut scratch = Vec::new();
    scratch
        .try_reserve_exact(bytes)
        .map_err(|_| Error::AllocationFailed)?;
    scratch.resize(bytes, 0);
    let program = super::program::ValidatedProgram::new(
        scalar,
        size,
        usize::try_from(descriptor.logical_extent).map_err(|_| Error::ExtentOverflow)?,
        schema,
        resources,
        initializers,
        steps,
        scratch.len(),
    )?;
    Ok(Plan {
        requirements: kernel.numerical_requirements,
        local_id,
        program,
        scratch,
    })
}
fn realization(scalar: PcuScalarType) -> Result<(u32, usize, Executable), Error> {
    let (id, size) = match scalar {
        PcuScalarType::F16 => (17408, 2),
        PcuScalarType::BF16 => (17409, 2),
        PcuScalarType::F8E4M3FN => (17410, 1),
        PcuScalarType::F8E5M2 => (17411, 1),
        PcuScalarType::F32 => (17412, 4),
        PcuScalarType::F64 => (17413, 8),
        _ => return Err(Error::UnsupportedProfile),
    };
    Ok((
        id,
        size,
        execution::select(scalar).ok_or(Error::UnsupportedProfile)?,
    ))
}
fn extent_bytes(elements: u32, size: usize) -> Result<usize, Error> {
    usize::try_from(elements)
        .ok()
        .and_then(|n| n.checked_mul(size))
        .ok_or(Error::ExtentOverflow)
}
fn schema(
    kernel: &PcuDispatchKernelIr<'_>,
    resources: &[Resource],
    size: usize,
) -> Result<Vec<(PcuBindingRef, usize, PcuBindingAccess)>, Error> {
    let mut schema = Vec::new();
    schema
        .try_reserve_exact(kernel.bindings.len())
        .map_err(|_| Error::AllocationFailed)?;
    for binding in kernel.bindings {
        // Host arguments have actual shared or mutable Rust borrows, not a forged WriteOnly view.
        if !matches!(
            binding.access,
            PcuBindingAccess::ReadOnly | PcuBindingAccess::ReadWrite
        ) {
            return Err(Error::UnsupportedProfile);
        }
        let elements = resources
            .iter()
            .find(|resource| resource.binding == binding.reference())
            .map_or(0, |resource| {
                resource.read_bytes.max(resource.write_bytes) / size
            });
        schema.push((binding.reference(), elements, binding.access));
    }
    Ok(schema)
}
fn slot(value: fusion_pcu::PcuDispatchValueId) -> Result<usize, Error> {
    let index = usize::from(value.0);
    if index >= REGISTERS {
        return Err(Error::UnsupportedProfile);
    }
    Ok(index)
}
fn resource(binding: PcuBindingRef, resources: &[Resource]) -> Result<usize, Error> {
    resources
        .iter()
        .position(|resource| resource.binding == binding)
        .ok_or(Error::InvalidProgram)
}
fn step(operation: PcuDispatchOp<'_>, resources: &[Resource]) -> Result<Step, Error> {
    match operation {
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result,
            binding,
            index,
        }) => Ok(Step::Load {
            result: slot(result)?,
            resource: resource(binding, resources)?,
            zero: index == PcuDispatchIndex::BindingElementZero,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::Constant { result, value }) => {
            let mut bytes = [0_u8; 8];
            match value {
                PcuParameterValue::F32(bits) => bytes[..4].copy_from_slice(&bits.to_ne_bytes()),
                PcuParameterValue::F64(bits) => bytes.copy_from_slice(&bits.to_ne_bytes()),
                _ => return Err(Error::UnsupportedProfile),
            }
            Ok(Step::Constant {
                result: slot(result)?,
                bytes,
            })
        }
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
            result,
            op,
            lhs,
            rhs,
            range_policy,
            underflow_policy,
            ..
        }) => Ok(Step::Binary {
            result: slot(result)?,
            left: slot(lhs)?,
            right: slot(rhs)?,
            operation: op,
            range: range_policy,
            underflow: underflow_policy,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
            result,
            op,
            value,
            range_policy,
            underflow_policy,
            ..
        }) => Ok(Step::Unary {
            result: slot(result)?,
            value: slot(value)?,
            operation: op,
            range: range_policy,
            underflow: underflow_policy,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore { binding, value, .. }) => {
            Ok(Step::Store {
                resource: resource(binding, resources)?,
                value: slot(value)?,
            })
        }
        _ => Err(Error::UnsupportedProfile),
    }
}

#[path = "integer/integer.rs"]
pub(super) mod integer;

// Verify the detached program itself before removing warm register-validity checks.
// Slots are assigned by definition order, independent of the source SSA numbering.
fn densify(steps: &mut [Step]) -> Result<(), Error> {
    // All source slots passed `slot` during lowering and are below REGISTERS.
    // At most one definition per step makes dense slots strictly below STEPS.
    let mut slots = [None; REGISTERS];
    let mut next = 0;
    for step in steps {
        let result = match step {
            Step::Load { result, .. }
            | Step::Constant { result, .. }
            | Step::IntegerConstant { result, .. } => Some(result),
            Step::Binary {
                result,
                left,
                right,
                ..
            }
            | Step::IntegerBinary {
                result,
                left,
                right,
                ..
            } => {
                *left = slots[*left].ok_or(Error::InvalidProgram)?;
                *right = slots[*right].ok_or(Error::InvalidProgram)?;
                Some(result)
            }
            Step::Unary { result, value, .. } => {
                *value = slots[*value].ok_or(Error::InvalidProgram)?;
                Some(result)
            }
            Step::Store { value, .. } => {
                *value = slots[*value].ok_or(Error::InvalidProgram)?;
                None
            }
        };
        if let Some(result) = result {
            if slots[*result].is_some() {
                return Err(Error::InvalidProgram);
            }
            slots[*result] = Some(next);
            *result = next;
            next += 1;
        }
    }
    Ok(())
}
