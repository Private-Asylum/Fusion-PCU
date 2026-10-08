//! Owned, immutable proof boundary for detached CPU instructions and resource spans.
use alloc::vec::Vec;
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuScalar,
    PcuScalarType,
};
#[rustfmt::skip]
use super::{
    execution,
    Executable,
    PcuCpuComposedMapError as Error,
    Resource,
    Step,
    BINDINGS,
    STEPS,
};

#[path = "integer/integer.rs"]
mod integer;
pub(super) use integer::Instruction as IntegerInstruction;

/// Fields are private to this module. Construction checks the complete partitioned
/// program; shared views and Clone cannot invalidate the proof. Scratch and call-local
/// resource bases are deliberately outside this immutable object.
#[derive(Debug, Clone)]
pub(super) struct ValidatedProgram {
    scalar: PcuScalarType,
    size: usize,
    extent: usize,
    schema: Vec<(PcuBindingRef, usize, PcuBindingAccess)>,
    resources: Vec<Resource>,
    initializers: Vec<Step>,
    steps: Vec<Step>,
    scratch_len: usize,
    execute: Executable,
    integer_instructions: Option<Vec<IntegerInstruction>>,
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;

impl ValidatedProgram {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        scalar: PcuScalarType,
        size: usize,
        extent: usize,
        schema: Vec<(PcuBindingRef, usize, PcuBindingAccess)>,
        resources: Vec<Resource>,
        initializers: Vec<Step>,
        steps: Vec<Step>,
        scratch_len: usize,
    ) -> Result<Self, Error> {
        let (carrier_size, integer) = carrier(scalar)?;
        if size != carrier_size
            || !(2..=BINDINGS).contains(&schema.len())
            || resources.len() > BINDINGS
            || initializers
                .len()
                .checked_add(steps.len())
                .ok_or(Error::ExtentOverflow)?
                > STEPS
        {
            return Err(Error::InvalidProgram);
        }
        let lane_bytes = extent.checked_mul(size).ok_or(Error::ExtentOverflow)?;
        validate_resources(&schema, &resources, size, lane_bytes, scratch_len)?;
        validate_steps(
            &schema,
            &resources,
            &initializers,
            &steps,
            size,
            lane_bytes,
            integer,
        )?;
        let integer_instructions = if integer {
            integer::lower(extent, &resources, &steps)?
        } else {
            None
        };
        let execute = if integer && integer_instructions.is_some() {
            execution::integer::select(scalar)
        } else if integer {
            execution::integer::select_scalar(scalar)
        } else {
            execution::select(scalar)
        }
        .ok_or(Error::InvalidProgram)?;
        Ok(Self {
            scalar,
            size,
            extent,
            schema,
            resources,
            initializers,
            steps,
            scratch_len,
            execute,
            integer_instructions,
        })
    }
    pub(super) const fn scalar(&self) -> PcuScalarType {
        self.scalar
    }
    pub(super) const fn size(&self) -> usize {
        self.size
    }
    pub(super) const fn extent(&self) -> usize {
        self.extent
    }
    pub(super) const fn schema(&self) -> &[(PcuBindingRef, usize, PcuBindingAccess)] {
        self.schema.as_slice()
    }
    pub(super) fn resources(&self) -> &[Resource] {
        &self.resources
    }
    pub(super) fn initializers(&self) -> &[Step] {
        &self.initializers
    }
    pub(super) fn steps(&self) -> &[Step] {
        &self.steps
    }
    pub(super) fn integer_instructions(&self) -> Option<&[IntegerInstruction]> {
        self.integer_instructions.as_deref()
    }
    #[cfg(test)]
    pub(super) fn use_scalar_integer_executor(&mut self) {
        self.execute =
            execution::integer::select_scalar(self.scalar).expect("integer test program");
    }
    pub(super) const fn executor(&self) -> Executable {
        self.execute
    }
    pub(super) fn validate_carrier<T: PcuScalar>(&self, scratch_len: usize) -> Result<(), Error> {
        if T::TYPE == self.scalar && T::HOST_SIZE == self.size && scratch_len == self.scratch_len {
            Ok(())
        } else {
            Err(Error::InvalidProgram)
        }
    }
}

const fn carrier(scalar: PcuScalarType) -> Result<(usize, bool), Error> {
    Ok(match scalar {
        PcuScalarType::F8E4M3FN | PcuScalarType::F8E5M2 => (1, false),
        PcuScalarType::F16 | PcuScalarType::BF16 => (2, false),
        PcuScalarType::F32 => (4, false),
        PcuScalarType::F64 => (8, false),
        PcuScalarType::I8 | PcuScalarType::U8 => (1, true),
        PcuScalarType::I16 | PcuScalarType::U16 => (2, true),
        PcuScalarType::I32 | PcuScalarType::U32 => (4, true),
        PcuScalarType::I64 | PcuScalarType::U64 => (8, true),
        PcuScalarType::I128 | PcuScalarType::U128 => (16, true),
        PcuScalarType::I256 | PcuScalarType::U256 => (32, true),
        PcuScalarType::I512 | PcuScalarType::U512 => (64, true),
        _ => return Err(Error::InvalidProgram),
    })
}

fn validate_resources(
    schema: &[(PcuBindingRef, usize, PcuBindingAccess)],
    resources: &[Resource],
    size: usize,
    lane_bytes: usize,
    scratch_len: usize,
) -> Result<(), Error> {
    for (index, &(binding, elements, access)) in schema.iter().enumerate() {
        elements.checked_mul(size).ok_or(Error::ExtentOverflow)?;
        if !matches!(
            access,
            PcuBindingAccess::ReadOnly | PcuBindingAccess::ReadWrite
        ) || schema[..index].iter().any(|entry| entry.0 == binding)
        {
            return Err(Error::InvalidProgram);
        }
    }
    for (index, resource) in resources.iter().enumerate() {
        let &(binding, elements, access) = schema
            .get(resource.declaration)
            .ok_or(Error::InvalidProgram)?;
        let length = resource.read_bytes.max(resource.write_bytes);
        if binding != resource.binding
            || resource.read_bytes % size != 0
            || resource.write_bytes % size != 0
            || length > elements.checked_mul(size).ok_or(Error::ExtentOverflow)?
            || resources[..index]
                .iter()
                .any(|other| other.declaration == resource.declaration)
            || (resource.write_bytes != 0
                && (access != PcuBindingAccess::ReadWrite || resource.write_bytes != lane_bytes))
            || resource.shadow.is_some() != (resource.write_bytes != 0)
        {
            return Err(Error::InvalidProgram);
        }
        if let Some(start) = resource.shadow {
            let end = start.checked_add(length).ok_or(Error::ExtentOverflow)?;
            if end > scratch_len {
                return Err(Error::InvalidProgram);
            }
            for other in &resources[..index] {
                if let Some(other_start) = other.shadow {
                    let other_end = other_start
                        .checked_add(other.read_bytes.max(other.write_bytes))
                        .ok_or(Error::ExtentOverflow)?;
                    if start < other_end && other_start < end {
                        return Err(Error::InvalidProgram);
                    }
                }
            }
        }
    }
    Ok(())
}

fn validate_steps(
    schema: &[(PcuBindingRef, usize, PcuBindingAccess)],
    resources: &[Resource],
    initializers: &[Step],
    steps: &[Step],
    size: usize,
    lane_bytes: usize,
    integer: bool,
) -> Result<(), Error> {
    let mut defined = [false; STEPS];
    let mut stored = [false; BINDINGS];
    for (initializer, stream) in [(true, initializers), (false, steps)] {
        for step in stream {
            let read = |value: usize| -> Result<(), Error> {
                if defined.get(value).copied() == Some(true) {
                    Ok(())
                } else {
                    Err(Error::InvalidProgram)
                }
            };
            let result = match *step {
                Step::Load {
                    result,
                    resource,
                    zero,
                } => {
                    let resource = resources.get(resource).ok_or(Error::InvalidProgram)?;
                    if resource.read_bytes < if zero { size } else { lane_bytes }
                        || (initializer && (!zero || resource.write_bytes != 0))
                    {
                        return Err(Error::InvalidProgram);
                    }
                    Some(result)
                }
                Step::Constant { result, .. } if initializer && !integer && size <= 8 => {
                    Some(result)
                }
                Step::IntegerConstant { result, .. } if initializer && integer && size <= 16 => {
                    Some(result)
                }
                Step::Binary {
                    result,
                    left,
                    right,
                    ..
                } if !initializer && !integer => {
                    read(left)?;
                    read(right)?;
                    Some(result)
                }
                Step::IntegerBinary {
                    result,
                    left,
                    right,
                    ..
                } if !initializer && integer => {
                    read(left)?;
                    read(right)?;
                    Some(result)
                }
                Step::Unary { result, value, .. } if !initializer && !integer => {
                    read(value)?;
                    Some(result)
                }
                Step::Store { resource, value } if !initializer => {
                    read(value)?;
                    let resource_index = resource;
                    let resource = resources.get(resource).ok_or(Error::InvalidProgram)?;
                    stored[resource_index] = true;
                    if schema[resource.declaration].2 != PcuBindingAccess::ReadWrite
                        || resource.write_bytes < lane_bytes
                        || (lane_bytes != 0 && resource.shadow.is_none())
                    {
                        return Err(Error::InvalidProgram);
                    }
                    None
                }
                _ => return Err(Error::InvalidProgram),
            };
            if let Some(result) = result {
                let slot = defined.get_mut(result).ok_or(Error::InvalidProgram)?;
                if *slot {
                    return Err(Error::InvalidProgram);
                }
                *slot = true;
            }
        }
    }
    if resources
        .iter()
        .enumerate()
        .any(|(index, resource)| resource.write_bytes != 0 && !stored[index])
    {
        return Err(Error::InvalidProgram);
    }
    Ok(())
}
