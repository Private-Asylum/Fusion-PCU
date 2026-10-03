//! Cold actual operand and declaration projection; no warm IR or resource reconstruction.
#[rustfmt::skip]
use fusion_pcu::{
    assess_checked_integer_div_rem_operands,
    validate_integer_checked_div_rem_kernel,
    IntegerMapValidationError,
    PcuBindingAccess,
    PcuCheckedIntegerDivision,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuValueType,
    PcuValueTypeCaps,
};
#[rustfmt::skip]
use super::{
    bytes,
    integer_id,
    PcuCpuCheckedIntegerError as Error,
    PcuCpuPreparedDivRem,
};

pub(super) fn prepare<T: PcuCheckedIntegerDivision>(
    kernel: &PcuDispatchKernelIr<'_>,
) -> Result<PcuCpuPreparedDivRem, Error> {
    let legacy_id = integer_id(T::TYPE).ok_or(Error::UnsupportedProfile)?;
    if kernel.numerical_requirements.range_policy != fusion_pcu::PcuRangePolicy::Reject {
        return Err(Error::UnsupportedProfile);
    }
    let portable = kernel
        .numerical_requirements
        .numerical_options
        .reproducibility
        == fusion_pcu::PcuReproducibility::PortableV1;
    if portable {
        fusion_pcu::describe_portable_v1_integer_div_rem_map(kernel)
            .map_err(|_| Error::UnsupportedProfile)?;
    }
    // Only the already descriptor-proved exact Portable joint map may use this
    // structural view. Original requirements remain authoritative for offers/cache/IDs.
    let mut structural = *kernel;
    structural
        .numerical_requirements
        .numerical_options
        .reproducibility = fusion_pcu::PcuReproducibility::Unspecified;
    let (_, extent) = crate::checked_integer::validated_integer_region(&structural)?;
    let value_type = PcuValueType::Scalar(T::TYPE);
    let caps = PcuValueTypeCaps::for_scalar(T::TYPE);
    let roles =
        assess_checked_integer_div_rem_operands(kernel, value_type, caps).map_err(profile_error)?;
    let legacy = validate_integer_checked_div_rem_kernel(kernel, value_type, caps).is_ok();
    let extent = usize::try_from(extent).map_err(|_| Error::UnsupportedProfile)?;
    let counts = roles.input_element_counts(extent);
    let loaded = roles.input_bindings();
    let first = loaded.first().copied().ok_or(Error::UnsupportedProfile)?;
    let inputs = [first, loaded.get(1).copied().unwrap_or(first)];
    let outputs = roles.output_bindings();
    let mut schema = [(outputs[0], 0, PcuBindingAccess::ReadOnly); 4];
    for (slot, binding) in kernel.bindings.iter().enumerate() {
        let target = binding.reference();
        schema[slot] = if outputs.contains(&target) {
            (target, extent, PcuBindingAccess::ReadWrite)
        } else {
            let count = loaded
                .iter()
                .position(|input| *input == target)
                .map_or(0, |input| counts[input]);
            (target, count, PcuBindingAccess::ReadOnly)
        };
    }
    let indices = roles.operand_indices();
    Ok(PcuCpuPreparedDivRem {
        schema,
        argument_count: kernel.bindings.len(),
        inputs,
        outputs,
        operands: roles.operand_inputs(),
        extent,
        scalar: T::TYPE,
        local_id: if portable {
            12288 + role_offset(legacy_id)?
        } else if legacy {
            legacy_id
        } else {
            8192 + role_offset(legacy_id)?
        },
        size: T::HOST_SIZE,
        execute: bytes::prepare::<T>(
            indices[0] == PcuDispatchIndex::BindingElementZero,
            indices[1] == PcuDispatchIndex::BindingElementZero,
        ),
    })
}

const fn profile_error(error: IntegerMapValidationError) -> Error {
    match error {
        IntegerMapValidationError::UnsupportedInterface
        | IntegerMapValidationError::UnsupportedRequirements
        | IntegerMapValidationError::UnsupportedOperation(_) => Error::UnsupportedProfile,
        other => Error::InvalidKernel(other),
    }
}

const fn role_offset(id: u32) -> Result<u32, Error> {
    match id {
        128..=135 => Ok(id - 128),
        512..=517 => Ok(id - 512 + 8),
        _ => Err(Error::UnsupportedProfile),
    }
}
