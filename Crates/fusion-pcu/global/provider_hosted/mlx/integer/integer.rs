//! Cold checked-integer shape preparation, independent from float admission.
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxCheckedIntegerPlan,
    MlxSession,
};
#[rustfmt::skip]
use crate::{
    PcuDispatchKernelIr,
    PcuExecutionError,
    PcuHostKernelBackend,
};
#[rustfmt::skip]
use super::{
    map_mlx_error,
    readonly_declarations,
    MlxInputLayout,
    MlxKernel,
};

pub(super) fn prepare(
    session: &MlxSession,
    source: &PcuDispatchKernelIr<'_>,
    inputs: MlxInputLayout,
    plan: &MlxCheckedIntegerPlan,
) -> Result<MlxKernel, PcuExecutionError> {
    let minimum = plan.input_element_counts();
    let extents = inputs.extents(plan.input_bindings(), minimum)?;
    let backend = session.checked_integer_backend();
    // Wide limb carriers retain their full native backing too. Logical spans
    // and checked range semantics remain the integer plan's original contract.
    let kernel = if extents == minimum {
        backend.prepare_host_kernel(source)
    } else {
        backend
            .prepare_host_kernel_with_input_extents(source, &extents[..plan.input_bindings().len()])
    }
    .map_err(map_mlx_error)?;
    let (declarations, declaration_count) =
        readonly_declarations(source, kernel.input_bindings()[0]);
    Ok(MlxKernel::Integer {
        kernel,
        declarations,
        declaration_count,
    })
}
