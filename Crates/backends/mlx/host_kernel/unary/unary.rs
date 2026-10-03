//! Detached exact unary assessment; native ownership stays with the retained primitive.
#[rustfmt::skip]
use fusion_pcu::{
    describe_checked_float_unary_map,
    describe_portable_v1_unary_map,
    PcuCheckedFloatUnaryMapDescription,
    PcuDispatchKernelIr,
};
#[rustfmt::skip]
use super::{
    MlxCheckedUnaryPlan,
    MlxError,
};
pub(super) fn portable_plan(
    kernel: &PcuDispatchKernelIr<'_>,
) -> Result<MlxCheckedUnaryPlan, MlxError> {
    if cfg!(target_endian = "big") {
        return Err(MlxError::InvalidRequest(
            "MLX Portable unary requires little-endian exact carriers".into(),
        ));
    }
    let description = describe_portable_v1_unary_map(kernel)
        .map_err(|_| MlxError::InvalidRequest("unsupported MLX Portable unary profile".into()))?;
    from_description(description)
}

pub(super) fn normal_plan(
    kernel: &PcuDispatchKernelIr<'_>,
) -> Result<MlxCheckedUnaryPlan, MlxError> {
    if cfg!(target_endian = "big") {
        return Err(MlxError::InvalidRequest(
            "MLX checked unary requires little-endian exact carriers".into(),
        ));
    }
    let description = describe_checked_float_unary_map(kernel)
        .map_err(|_| MlxError::InvalidRequest("unsupported MLX checked unary profile".into()))?;
    from_description(description)
}

fn from_description(
    description: PcuCheckedFloatUnaryMapDescription,
) -> Result<MlxCheckedUnaryPlan, MlxError> {
    let count = usize::try_from(description.logical_extent).map_err(|_| MlxError::InvalidExtent)?;
    let width = usize::from(description.scalar.bit_width()) / 8;
    let output_bytes = count.checked_mul(width).ok_or(MlxError::InvalidExtent)?;
    i32::try_from(output_bytes / width.min(4)).map_err(|_| MlxError::InvalidExtent)?;
    isize::try_from(output_bytes).map_err(|_| MlxError::InvalidExtent)?;
    let status = count.checked_mul(4).ok_or(MlxError::InvalidExtent)?;
    isize::try_from(status).map_err(|_| MlxError::InvalidExtent)?;
    Ok(MlxCheckedUnaryPlan {
        scalar: description.scalar,
        operation: description.operation,
        underflow: description.requirements.float_underflow,
        range: description.requirements.range_policy,
        count,
        broadcast: description.broadcast_input,
        input: description.input_binding,
        output: description.output_binding,
        input_bytes: if description.broadcast_input {
            width
        } else {
            output_bytes
        },
        output_bytes,
        requirements: description.requirements,
    })
}
