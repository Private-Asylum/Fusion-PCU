use fusion_pcu_macros::pcu;

#[pcu(crate_path = ::pcu_alias)]
fn borrowed(input: &[f32]) -> Result<pcu_alias::PcuTensor<f32>, pcu_alias::PcuExecutionError> {
    Ok(pcu::identity(input)?)
}

#[pcu(crate_path = ::pcu_alias)]
fn invalid(
    input: pcu_alias::PcuTensor<f32>,
) -> Result<pcu_alias::PcuTensor<f32>, pcu_alias::PcuExecutionError> {
    Ok(borrowed(input)?)
}

fn main() {}
