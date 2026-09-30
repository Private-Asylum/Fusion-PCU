use fusion_pcu_macros::pcu;

#[pcu(crate_path = ::pcu_alias, flag(clamp_range))]
fn unsupported(input: &[f32]) -> Result<pcu_alias::PcuTensor<f32>, pcu_alias::PcuExecutionError> {
    Ok(pcu::identity(input)?)
}

fn main() {}
