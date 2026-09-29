use fusion_pcu_macros::pcu;

#[pcu(crate_path = ::pcu_alias)]
fn invalid(
    lhs: pcu_alias::PcuTensor<f32>,
    rhs: pcu_alias::PcuTensor<f32>,
) -> Result<pcu_alias::PcuTensor<f32>, pcu_alias::PcuExecutionError> {
    let relu = pcu::relu(rhs)?;
    let first = pcu::add(lhs, relu)?;
    Ok(pcu::add(first, rhs)?)
}

fn main() {}
