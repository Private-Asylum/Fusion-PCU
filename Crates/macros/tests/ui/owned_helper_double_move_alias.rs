use fusion_pcu_macros::pcu;

#[pcu(crate_path = ::pcu_alias)]
fn consuming(
    input: pcu_alias::PcuTensor<f32>,
) -> Result<pcu_alias::PcuTensor<f32>, pcu_alias::PcuExecutionError> {
    Ok(pcu::relu(input)?)
}

#[pcu(crate_path = ::pcu_alias)]
fn invalid(
    input: pcu_alias::PcuTensor<f32>,
) -> Result<pcu_alias::PcuTensor<f32>, pcu_alias::PcuExecutionError> {
    let owner = input;
    let alias = owner;
    let first = consuming(alias)?;
    Ok(pcu::add(first, owner)?)
}

fn main() {}
