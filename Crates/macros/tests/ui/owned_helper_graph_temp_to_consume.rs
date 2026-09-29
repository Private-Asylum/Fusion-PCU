use fusion_pcu_macros::pcu;

#[pcu(crate_path = ::pcu_alias)]
fn consuming(
    input: pcu_alias::PcuTensor<f32>,
) -> Result<pcu_alias::PcuTensor<f32>, pcu_alias::PcuExecutionError> {
    Ok(pcu::relu(input)?)
}

#[pcu(crate_path = ::pcu_alias)]
fn valid(
    input: pcu_alias::PcuTensor<f32>,
) -> Result<pcu_alias::PcuTensor<f32>, pcu_alias::PcuExecutionError> {
    let multiplied = pcu::mul(&input, &input)?;
    let added = pcu::add(multiplied, input)?;
    let activated = pcu::relu(added)?;
    Ok(consuming(activated)?)
}

fn main() {}
