use fusion_pcu_macros::pcu;

#[pcu(crate_path = ::pcu_alias)]
fn invalid(
    input: pcu_alias::PcuTensor<f32>,
) -> Result<pcu_alias::PcuTensor<f32>, pcu_alias::PcuExecutionError> {
    let owner = input;
    let first = pcu::relu(owner)?;
    let owner = pcu::identity(owner)?;
    Ok(pcu::add(first, owner)?)
}

fn main() {}
