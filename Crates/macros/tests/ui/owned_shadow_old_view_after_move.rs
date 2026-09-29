use fusion_pcu_macros::pcu;

#[pcu(crate_path = ::pcu_alias)]
fn invalid(
    input: pcu_alias::PcuTensor<f32>,
) -> Result<pcu_alias::PcuTensor<f32>, pcu_alias::PcuExecutionError> {
    let owner = input;
    let old_view = &owner;
    let owner = pcu::relu(owner)?;
    let copied = pcu::identity(old_view)?;
    Ok(pcu::add(copied, owner)?)
}

fn main() {}
