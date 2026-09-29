use fusion_pcu_macros::pcu;

#[pcu(crate_path = ::pcu_alias)]
fn accepted(
    input: pcu_alias::PcuTensor<f32>,
) -> Result<pcu_alias::PcuTensor<f32>, pcu_alias::PcuExecutionError> {
    let owner = input;
    let alias = owner;
    Ok(pcu::relu(alias)?)
}

fn main() {}
