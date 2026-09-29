use fusion_pcu_macros::pcu;

#[pcu(crate_path = ::pcu_alias)]
fn invalid(input: pcu_alias::PcuTensor<f32>)
    -> Result<pcu_alias::PcuTensor<f32>, pcu_alias::PcuExecutionError>
{
    let first = pcu::relu(input)?;
    Ok(pcu::add(first, &input)?)
}

fn main() {}
