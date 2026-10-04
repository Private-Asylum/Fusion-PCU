use fusion_pcu_macros::pcu;
#[pcu(crate_path = ::pcu_alias)]
fn unsupported(input: &[u32]) -> Result<pcu_alias::PcuTensor<u32>, pcu_alias::PcuExecutionError> {
    pcu::constant(const { [input[0]; 3] })
}
fn main() {}
