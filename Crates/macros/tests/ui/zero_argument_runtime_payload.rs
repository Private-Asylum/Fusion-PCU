use fusion_pcu_macros::pcu;
fn runtime_payload() -> [u32; 3] { [7; 3] }
#[pcu(crate_path = ::pcu_alias)]
fn unsupported() -> Result<pcu_alias::PcuTensor<u32>, pcu_alias::PcuExecutionError> {
    pcu::constant(runtime_payload())
}
fn main() {}
