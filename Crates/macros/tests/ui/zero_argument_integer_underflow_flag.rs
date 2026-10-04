use fusion_pcu_macros::pcu;
#[pcu(crate_path = ::pcu_alias, flag(ieee_underflow))]
fn unsupported() -> Result<pcu_alias::PcuTensor<u32>, pcu_alias::PcuExecutionError> {
    pcu::constant(const { [7_u32; 3] })
}
fn main() {}
