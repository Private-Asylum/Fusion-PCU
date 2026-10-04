use fusion_pcu_macros::pcu;
fn runtime_helper() -> Result<pcu_alias::PcuTensor<u32>, pcu_alias::PcuExecutionError> {
    Err(pcu_alias::PcuExecutionError::InvalidTensorSourcePlan)
}
#[pcu(crate_path = ::pcu_alias)]
fn unsupported() -> Result<pcu_alias::PcuTensor<u32>, pcu_alias::PcuExecutionError> {
    runtime_helper()
}
fn main() {}
