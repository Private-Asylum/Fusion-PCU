//! Explicitly native F32 squared differences; reduction uses the selected cuBLAS contract.
//!
//! This private source never grants ordinary scalar IR unchecked arithmetic. The tensor assessor
//! admits only Boundary + `BackendDefined` MSE and owns this exact source/storage ABI together.
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingType,
    PcuOwnedBindingRequirement,
    PcuValueType,
};

pub fn requirements(count: u32) -> Vec<PcuOwnedBindingRequirement> {
    (0..3)
        .map(|slot| PcuOwnedBindingRequirement {
            target: PcuBindingRef::new(0, slot),
            access: if slot == 2 {
                PcuBindingAccess::WriteOnly
            } else {
                PcuBindingAccess::ReadOnly
            },
            binding_type: PcuBindingType::Value(PcuValueType::f32()),
            min_required_bytes: u64::from(count) * 4,
        })
        .collect()
}

pub fn source(count: u32) -> String {
    // CUDA's explicit rounding intrinsics preserve two destination-width operations. No fused
    // multiply-add, checked scalar permission or portable reduction order is inferred here.
    format!(
        r#"extern "C" __global__ void fusion_kernel(const float* prediction, const float* target, float* squared) {{
    const unsigned long long id = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id >= {count}ull) return;
    const float difference = __fsub_rn(prediction[id], target[id]);
    squared[id] = __fmul_rn(difference, difference);
}}
"#
    )
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
