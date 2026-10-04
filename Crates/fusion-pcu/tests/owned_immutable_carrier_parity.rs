//! One genuine source/ownership fixture across every active carrier provider.
#![cfg(all(
    feature = "tensor",
    any(
        feature = "rocm",
        feature = "cuda",
        feature = "vulkan",
        feature = "metal",
        feature = "mlx"
    )
))]
#[path = "owned_immutable_literals/carriers/carriers.rs"]
mod carriers;

macro_rules! provider {
    ($feature:literal, $name:ident, $backend:ident, $reason:literal) => {
        #[cfg(feature = $feature)]
        #[test]
        #[ignore = $reason]
        fn $name() {
            carriers::run(fusion_pcu::global::PcuBackendChoice::$backend);
        }
    };
}
provider!(
    "rocm",
    rocm_immutable_carrier_source_contract,
    Rocm,
    "Requires a native ROCm device"
);
provider!(
    "cuda",
    cuda_immutable_carrier_source_contract,
    Cuda,
    "Requires a native CUDA device"
);
provider!(
    "vulkan",
    vulkan_immutable_carrier_source_contract,
    Vulkan,
    "Requires a native Vulkan device"
);
provider!(
    "metal",
    metal_immutable_carrier_source_contract,
    Metal,
    "Requires native Apple Metal"
);
provider!(
    "mlx",
    mlx_immutable_carrier_source_contract,
    Mlx,
    "Requires a native Apple MLX runtime"
);
