//! Every provider executes the same bounded training and escaped-owner source contract.
//! Ignored hardware gates remain incomplete until exercised on the actual provider.
#[cfg(any(
    feature = "cpu",
    feature = "rocm",
    feature = "cuda",
    feature = "metal",
    feature = "mlx",
    feature = "vulkan"
))]
#[path = "compound/compound.rs"]
mod compound;

macro_rules! compound_gate {
    ($feature:literal, $test:ident, $backend:ident) => {
        #[cfg(feature = $feature)]
        #[test]
        #[ignore = "required native F32/F64 strict compound permission parity"]
        fn $test() {
            compound::verify(fusion_pcu::global::PcuBackendChoice::$backend);
        }
    };
}
#[cfg(feature = "cpu")]
#[test]
fn cpu_strict_compound_permission_contract() {
    compound::verify(fusion_pcu::global::PcuBackendChoice::Cpu);
}
compound_gate!("rocm", rocm_strict_compound_permission_contract, Rocm);
compound_gate!("cuda", cuda_strict_compound_permission_contract, Cuda);
compound_gate!("metal", metal_strict_compound_permission_contract, Metal);
compound_gate!("mlx", mlx_strict_compound_permission_contract, Mlx);
compound_gate!("vulkan", vulkan_strict_compound_permission_contract, Vulkan);
#[cfg(any(
    feature = "cpu",
    feature = "rocm",
    feature = "cuda",
    feature = "metal",
    feature = "mlx",
    feature = "vulkan"
))]
#[path = "backward/backward.rs"]
mod backward;

macro_rules! backward_gate {
    ($feature:literal, $test:ident, $backend:ident) => {
        #[cfg(feature = $feature)]
        #[test]
        #[ignore = "required native six-format owned ReLU backward parity"]
        fn $test() {
            backward::verify(fusion_pcu::global::PcuBackendChoice::$backend);
        }
    };
}
#[cfg(feature = "cpu")]
#[test]
fn cpu_six_format_relu_backward_tensor_contract() {
    backward::verify(fusion_pcu::global::PcuBackendChoice::Cpu);
}
backward_gate!("rocm", rocm_six_format_relu_backward_tensor_contract, Rocm);
backward_gate!("cuda", cuda_six_format_relu_backward_tensor_contract, Cuda);
backward_gate!(
    "metal",
    metal_six_format_relu_backward_tensor_contract,
    Metal
);
backward_gate!("mlx", mlx_six_format_relu_backward_tensor_contract, Mlx);
backward_gate!(
    "vulkan",
    vulkan_six_format_relu_backward_tensor_contract,
    Vulkan
);

#[cfg(any(
    feature = "cpu",
    feature = "rocm",
    feature = "cuda",
    feature = "metal",
    feature = "mlx",
    feature = "vulkan"
))]
#[path = "source/source.rs"]
mod source;

#[cfg(any(
    feature = "cpu",
    feature = "rocm",
    feature = "cuda",
    feature = "metal",
    feature = "mlx",
    feature = "vulkan"
))]
#[path = "contract/contract.rs"]
mod contract;

#[cfg(any(
    feature = "cpu",
    feature = "rocm",
    feature = "cuda",
    feature = "metal",
    feature = "mlx",
    feature = "vulkan"
))]
#[path = "integer/integer.rs"]
mod integer;

#[cfg(any(
    feature = "cpu",
    feature = "rocm",
    feature = "cuda",
    feature = "metal",
    feature = "mlx",
    feature = "vulkan"
))]
#[path = "low_float/low_float.rs"]
mod low_float;

macro_rules! low_float_gate {
    ($feature:literal, $test:ident, $backend:ident) => {
        #[cfg(feature = $feature)]
        #[test]
        #[ignore = "required native four-low-format owned tensor parity"]
        fn $test() {
            low_float::verify(fusion_pcu::global::PcuBackendChoice::$backend);
        }
    };
}

#[cfg(any(
    feature = "cpu",
    feature = "rocm",
    feature = "cuda",
    feature = "metal",
    feature = "mlx",
    feature = "vulkan"
))]
#[path = "carrier/carrier.rs"]
mod carrier;

macro_rules! carrier_gate {
    ($feature:literal, $test:ident, $backend:ident) => {
        #[cfg(feature = $feature)]
        #[test]
        #[ignore = "required native twenty-two-carrier owned tensor parity"]
        fn $test() {
            carrier::verify(fusion_pcu::global::PcuBackendChoice::$backend);
        }
    };
}
#[cfg(feature = "cpu")]
#[test]
fn cpu_twenty_two_carrier_owned_contract() {
    carrier::verify(fusion_pcu::global::PcuBackendChoice::Cpu);
}
carrier_gate!("rocm", rocm_twenty_two_carrier_owned_contract, Rocm);
carrier_gate!("cuda", cuda_twenty_two_carrier_owned_contract, Cuda);
carrier_gate!("metal", metal_twenty_two_carrier_owned_contract, Metal);
carrier_gate!("mlx", mlx_twenty_two_carrier_owned_contract, Mlx);
carrier_gate!("vulkan", vulkan_twenty_two_carrier_owned_contract, Vulkan);
#[cfg(feature = "cpu")]
#[test]
fn cpu_four_low_format_tensor_contract() {
    low_float::verify(fusion_pcu::global::PcuBackendChoice::Cpu);
}
low_float_gate!("rocm", rocm_four_low_format_tensor_contract, Rocm);
low_float_gate!("cuda", cuda_four_low_format_tensor_contract, Cuda);
low_float_gate!("metal", metal_four_low_format_tensor_contract, Metal);
low_float_gate!("mlx", mlx_four_low_format_tensor_contract, Mlx);
low_float_gate!("vulkan", vulkan_four_low_format_tensor_contract, Vulkan);

#[cfg(feature = "cpu")]
#[test]
fn cpu_fourteen_width_integer_tensor_contract() {
    integer::verify(fusion_pcu::global::PcuBackendChoice::Cpu);
}

#[cfg(feature = "rocm")]
#[test]
#[ignore = "requires ROCm; required fourteen-width tensor source parity"]
fn rocm_fourteen_width_integer_tensor_contract() {
    integer::verify(fusion_pcu::global::PcuBackendChoice::Rocm);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA; required fourteen-width tensor source parity"]
fn cuda_fourteen_width_integer_tensor_contract() {
    integer::verify(fusion_pcu::global::PcuBackendChoice::Cuda);
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "requires Metal; required fourteen-width tensor source parity"]
fn metal_fourteen_width_integer_tensor_contract() {
    integer::verify(fusion_pcu::global::PcuBackendChoice::Metal);
}

#[cfg(feature = "mlx")]
#[test]
#[ignore = "requires MLX; required fourteen-width tensor source parity"]
fn mlx_fourteen_width_integer_tensor_contract() {
    integer::verify(fusion_pcu::global::PcuBackendChoice::Mlx);
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires Vulkan; required fourteen-width tensor source parity"]
fn vulkan_fourteen_width_integer_tensor_contract() {
    integer::verify(fusion_pcu::global::PcuBackendChoice::Vulkan);
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_strict_training_and_owned_composition() {
    contract::verify(fusion_pcu::global::PcuBackendChoice::Cpu);
}

#[cfg(feature = "rocm")]
#[test]
#[ignore = "requires an authorized ROCm device; required tensor source parity gate"]
fn rocm_strict_training_and_owned_composition() {
    contract::verify(fusion_pcu::global::PcuBackendChoice::Rocm);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires an authorized CUDA device; required tensor source parity gate"]
fn cuda_strict_training_and_owned_composition() {
    contract::verify(fusion_pcu::global::PcuBackendChoice::Cuda);
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "requires an authorized Metal device; required tensor source parity gate"]
fn metal_strict_training_and_owned_composition() {
    contract::verify(fusion_pcu::global::PcuBackendChoice::Metal);
}

#[cfg(feature = "mlx")]
#[test]
#[ignore = "requires an authorized MLX device; required tensor source parity gate"]
fn mlx_strict_training_and_owned_composition() {
    contract::verify(fusion_pcu::global::PcuBackendChoice::Mlx);
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires an authorized Vulkan device; required tensor source parity gate"]
fn vulkan_strict_training_and_owned_composition() {
    contract::verify(fusion_pcu::global::PcuBackendChoice::Vulkan);
}

#[cfg(any(
    feature = "cpu",
    feature = "rocm",
    feature = "cuda",
    feature = "vulkan"
))]
#[path = "gradient/gradient.rs"]
mod gradient;

#[cfg(feature = "cpu")]
#[test]
fn cpu_requested_gradient_contract() {
    gradient::verify(fusion_pcu::global::PcuBackendChoice::Cpu);
}

#[cfg(feature = "rocm")]
#[test]
#[ignore = "required native target-selective gradient and checked forward effects"]
fn rocm_requested_gradient_contract() {
    gradient::verify(fusion_pcu::global::PcuBackendChoice::Rocm);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "required native target-selective gradient and checked forward effects"]
fn cuda_requested_gradient_contract() {
    gradient::verify(fusion_pcu::global::PcuBackendChoice::Cuda);
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "required native target-selective gradient and checked forward effects"]
fn vulkan_requested_gradient_contract() {
    gradient::verify(fusion_pcu::global::PcuBackendChoice::Vulkan);
}

macro_rules! carrier_prefix_gate {
    ($feature:literal, $test:ident, $backend:ident) => {
        #[cfg(feature = $feature)]
        #[test]
        #[ignore = "required native full-capacity carrier prefix residency qualification"]
        fn $test() {
            carrier::prefix::verify(fusion_pcu::global::PcuBackendChoice::$backend);
        }
    };
}
#[cfg(feature = "cpu")]
#[test]
fn cpu_twenty_two_carrier_prefix_contract() {
    carrier::prefix::verify(fusion_pcu::global::PcuBackendChoice::Cpu);
}
carrier_prefix_gate!("rocm", rocm_twenty_two_carrier_prefix_contract, Rocm);
carrier_prefix_gate!("cuda", cuda_twenty_two_carrier_prefix_contract, Cuda);
carrier_prefix_gate!("metal", metal_twenty_two_carrier_prefix_contract, Metal);
carrier_prefix_gate!("mlx", mlx_twenty_two_carrier_prefix_contract, Mlx);
carrier_prefix_gate!("vulkan", vulkan_twenty_two_carrier_prefix_contract, Vulkan);

#[cfg(any(feature = "cpu", feature = "metal", feature = "mlx"))]
#[path = "selected_numerical/selected_numerical.rs"]
mod selected_numerical;
#[cfg(feature = "cpu")]
#[test]
fn cpu_selected_numerical_source_contract() {
    selected_numerical::verify(fusion_pcu::global::PcuBackendChoice::Cpu);
}
#[cfg(feature = "metal")]
#[test]
#[ignore = "required actual M4 ordinary selected numerical facade lift"]
fn metal_selected_numerical_source_contract() {
    selected_numerical::verify(fusion_pcu::global::PcuBackendChoice::Metal);
}
#[cfg(feature = "mlx")]
#[test]
#[ignore = "required actual M4 ordinary selected numerical facade lift"]
fn mlx_selected_numerical_source_contract() {
    selected_numerical::verify(fusion_pcu::global::PcuBackendChoice::Mlx);
}

#[cfg(any(feature = "cpu", feature = "metal", feature = "mlx"))]
#[path = "selected_graph/selected_graph.rs"]
mod selected_graph;
#[cfg(feature = "cpu")]
#[test]
fn cpu_selected_graph_source_contract() {
    selected_graph::verify(fusion_pcu::global::PcuBackendChoice::Cpu);
}
#[cfg(feature = "metal")]
#[test]
#[ignore = "required actual M4 ordinary multi-stage graph facade lift"]
fn metal_selected_graph_source_contract() {
    selected_graph::verify(fusion_pcu::global::PcuBackendChoice::Metal);
}
#[cfg(feature = "mlx")]
#[test]
#[ignore = "required actual M4 ordinary multi-stage graph facade lift"]
fn mlx_selected_graph_source_contract() {
    selected_graph::verify(fusion_pcu::global::PcuBackendChoice::Mlx);
}

#[cfg(any(feature = "cpu", feature = "metal", feature = "mlx"))]
#[path = "scalable_graph/scalable_graph.rs"]
mod scalable_graph;
#[cfg(feature = "cpu")]
#[test]
fn cpu_scalable_graph_source_contract() {
    scalable_graph::verify(fusion_pcu::global::PcuBackendChoice::Cpu);
}
#[cfg(feature = "metal")]
#[test]
#[ignore = "required actual M4 compact MSE and 1,024-element training facade lift"]
fn metal_scalable_graph_source_contract() {
    scalable_graph::verify(fusion_pcu::global::PcuBackendChoice::Metal);
}
#[cfg(feature = "mlx")]
#[test]
#[ignore = "required actual M4 compact MSE and 1,024-element training facade lift"]
fn mlx_scalable_graph_source_contract() {
    scalable_graph::verify(fusion_pcu::global::PcuBackendChoice::Mlx);
}

#[cfg(any(feature = "cpu", feature = "metal", feature = "mlx"))]
#[path = "low_graph_source/low_graph_source.rs"]
mod low_graph_source;
#[cfg(feature = "cpu")]
#[test]
fn cpu_low_graph_source_contract() {
    low_graph_source::verify(fusion_pcu::global::PcuBackendChoice::Cpu);
}
#[cfg(feature = "metal")]
#[test]
#[ignore = "required actual M4 ordinary six-format pointwise backward graph"]
fn metal_low_graph_source_contract() {
    low_graph_source::verify(fusion_pcu::global::PcuBackendChoice::Metal);
}
#[cfg(feature = "mlx")]
#[test]
#[ignore = "required actual M4 ordinary six-format pointwise backward graph"]
fn mlx_low_graph_source_contract() {
    low_graph_source::verify(fusion_pcu::global::PcuBackendChoice::Mlx);
}
