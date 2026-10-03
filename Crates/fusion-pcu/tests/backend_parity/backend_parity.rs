//! Cross-provider commit gates execute the same source, including fatal rollback and reuse.
//! Device tests are explicit hardware gates; ignoring them does not establish parity.

#[cfg(all(
    feature = "tensor",
    any(
        feature = "cpu",
        feature = "rocm",
        feature = "cuda",
        feature = "metal",
        feature = "mlx",
        feature = "vulkan"
    )
))]
#[path = "publication/publication.rs"]
mod publication;

macro_rules! unary_role_gate {
    ($feature:literal, $test:ident, $backend:ident) => {
        #[cfg(feature = $feature)]
        #[test]
        #[ignore = "required native ordinary unary actual-role source qualification"]
        fn $test() {
            unary::roles::verify(fusion_pcu::global::PcuBackendChoice::$backend);
        }
    };
}
#[cfg(feature = "cpu")]
#[test]
fn cpu_six_format_unary_actual_role_contract() {
    unary::roles::verify(fusion_pcu::global::PcuBackendChoice::Cpu);
}
unary_role_gate!("rocm", rocm_six_format_unary_actual_role_contract, Rocm);
unary_role_gate!("cuda", cuda_six_format_unary_actual_role_contract, Cuda);
unary_role_gate!("metal", metal_six_format_unary_actual_role_contract, Metal);
unary_role_gate!("mlx", mlx_six_format_unary_actual_role_contract, Mlx);
unary_role_gate!(
    "vulkan",
    vulkan_six_format_unary_actual_role_contract,
    Vulkan
);

macro_rules! portable_unary_gate {
    ($feature:literal, $test:ident, $backend:ident) => {
        #[cfg(feature = $feature)]
        #[test]
        #[ignore = "required native exact Portable unary source/publication conformance"]
        fn $test() {
            unary::portable::verify(fusion_pcu::global::PcuBackendChoice::$backend);
        }
    };
}
#[cfg(feature = "cpu")]
#[test]
fn cpu_six_format_portable_unary_contract() {
    unary::portable::verify(fusion_pcu::global::PcuBackendChoice::Cpu);
}
portable_unary_gate!("rocm", rocm_six_format_portable_unary_contract, Rocm);
portable_unary_gate!("cuda", cuda_six_format_portable_unary_contract, Cuda);
portable_unary_gate!("metal", metal_six_format_portable_unary_contract, Metal);
portable_unary_gate!("mlx", mlx_six_format_portable_unary_contract, Mlx);
portable_unary_gate!("vulkan", vulkan_six_format_portable_unary_contract, Vulkan);

macro_rules! unary_prefix_gate {
    ($feature:literal, $test:ident, $backend:ident) => {
        #[cfg(all(feature = $feature, feature = "tensor"))]
        #[test]
        #[ignore = "required native full-capacity unary prefix source qualification"]
        fn $test() {
            unary::prefix::verify(fusion_pcu::global::PcuBackendChoice::$backend);
        }
    };
}
unary_prefix_gate!("cpu", cpu_six_format_unary_prefix_contract, Cpu);
unary_prefix_gate!("mlx", mlx_six_format_unary_prefix_contract, Mlx);
unary_prefix_gate!("rocm", rocm_six_format_unary_prefix_contract, Rocm);
unary_prefix_gate!("cuda", cuda_six_format_unary_prefix_contract, Cuda);
unary_prefix_gate!("metal", metal_six_format_unary_prefix_contract, Metal);
unary_prefix_gate!("vulkan", vulkan_six_format_unary_prefix_contract, Vulkan);

macro_rules! binary_prefix_gate {
    ($feature:literal, $test:ident, $backend:ident) => {
        #[cfg(all(feature = $feature, feature = "tensor"))]
        #[test]
        #[ignore = "required native full-capacity binary prefix source qualification"]
        fn $test() {
            binary::prefix::verify(fusion_pcu::global::PcuBackendChoice::$backend);
        }
    };
}
binary_prefix_gate!("cpu", cpu_six_format_binary_prefix_contract, Cpu);
binary_prefix_gate!("mlx", mlx_six_format_binary_prefix_contract, Mlx);
binary_prefix_gate!("rocm", rocm_six_format_binary_prefix_contract, Rocm);
binary_prefix_gate!("cuda", cuda_six_format_binary_prefix_contract, Cuda);
binary_prefix_gate!("metal", metal_six_format_binary_prefix_contract, Metal);
binary_prefix_gate!("vulkan", vulkan_six_format_binary_prefix_contract, Vulkan);

macro_rules! integer_prefix_gate {
    ($feature:literal, $test:ident, $backend:ident) => {
        #[cfg(all(feature = $feature, feature = "tensor"))]
        #[test]
        #[ignore = "required native fourteen-width full-capacity integer prefix source qualification"]
        fn $test() {
            integer::prefix::verify(fusion_pcu::global::PcuBackendChoice::$backend);
        }
    };
}
integer_prefix_gate!("cpu", cpu_fourteen_width_integer_prefix_contract, Cpu);
integer_prefix_gate!("mlx", mlx_fourteen_width_integer_prefix_contract, Mlx);
integer_prefix_gate!("rocm", rocm_fourteen_width_integer_prefix_contract, Rocm);
integer_prefix_gate!("cuda", cuda_fourteen_width_integer_prefix_contract, Cuda);
integer_prefix_gate!("metal", metal_fourteen_width_integer_prefix_contract, Metal);
integer_prefix_gate!(
    "vulkan",
    vulkan_fourteen_width_integer_prefix_contract,
    Vulkan
);

macro_rules! div_rem_operand_gate {
    ($feature:literal, $test:ident, $backend:ident) => {
        #[cfg(feature = $feature)]
        #[test]
        #[ignore = "required native opt-in fourteen-width division resource-role source parity"]
        fn $test() {
            integer::div_rem::verify_roles(fusion_pcu::global::PcuBackendChoice::$backend);
        }
    };
}
#[cfg(feature = "cpu")]
#[test]
fn cpu_fourteen_width_div_rem_operand_contract() {
    integer::div_rem::verify_roles(fusion_pcu::global::PcuBackendChoice::Cpu);
}
div_rem_operand_gate!("rocm", rocm_fourteen_width_div_rem_operand_contract, Rocm);
div_rem_operand_gate!("cuda", cuda_fourteen_width_div_rem_operand_contract, Cuda);
div_rem_operand_gate!(
    "metal",
    metal_fourteen_width_div_rem_operand_contract,
    Metal
);
div_rem_operand_gate!("mlx", mlx_fourteen_width_div_rem_operand_contract, Mlx);
div_rem_operand_gate!(
    "vulkan",
    vulkan_fourteen_width_div_rem_operand_contract,
    Vulkan
);

macro_rules! portable_div_rem_gate {
    ($feature:literal, $test:ident, $backend:ident) => {
        #[cfg(feature = $feature)]
        #[test]
        #[ignore = "required native PortableV1 joint division source and resource-role conformance"]
        fn $test() {
            integer::div_rem::verify_portable(fusion_pcu::global::PcuBackendChoice::$backend);
        }
    };
}
#[cfg(feature = "cpu")]
#[test]
fn cpu_fourteen_width_portable_div_rem_contract() {
    integer::div_rem::verify_portable(fusion_pcu::global::PcuBackendChoice::Cpu);
}
portable_div_rem_gate!("rocm", rocm_fourteen_width_portable_div_rem_contract, Rocm);
portable_div_rem_gate!("cuda", cuda_fourteen_width_portable_div_rem_contract, Cuda);
portable_div_rem_gate!(
    "metal",
    metal_fourteen_width_portable_div_rem_contract,
    Metal
);
portable_div_rem_gate!("mlx", mlx_fourteen_width_portable_div_rem_contract, Mlx);
portable_div_rem_gate!(
    "vulkan",
    vulkan_fourteen_width_portable_div_rem_contract,
    Vulkan
);

#[cfg(any(
    feature = "cpu",
    feature = "rocm",
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "mlx"
))]
#[path = "operands/operands.rs"]
mod operands;

macro_rules! operand_gate {
    ($feature:literal, $test:ident, $backend:ident) => {
        #[cfg(feature = $feature)]
        #[test]
        #[ignore = "required native repeated-read and unused-declaration source parity"]
        fn $test() {
            operands::verify(fusion_pcu::global::PcuBackendChoice::$backend);
        }
    };
}
#[cfg(feature = "cpu")]
#[test]
fn cpu_six_format_operand_contract() {
    operands::verify(fusion_pcu::global::PcuBackendChoice::Cpu);
}
operand_gate!("rocm", rocm_six_format_operand_contract, Rocm);
operand_gate!("cuda", cuda_six_format_operand_contract, Cuda);
operand_gate!("metal", metal_six_format_operand_contract, Metal);
operand_gate!("mlx", mlx_six_format_operand_contract, Mlx);
operand_gate!("vulkan", vulkan_six_format_operand_contract, Vulkan);

#[cfg(any(
    feature = "cpu",
    feature = "rocm",
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "mlx"
))]
#[path = "conversion/conversion.rs"]
mod conversion;

macro_rules! conversion_gate {
    ($feature:literal, $test:ident, $backend:ident) => {
        #[cfg(feature = $feature)]
        #[test]
        #[ignore = "required native checked F32/F64 conversion source parity"]
        fn $test() {
            conversion::verify(fusion_pcu::global::PcuBackendChoice::$backend);
        }
    };
}
#[cfg(feature = "cpu")]
#[test]
fn cpu_checked_float_conversion_contract() {
    conversion::verify(fusion_pcu::global::PcuBackendChoice::Cpu);
}
conversion_gate!("rocm", rocm_checked_float_conversion_contract, Rocm);
conversion_gate!("cuda", cuda_checked_float_conversion_contract, Cuda);
conversion_gate!("metal", metal_checked_float_conversion_contract, Metal);
conversion_gate!("mlx", mlx_checked_float_conversion_contract, Mlx);
conversion_gate!("vulkan", vulkan_checked_float_conversion_contract, Vulkan);

#[cfg(all(feature = "cpu", unix))]
#[path = "selection/selection.rs"]
mod selection;

#[cfg(all(feature = "cpu", unix))]
#[test]
fn cpu_explicit_selection_does_not_probe_other_sdks() {
    selection::verify();
}

#[cfg(any(
    feature = "cpu",
    feature = "rocm",
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "mlx"
))]
#[path = "source/source.rs"]
mod source;

#[cfg(any(
    feature = "cpu",
    feature = "rocm",
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "mlx"
))]
#[path = "binary/binary.rs"]
mod binary;

#[cfg(any(
    feature = "cpu",
    feature = "rocm",
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "mlx"
))]
#[path = "integer/integer.rs"]
mod integer;

macro_rules! integer_operand_gate {
    ($feature:literal, $test:ident, $backend:ident) => {
        #[cfg(feature = $feature)]
        #[test]
        #[ignore = "required native opt-in fourteen-width integer operand source parity"]
        fn $test() {
            integer::operands::verify(fusion_pcu::global::PcuBackendChoice::$backend);
        }
    };
}
#[cfg(feature = "cpu")]
#[test]
fn cpu_fourteen_width_integer_operand_contract() {
    integer::operands::verify(fusion_pcu::global::PcuBackendChoice::Cpu);
}
integer_operand_gate!("rocm", rocm_fourteen_width_integer_operand_contract, Rocm);
integer_operand_gate!("cuda", cuda_fourteen_width_integer_operand_contract, Cuda);
integer_operand_gate!(
    "metal",
    metal_fourteen_width_integer_operand_contract,
    Metal
);
integer_operand_gate!("mlx", mlx_fourteen_width_integer_operand_contract, Mlx);
integer_operand_gate!(
    "vulkan",
    vulkan_fourteen_width_integer_operand_contract,
    Vulkan
);

macro_rules! portable_integer_operand_gate {
    ($feature:literal, $test:ident, $backend:ident) => {
        #[cfg(feature = $feature)]
        #[test]
        #[ignore = "required native opt-in Portable fourteen-width integer source parity"]
        fn $test() {
            integer::operands::verify_portable(fusion_pcu::global::PcuBackendChoice::$backend);
        }
    };
}
#[cfg(feature = "cpu")]
#[test]
fn cpu_fourteen_width_portable_integer_contract() {
    integer::operands::verify_portable(fusion_pcu::global::PcuBackendChoice::Cpu);
}
portable_integer_operand_gate!("rocm", rocm_fourteen_width_portable_integer_contract, Rocm);
portable_integer_operand_gate!("cuda", cuda_fourteen_width_portable_integer_contract, Cuda);
portable_integer_operand_gate!(
    "metal",
    metal_fourteen_width_portable_integer_contract,
    Metal
);
portable_integer_operand_gate!("mlx", mlx_fourteen_width_portable_integer_contract, Mlx);
portable_integer_operand_gate!(
    "vulkan",
    vulkan_fourteen_width_portable_integer_contract,
    Vulkan
);

#[cfg(any(
    feature = "cpu",
    feature = "rocm",
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "mlx"
))]
#[path = "unary/unary.rs"]
mod unary;

#[cfg(any(
    feature = "cpu",
    feature = "rocm",
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "mlx"
))]
#[path = "identity/identity.rs"]
mod identity;

#[cfg(any(
    feature = "cpu",
    feature = "rocm",
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "mlx"
))]
#[path = "support/support.rs"]
mod support;

#[cfg(feature = "mlx")]
#[path = "mlx/mlx.rs"]
mod mlx;

#[cfg(feature = "cpu")]
#[test]
fn cpu_eight_width_div_rem_contract() {
    integer::div_rem::verify(fusion_pcu::global::PcuBackendChoice::Cpu);
}

macro_rules! wide_div_rem_gate {
    ($feature:literal, $test:ident, $backend:ident) => {
        #[cfg(feature = $feature)]
        #[test]
        #[ignore = "required native opt-in fourteen-width checked joint division source parity"]
        fn $test() {
            integer::div_rem::verify_fourteen(fusion_pcu::global::PcuBackendChoice::$backend);
        }
    };
}
#[cfg(feature = "cpu")]
#[test]
fn cpu_fourteen_width_div_rem_contract() {
    integer::div_rem::verify_fourteen(fusion_pcu::global::PcuBackendChoice::Cpu);
}
wide_div_rem_gate!("rocm", rocm_fourteen_width_div_rem_contract, Rocm);
wide_div_rem_gate!("cuda", cuda_fourteen_width_div_rem_contract, Cuda);
wide_div_rem_gate!("metal", metal_fourteen_width_div_rem_contract, Metal);
wide_div_rem_gate!("mlx", mlx_fourteen_width_div_rem_contract, Mlx);
wide_div_rem_gate!("vulkan", vulkan_fourteen_width_div_rem_contract, Vulkan);

#[cfg(feature = "rocm")]
#[test]
#[ignore = "requires ROCm; required checked quotient/remainder source parity gate"]
fn rocm_eight_width_div_rem_contract() {
    integer::div_rem::verify(fusion_pcu::global::PcuBackendChoice::Rocm);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA; required checked quotient/remainder source parity gate"]
fn cuda_eight_width_div_rem_contract() {
    integer::div_rem::verify(fusion_pcu::global::PcuBackendChoice::Cuda);
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "requires Metal; required checked quotient/remainder source parity gate"]
fn metal_eight_width_div_rem_contract() {
    integer::div_rem::verify(fusion_pcu::global::PcuBackendChoice::Metal);
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires Vulkan; required checked quotient/remainder source parity gate"]
fn vulkan_eight_width_div_rem_contract() {
    integer::div_rem::verify(fusion_pcu::global::PcuBackendChoice::Vulkan);
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_twenty_two_carrier_broadcast_contract() {
    identity::broadcast::verify(fusion_pcu::global::PcuBackendChoice::Cpu);
}

#[cfg(feature = "rocm")]
#[test]
#[ignore = "requires ROCm; required twenty-two-carrier scalar-broadcast parity gate"]
fn rocm_twenty_two_carrier_broadcast_contract() {
    identity::broadcast::verify(fusion_pcu::global::PcuBackendChoice::Rocm);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA; required twenty-two-carrier scalar-broadcast parity gate"]
fn cuda_twenty_two_carrier_broadcast_contract() {
    identity::broadcast::verify(fusion_pcu::global::PcuBackendChoice::Cuda);
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "requires Metal; required twenty-two-carrier scalar-broadcast parity gate"]
fn metal_twenty_two_carrier_broadcast_contract() {
    identity::broadcast::verify(fusion_pcu::global::PcuBackendChoice::Metal);
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires Vulkan; required twenty-two-carrier scalar-broadcast parity gate"]
fn vulkan_twenty_two_carrier_broadcast_contract() {
    identity::broadcast::verify(fusion_pcu::global::PcuBackendChoice::Vulkan);
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_fourteen_width_clamped_integer_contract() {
    integer::clamped::verify(fusion_pcu::global::PcuBackendChoice::Cpu);
}

#[cfg(feature = "rocm")]
#[test]
#[ignore = "requires ROCm and saturated-result publication; required Clamp parity gate"]
fn rocm_fourteen_width_clamped_integer_contract() {
    integer::clamped::verify(fusion_pcu::global::PcuBackendChoice::Rocm);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA and saturated-result publication; required Clamp parity gate"]
fn cuda_fourteen_width_clamped_integer_contract() {
    integer::clamped::verify(fusion_pcu::global::PcuBackendChoice::Cuda);
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "requires Metal and saturated-result publication; required Clamp parity gate"]
fn metal_fourteen_width_clamped_integer_contract() {
    integer::clamped::verify(fusion_pcu::global::PcuBackendChoice::Metal);
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires Vulkan and saturated-result publication; required Clamp parity gate"]
fn vulkan_fourteen_width_clamped_integer_contract() {
    integer::clamped::verify(fusion_pcu::global::PcuBackendChoice::Vulkan);
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_six_format_binary_contract() {
    binary::verify(fusion_pcu::global::PcuBackendChoice::Cpu);
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_fourteen_width_integer_contract() {
    integer::verify(fusion_pcu::global::PcuBackendChoice::Cpu);
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_six_format_unary_contract() {
    unary::verify(fusion_pcu::global::PcuBackendChoice::Cpu);
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_twenty_two_carrier_identity_contract() {
    identity::verify(fusion_pcu::global::PcuBackendChoice::Cpu);
}

#[cfg(feature = "rocm")]
#[test]
#[ignore = "requires an authorized ROCm device; required full transport parity gate"]
fn rocm_twenty_two_carrier_identity_contract() {
    identity::verify(fusion_pcu::global::PcuBackendChoice::Rocm);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires an authorized CUDA device; required full transport parity gate"]
fn cuda_twenty_two_carrier_identity_contract() {
    identity::verify(fusion_pcu::global::PcuBackendChoice::Cuda);
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "requires an authorized macOS Metal device; required full transport parity gate"]
fn metal_twenty_two_carrier_identity_contract() {
    identity::verify(fusion_pcu::global::PcuBackendChoice::Metal);
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires an authorized Vulkan device; required full transport parity gate"]
fn vulkan_twenty_two_carrier_identity_contract() {
    identity::verify(fusion_pcu::global::PcuBackendChoice::Vulkan);
}

#[cfg(feature = "rocm")]
#[test]
#[ignore = "requires an authorized ROCm device; required unary range-policy parity gate"]
fn rocm_six_format_unary_contract() {
    unary::verify(fusion_pcu::global::PcuBackendChoice::Rocm);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires an authorized CUDA device; required unary range-policy parity gate"]
fn cuda_six_format_unary_contract() {
    unary::verify(fusion_pcu::global::PcuBackendChoice::Cuda);
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "requires an authorized macOS Metal device; required unary range-policy parity gate"]
fn metal_six_format_unary_contract() {
    unary::verify(fusion_pcu::global::PcuBackendChoice::Metal);
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires an authorized Vulkan device; required unary range-policy parity gate"]
fn vulkan_six_format_unary_contract() {
    unary::verify(fusion_pcu::global::PcuBackendChoice::Vulkan);
}

#[cfg(feature = "rocm")]
#[test]
#[ignore = "requires an authorized ROCm device; required fourteen-width parity gate"]
fn rocm_fourteen_width_integer_contract() {
    integer::verify(fusion_pcu::global::PcuBackendChoice::Rocm);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires an authorized CUDA device; required fourteen-width parity gate"]
fn cuda_fourteen_width_integer_contract() {
    integer::verify(fusion_pcu::global::PcuBackendChoice::Cuda);
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "requires an authorized macOS Metal device; required fourteen-width parity gate"]
fn metal_fourteen_width_integer_contract() {
    integer::verify(fusion_pcu::global::PcuBackendChoice::Metal);
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires an authorized Vulkan device; required fourteen-width parity gate"]
fn vulkan_fourteen_width_integer_contract() {
    integer::verify(fusion_pcu::global::PcuBackendChoice::Vulkan);
}

#[cfg(feature = "rocm")]
#[test]
#[ignore = "requires an authorized ROCm device; required native parity gate"]
fn rocm_six_format_binary_contract() {
    binary::verify(fusion_pcu::global::PcuBackendChoice::Rocm);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires an authorized CUDA device; required native parity gate"]
fn cuda_six_format_binary_contract() {
    binary::verify(fusion_pcu::global::PcuBackendChoice::Cuda);
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "requires an authorized macOS Metal device; required native parity gate"]
fn metal_six_format_binary_contract() {
    binary::verify(fusion_pcu::global::PcuBackendChoice::Metal);
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires an authorized Vulkan device; missing formats must fail the parity gate"]
fn vulkan_six_format_binary_contract() {
    binary::verify(fusion_pcu::global::PcuBackendChoice::Vulkan);
}
