use fusion_pcu_cpu::{PcuF32Reference, PcuF64Reference};
use fusion_pcu_macros::pcu;
#[rustfmt::skip]
use pcu_alias::{
    PcuBindingRef,
    PcuHostScalarBinding,
    PcuHostScalarSlice,
    PcuInvocationParameters,
    PcuInvocationShape,
    PcuSynchronousHostDispatchBackend,
};
use core::num::NonZeroU32;

extern crate pcu_alias;

mod scalar {
    use fusion_pcu_macros::pcu;

    #[allow(dead_code)]
    #[pcu(crate_path = ::pcu_alias)]
    pub fn scale(value: f32) -> f32 {
        value * 2.0
    }

    #[allow(dead_code)]
    #[pcu(crate_path = ::pcu_alias)]
    pub fn multiply(value: f32, factor: f32) -> f32 {
        value * factor
    }

    pub mod nested {
        use fusion_pcu_macros::pcu;

        #[allow(dead_code)]
        #[pcu(crate_path = ::pcu_alias)]
        pub fn offset(value: f32) -> f32 {
            super::scale(value) + 1.0
        }
    }
}

mod arithmetic64 {
    use fusion_pcu_macros::pcu;

    #[cfg_attr(all(), inline)]
    #[pcu(crate_path = ::pcu_alias)]
    pub fn scaled(value: f64, factor: f64) -> f64 {
        value * factor
    }

    #[cfg(any())]
    #[pcu(crate_path = ::pcu_alias)]
    fn cfg_disabled(value: f64) -> f64 {
        value
    }

    pub mod nested {
        use fusion_pcu_macros::pcu;

        #[pcu(crate_path = ::pcu_alias)]
        pub fn affine(value: f64, factor: f64, seed: f64) -> f64 {
            super::scaled(value, factor) + seed
        }
    }
}

use arithmetic64::nested::affine as affine_alias;

mod recursive {
    use fusion_pcu_macros::pcu;

    // Deliberate recursion verifies the cold lowering depth limit; never call the native function.
    #[allow(dead_code, unconditional_recursion, clippy::only_used_in_recursion)]
    #[pcu(crate_path = ::pcu_alias)]
    pub fn repeat(value: f32) -> f32 {
        repeat(value)
    }
}

#[pcu(invocations = 4, crate_path = ::pcu_alias)]
fn map(input: &[f32], output: &mut [f32]) {
    let invocation = context.global_invocation_id;
    output[invocation] = scalar::nested::offset(input[invocation]);
}

#[pcu(invocations: N, crate_path = ::pcu_alias)]
fn map_grid_stride<const N: usize>(input: &[f32], output: &mut [f32]) {
    let mut invocation = context.global_invocation_id;
    let stride = context.invocation_count;
    while invocation < N {
        output[invocation] = scalar::nested::offset(input[invocation]);
        invocation += stride;
    }
}

#[pcu(invocations = 1, crate_path = ::pcu_alias)]
fn recursive_map(input: &[f32], output: &mut [f32]) {
    let invocation = context.global_invocation_id;
    output[invocation] = recursive::repeat(input[invocation]);
}

#[pcu(invocations = 4, crate_path = ::pcu_alias)]
fn map_with_scalar<'input, 'factor, 'output>(
    input: &'input [f32],
    factor: &'factor f32,
    output: &'output mut [f32],
) {
    let invocation = context.global_invocation_id;
    output[invocation] = scalar::multiply(input[invocation], *factor);
}

#[pcu(invocations: N, crate_path = ::pcu_alias)]
fn map_grid_with_scalar<'input, 'factor, 'output, const N: usize>(
    input: &'input [f32],
    factor: &'factor f32,
    output: &'output mut [f32],
) {
    let mut invocation = context.global_invocation_id;
    let stride = context.invocation_count;
    while invocation < N {
        output[invocation] = scalar::multiply(input[invocation], factor);
        invocation += stride;
    }
}

#[pcu(invocations = 4, crate_path = ::pcu_alias)]
fn map_f64_with_seed<'input, 'seed, 'output>(
    input: &'input [f64],
    seed: &'seed f64,
    output: &'output mut [f64],
) {
    let invocation = context.global_invocation_id;
    output[invocation] = input[invocation] + seed;
}

#[pcu(invocations = 4, crate_path = ::pcu_alias)]
fn map_f64_helper<'input, 'factor, 'seed, 'output>(
    input: &'input [f64],
    factor: &'factor f64,
    seed: &'seed f64,
    output: &'output mut [f64],
) {
    let invocation = context.global_invocation_id;
    output[invocation] = affine_alias(input[invocation], *factor, *seed);
}

#[pcu(invocations = 4, crate_path = ::pcu_alias)]
fn matrix_f64_helper(input: &[[f64; 2]; 2], factor: &f64, seed: &f64, output: &mut [[f64; 2]; 2]) {
    let invocation = context.global_invocation_id;
    output[invocation / 2][invocation % 2] =
        arithmetic64::nested::affine(input[invocation / 2][invocation % 2], *factor, *seed);
}

#[test]
fn per_function_helpers_compose_through_rust_resolved_companions() {
    let descriptors = map_bindings();
    let builder = map_ir(&descriptors).expect("per-function helper lowers");
    let kernel = builder.ir();
    let submission = pcu_alias::PcuDispatchSubmission {
        kernel: &kernel,
        shape: PcuInvocationShape::invocations(NonZeroU32::new(4).expect("nonzero")),
    };
    let input = [1.0_f32, 2.0, 3.0, 4.0];
    let mut output = [0.0_f32; 4];
    let mut bindings = [
        PcuHostScalarBinding {
            target: PcuBindingRef::new(0, 0),
            slice: PcuHostScalarSlice::Read(&input),
        },
        PcuHostScalarBinding {
            target: PcuBindingRef::new(0, 1),
            slice: PcuHostScalarSlice::ReadWrite(&mut output),
        },
    ];
    PcuF32Reference
        .run_host(submission, &mut bindings, PcuInvocationParameters::empty())
        .expect("CPU reference executes lowered companion IR");
    assert_eq!(
        output.map(f32::to_bits),
        [3.0_f32, 5.0, 7.0, 9.0].map(f32::to_bits)
    );
}

#[test]
fn per_function_helpers_lower_into_owned_grid_stride_body() {
    let descriptors = map_grid_stride_bindings();
    let builder = map_grid_stride_ir::<4>(&descriptors).expect("helper grid-stride lowers");
    builder.with_ir(|ir| {
        assert!(ir.ops.iter().any(|operation| matches!(
            operation,
            pcu_alias::PcuDispatchOp::GridStrideLoop { body, .. }
                if body.iter().any(|nested| matches!(
                    nested,
                    pcu_alias::PcuDispatchOp::Data(pcu_alias::PcuDispatchDataOp::Alu {
                        op: pcu_alias::PcuDispatchAluOp::Mul,
                        ..
                    })
                ))
        )));
    });
}

#[test]
fn recursive_companions_fail_at_the_bounded_depth() {
    let bindings = recursive_map_bindings();
    assert!(recursive_map_ir(&bindings).is_err());
}

#[test]
fn readonly_scalar_references_broadcast_element_zero_for_f32_and_f64() {
    let f32_input = [1.0_f32, 2.0, 3.0, 4.0];
    let factor = 2.5_f32;
    let mut f32_output = [0.0_f32; 4];
    let f32_descriptors = map_with_scalar_bindings();
    let f32_builder = map_with_scalar_ir(&f32_descriptors).expect("f32 scalar lowers");
    let f32_ir = f32_builder.ir();
    let f32_submission = pcu_alias::PcuDispatchSubmission {
        kernel: &f32_ir,
        shape: PcuInvocationShape::invocations(NonZeroU32::new(4).expect("nonzero")),
    };
    let mut f32_bindings = [
        PcuHostScalarBinding {
            target: PcuBindingRef::new(0, 0),
            slice: PcuHostScalarSlice::Read(&f32_input),
        },
        PcuHostScalarBinding {
            target: PcuBindingRef::new(0, 1),
            slice: PcuHostScalarSlice::Read(core::slice::from_ref(&factor)),
        },
        PcuHostScalarBinding {
            target: PcuBindingRef::new(0, 2),
            slice: PcuHostScalarSlice::ReadWrite(&mut f32_output),
        },
    ];
    PcuF32Reference
        .run_host(
            f32_submission,
            &mut f32_bindings,
            PcuInvocationParameters::empty(),
        )
        .expect("f32 scalar executes");
    assert_eq!(
        f32_output.map(f32::to_bits),
        [2.5_f32, 5.0, 7.5, 10.0].map(f32::to_bits)
    );

    let f64_input = [1.0_f64, 2.0, 3.0, 4.0];
    let seed = 0.25_f64;
    let mut f64_output = [0.0_f64; 4];
    let f64_descriptors = map_f64_with_seed_bindings();
    let f64_builder = map_f64_with_seed_ir(&f64_descriptors).expect("f64 scalar lowers");
    let f64_ir = f64_builder.ir();
    let f64_submission = pcu_alias::PcuDispatchSubmission {
        kernel: &f64_ir,
        shape: PcuInvocationShape::invocations(NonZeroU32::new(4).expect("nonzero")),
    };
    let mut f64_bindings = [
        PcuHostScalarBinding {
            target: PcuBindingRef::new(0, 0),
            slice: PcuHostScalarSlice::Read(&f64_input),
        },
        PcuHostScalarBinding {
            target: PcuBindingRef::new(0, 1),
            slice: PcuHostScalarSlice::Read(core::slice::from_ref(&seed)),
        },
        PcuHostScalarBinding {
            target: PcuBindingRef::new(0, 2),
            slice: PcuHostScalarSlice::ReadWrite(&mut f64_output),
        },
    ];
    PcuF64Reference
        .run_host(
            f64_submission,
            &mut f64_bindings,
            PcuInvocationParameters::empty(),
        )
        .expect("f64 scalar executes");
    assert_eq!(
        f64_output.map(f64::to_bits),
        [1.25_f64, 2.25, 3.25, 4.25].map(f64::to_bits)
    );
}

#[test]
#[allow(clippy::suboptimal_flops)] // The lowered IR preserves separate multiply/add rounding.
fn f64_helpers_compose_through_qualified_paths_and_import_aliases() {
    let input = [1.0e40_f64, 2.0e40, 3.0e40, 4.0e40];
    let factor = 1.5_f64;
    let seed = 0.125_f64;
    let mut output = [0.0_f64; 4];
    let descriptors = map_f64_helper_bindings();
    let builder = map_f64_helper_ir(&descriptors).expect("f64 helper lowers");
    let ir = builder.ir();
    let submission = pcu_alias::PcuDispatchSubmission {
        kernel: &ir,
        shape: PcuInvocationShape::invocations(NonZeroU32::new(4).expect("nonzero")),
    };
    let mut bindings = [
        PcuHostScalarBinding {
            target: PcuBindingRef::new(0, 0),
            slice: PcuHostScalarSlice::Read(&input),
        },
        PcuHostScalarBinding {
            target: PcuBindingRef::new(0, 1),
            slice: PcuHostScalarSlice::Read(core::slice::from_ref(&factor)),
        },
        PcuHostScalarBinding {
            target: PcuBindingRef::new(0, 2),
            slice: PcuHostScalarSlice::Read(core::slice::from_ref(&seed)),
        },
        PcuHostScalarBinding {
            target: PcuBindingRef::new(0, 3),
            slice: PcuHostScalarSlice::ReadWrite(&mut output),
        },
    ];
    PcuF64Reference
        .run_host(submission, &mut bindings, PcuInvocationParameters::empty())
        .expect("f64 companion IR executes");
    assert_eq!(
        output.map(f64::to_bits),
        input.map(|value| (value * factor + seed).to_bits())
    );
}

#[test]
fn f64_helper_argument_lowering_accepts_canonical_rank_two_loads() {
    let descriptors = matrix_f64_helper_bindings();
    let builder = matrix_f64_helper_ir(&descriptors).expect("rank-two helper lowers");
    builder.with_ir(|ir| {
        assert!(ir.ops.iter().any(|operation| matches!(
            operation,
            pcu_alias::PcuDispatchOp::Data(pcu_alias::PcuDispatchDataOp::Alu {
                value_type,
                op: pcu_alias::PcuDispatchAluOp::Mul,
                ..
            }) if *value_type == pcu_alias::PcuValueType::f64()
        )));
        assert!(ir.ops.iter().any(|operation| matches!(
            operation,
            pcu_alias::PcuDispatchOp::Data(pcu_alias::PcuDispatchDataOp::BindingLoad {
                binding: PcuBindingRef { binding: 0, .. },
                index: pcu_alias::PcuDispatchIndex::InvocationId,
                ..
            })
        )));
    });
}

#[test]
fn scalar_reference_dereference_and_helper_calls_work_in_grid_stride_ir() {
    let descriptors = map_grid_with_scalar_bindings();
    let builder = map_grid_with_scalar_ir::<4>(&descriptors).expect("scalar helper grid lowers");
    builder.with_ir(|ir| {
        assert!(ir.ops.iter().any(|operation| matches!(
            operation,
            pcu_alias::PcuDispatchOp::GridStrideLoop { body, .. }
                if body.iter().any(|nested| matches!(
                    nested,
                    pcu_alias::PcuDispatchOp::Data(pcu_alias::PcuDispatchDataOp::BindingLoad {
                        index: pcu_alias::PcuDispatchIndex::BindingElementZero,
                        ..
                    })
                ))
        )));
    });
}
