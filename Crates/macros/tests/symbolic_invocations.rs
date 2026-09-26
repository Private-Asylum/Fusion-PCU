use fusion_pcu_macros::{
    pcu,
    pcu_dispatch,
};
use core::num::NonZeroU32;
use fusion_pcu_cpu::{
    PcuF32Reference,
    PcuF32ReferenceError,
};
use pcu_alias::PcuBindingAccess;
use pcu_alias::{
    PcuBindingRef,
    PcuDispatchDataOp,
    PcuDispatchAluOp,
    PcuDispatchOp,
    PcuDispatchSubmission,
    PcuHostDispatchError,
    PcuHostScalarBinding,
    PcuHostScalarSlice,
    PcuInvocationParameters,
    PcuInvocationShape,
    PcuValueType,
    PcuSynchronousHostDispatchBackend,
};

extern crate pcu_alias as renamed_pcu;

#[pcu(invocations = N * 2, crate_path = ::renamed_pcu)]
fn concise_alias_map<const N: usize>(input: &[f32], output: &mut [f32]) {
    let invocation = context.global_invocation_id;
    output[invocation] = input[invocation] + 1.0;
}

#[pcu_dispatch(invocations = R * C, crate_path = ::renamed_pcu)]
fn matrix_map<const R: usize, const C: usize>(input: &[f32], output: &mut [f32]) {
    let invocation = context.global_invocation_id;
    output[invocation] = input[invocation] * 2.0;
}

#[pcu_dispatch(invocations = (R + 1) * C - C, crate_path = ::renamed_pcu)]
fn expression_map<const R: usize, const C: usize>(input: &[f32], output: &mut [f32]) {
    let invocation = context.global_invocation_id;
    output[invocation] = input[invocation] + 1.0;
}

#[pcu_dispatch(invocations = N, crate_path = ::renamed_pcu)]
fn in_place_map<const N: usize>(output: &mut [f32]) {
    let invocation = context.global_invocation_id;
    output[invocation] = output[invocation] + 1.0;
}

#[pcu_dispatch(invocations = N, crate_path = ::renamed_pcu)]
fn grid_stride_map<const N: usize>(input: &[f32], output: &mut [f32]) {
    let mut id = context.global_invocation_id;
    let stride = context.invocation_count;
    while id < N {
        output[id] = input[id] * 2.0;
        id += stride;
    }
}

#[pcu_dispatch(invocations = 4, crate_path = ::renamed_pcu)]
fn tiled_grid_stride_map<const N: usize>(input: &[f32], output: &mut [f32]) {
    let mut id = context.global_invocation_id;
    let stride = context.invocation_count;
    while id < N {
        output[id] = input[id] * 2.0;
        id += stride;
    }
}

#[pcu_dispatch(invocations = 250, crate_path = ::renamed_pcu)]
fn wide_grid_stride_map<const N: usize>(input: &[f32], output: &mut [f32]) {
    let mut id = context.global_invocation_id;
    let stride = context.invocation_count;
    while id < N {
        output[id] = input[id] * 2.0;
        id += stride;
    }
}

#[pcu_dispatch(invocations = 8, crate_path = ::renamed_pcu)]
fn padded_grid_stride_map<const N: usize>(input: &[f32], output: &mut [f32]) {
    let mut id = context.global_invocation_id;
    let stride = context.invocation_count;
    while id < N {
        output[id] = input[id] * 2.0;
        id += stride;
    }
}

#[pcu(invocations = N, crate_path = ::renamed_pcu)]
fn u32_copy<const N: usize>(input: &[u32], output: &mut [u32]) {
    let invocation = context.global_invocation_id;
    output[invocation] = input[invocation];
}

#[pcu(invocations = N, crate_path = ::renamed_pcu)]
fn u32_checked_div_rem<const N: usize>(
    left: &[u32],
    right: &[u32],
    quotient: &mut [u32],
    remainder: &mut [u32],
) {
    let id = context.global_invocation_id;
    let (q, r) = pcu::checked_div_rem(left[id], right[id]);
    quotient[id] = q;
    remainder[id] = r;
}

#[pcu_dispatch(invocations = 2, crate_path = ::renamed_pcu)]
fn u32_checked_div_rem_grid<const N: usize>(
    left: &[u32],
    right: &[u32],
    quotient: &mut [u32],
    remainder: &mut [u32],
) {
    let mut id = context.global_invocation_id;
    let stride = context.invocation_count;
    while id < N {
        let (q, r) = pcu::checked_div_rem(left[id], right[id]);
        quotient[id] = q;
        remainder[id] = r;
        id += stride;
    }
}

#[test]
fn u32_checked_div_rem_macro_builds_admitted_direct_and_grid_stride_ir() {
    let bindings = u32_checked_div_rem_bindings();
    let direct = u32_checked_div_rem::<8>(&bindings).expect("direct DivRem builds");
    pcu_alias::validate_u32_checked_div_rem_kernel(&direct.ir())
        .expect("direct checked DivRem profile admits");
    assert!(matches!(
        direct.ir().ops[2],
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem { .. })
    ));

    let bindings = u32_checked_div_rem_grid_bindings();
    let grid = u32_checked_div_rem_grid::<8>(&bindings).expect("grid DivRem builds");
    pcu_alias::validate_u32_checked_div_rem_kernel(&grid.ir())
        .expect("grid-stride checked DivRem profile admits");
    assert!(matches!(
        grid.ir().ops[0],
        PcuDispatchOp::GridStrideLoop { extent: 8, .. }
    ));
}

#[pcu_dispatch(invocations = N, crate_path = ::renamed_pcu)]
fn u16_wrapping_map<const N: usize>(left: &[u16], right: &[u16], output: &mut [u16]) {
    let invocation = context.global_invocation_id;
    output[invocation] = left[invocation].wrapping_add(right[invocation]);
}

#[pcu_dispatch(invocations = 2, crate_path = ::renamed_pcu)]
fn u16_grid_stride_map<const N: usize>(left: &[u16], right: &[u16], output: &mut [u16]) {
    let mut id = context.global_invocation_id;
    let stride = context.invocation_count;
    while id < N {
        output[id] = left[id].wrapping_add(right[id]).wrapping_mul(right[id]);
        id += stride;
    }
}

#[pcu_dispatch(invocations = N, crate_path = ::renamed_pcu)]
fn u8_wrapping_map<const N: usize>(left: &[u8], right: &[u8], output: &mut [u8]) {
    let invocation = context.global_invocation_id;
    output[invocation] = left[invocation].wrapping_add(right[invocation]);
}

#[pcu_dispatch(invocations = 2, crate_path = ::renamed_pcu)]
fn u8_grid_stride_map<const N: usize>(left: &[u8], right: &[u8], output: &mut [u8]) {
    let mut id = context.global_invocation_id;
    let stride = context.invocation_count;
    while id < N {
        output[id] = left[id].wrapping_add(right[id]).wrapping_mul(right[id]);
        id += stride;
    }
}

#[pcu_dispatch(invocations = N, crate_path = ::renamed_pcu)]
fn i16_wrapping_map<const N: usize>(left: &[i16], right: &[i16], output: &mut [i16]) {
    let invocation = context.global_invocation_id;
    output[invocation] = left[invocation].wrapping_add(right[invocation]);
}

#[pcu_dispatch(invocations = 2, crate_path = ::renamed_pcu)]
fn i16_grid_stride_map<const N: usize>(left: &[i16], right: &[i16], output: &mut [i16]) {
    let mut id = context.global_invocation_id;
    let stride = context.invocation_count;
    while id < N {
        output[id] = left[id].wrapping_add(right[id]).wrapping_mul(right[id]);
        id += stride;
    }
}

#[pcu_dispatch(invocations = N, crate_path = ::renamed_pcu)]
fn i8_wrapping_map<const N: usize>(left: &[i8], right: &[i8], output: &mut [i8]) {
    let invocation = context.global_invocation_id;
    output[invocation] = left[invocation].wrapping_add(right[invocation]);
}

#[pcu_dispatch(invocations = 2, crate_path = ::renamed_pcu)]
fn i8_grid_stride_map<const N: usize>(left: &[i8], right: &[i8], output: &mut [i8]) {
    let mut id = context.global_invocation_id;
    let stride = context.invocation_count;
    while id < N {
        output[id] = left[id].wrapping_add(right[id]).wrapping_mul(right[id]);
        id += stride;
    }
}

#[test]
fn i8_wrapping_dispatch_macro_builds_admitted_direct_and_grid_stride_maps() {
    let bindings = i8_wrapping_map_bindings();
    assert_eq!(bindings[0].value_type(), Some(PcuValueType::i8()));
    let direct = i8_wrapping_map::<8>(&bindings).expect("i8 direct map builds");
    pcu_alias::validate_i8_map_kernel(&direct.ir()).expect("direct i8 profile admits");

    let loop_bindings = i8_grid_stride_map_bindings();
    let grid = i8_grid_stride_map::<8>(&loop_bindings).expect("i8 grid-stride map builds");
    pcu_alias::validate_i8_map_kernel(&grid.ir()).expect("grid-stride i8 profile admits");
    assert!(matches!(
        grid.ir().ops[0],
        PcuDispatchOp::GridStrideLoop { extent: 8, .. }
    ));
}

#[test]
fn i16_wrapping_dispatch_macro_builds_admitted_direct_and_grid_stride_maps() {
    let bindings = i16_wrapping_map_bindings();
    assert_eq!(bindings[0].value_type(), Some(PcuValueType::i16()));
    let direct = i16_wrapping_map::<8>(&bindings).expect("i16 direct map builds");
    pcu_alias::validate_i16_map_kernel(&direct.ir()).expect("direct i16 profile admits");

    let loop_bindings = i16_grid_stride_map_bindings();
    let grid = i16_grid_stride_map::<8>(&loop_bindings).expect("i16 grid-stride map builds");
    pcu_alias::validate_i16_map_kernel(&grid.ir()).expect("grid-stride i16 profile admits");
    assert!(matches!(
        grid.ir().ops[0],
        PcuDispatchOp::GridStrideLoop { extent: 8, .. }
    ));
}

#[test]
fn u8_wrapping_dispatch_macro_builds_admitted_direct_and_grid_stride_maps() {
    let bindings = u8_wrapping_map_bindings();
    assert_eq!(bindings[0].value_type(), Some(PcuValueType::u8()));
    let direct = u8_wrapping_map::<8>(&bindings).expect("u8 direct map builds");
    pcu_alias::validate_u8_map_kernel(&direct.ir()).expect("direct u8 profile admits");

    let loop_bindings = u8_grid_stride_map_bindings();
    let grid = u8_grid_stride_map::<8>(&loop_bindings).expect("u8 grid-stride map builds");
    pcu_alias::validate_u8_map_kernel(&grid.ir()).expect("grid-stride u8 profile admits");
    assert!(matches!(
        grid.ir().ops[0],
        PcuDispatchOp::GridStrideLoop { extent: 8, .. }
    ));
}

#[test]
fn u16_wrapping_dispatch_macro_builds_admitted_direct_and_grid_stride_maps() {
    let bindings = u16_wrapping_map_bindings();
    assert_eq!(bindings[0].value_type(), Some(PcuValueType::u16()));
    let direct = u16_wrapping_map::<8>(&bindings).expect("u16 direct map builds");
    pcu_alias::validate_u16_map_kernel(&direct.ir()).expect("direct u16 profile admits");

    let loop_bindings = u16_grid_stride_map_bindings();
    let grid = u16_grid_stride_map::<8>(&loop_bindings).expect("u16 grid-stride map builds");
    pcu_alias::validate_u16_map_kernel(&grid.ir()).expect("grid-stride u16 profile admits");
    assert!(matches!(
        grid.ir().ops[0],
        PcuDispatchOp::GridStrideLoop { extent: 8, .. }
    ));
}

#[test]
fn const_generic_invocations_specialize_to_the_declared_shape() {
    let typed_bindings = u32_copy_bindings();
    assert_eq!(typed_bindings[0].value_type(), Some(PcuValueType::u32()));
    assert_eq!(typed_bindings[1].value_type(), Some(PcuValueType::u32()));
    let typed = u32_copy::<8>(&typed_bindings).expect("u32 identity copy builds typed IR");
    assert_eq!(typed.ir().entry.logical_shape, [8, 1, 1]);
    assert!(matches!(
        typed.ir().ops[0],
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad { .. })
    ));
    assert!(matches!(
        typed.ir().ops[1],
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore { .. })
    ));

    let alias_bindings = concise_alias_map_bindings();
    let alias = concise_alias_map::<12>(&alias_bindings).expect("the alias lowers identically");
    assert_eq!(alias.ir().entry.logical_shape, [24, 1, 1]);

    let bindings = matrix_map_bindings();
    assert_eq!(bindings[0].access, PcuBindingAccess::ReadOnly);
    assert_eq!(bindings[1].access, PcuBindingAccess::ReadWrite);
    let kernel = matrix_map::<32, 64>(&bindings).expect("the map IR is valid");
    assert_eq!(kernel.ir().entry.logical_shape, [2048, 1, 1]);

    let second = matrix_map::<4, 5>(&bindings).expect("the second specialization is valid");
    assert_eq!(second.ir().entry.logical_shape, [20, 1, 1]);

    let expression_bindings = expression_map_bindings();
    let expression = expression_map::<7, 9>(&expression_bindings)
        .expect("checked arithmetic expression is valid");
    assert_eq!(expression.ir().entry.logical_shape, [63, 1, 1]);

    let mutable_binding = in_place_map_bindings();
    assert_eq!(mutable_binding[0].access, PcuBindingAccess::ReadWrite);
    let in_place = in_place_map::<16>(&mutable_binding).expect("mutable reads and writes lower");
    assert_eq!(in_place.ir().entry.logical_shape, [16, 1, 1]);

    let grid_stride_bindings = grid_stride_map_bindings();
    let grid_stride = grid_stride_map::<16>(&grid_stride_bindings)
        .expect("equal extent and logical count reduce to one indexed map");
    assert_eq!(grid_stride.ir().entry.logical_shape, [16, 1, 1]);
    assert!(grid_stride.ir().ops.iter().any(|op| matches!(
        op,
        PcuDispatchOp::GridStrideLoop { extent: 16, body }
            if body.iter().any(|body_op| matches!(
                body_op,
                PcuDispatchOp::Data(pcu_alias::PcuDispatchDataOp::BindingStore {
                    index: pcu_alias::PcuDispatchIndex::GridStrideId,
                    ..
                })
            ))
    )));

    let tiled = tiled_grid_stride_map::<16>(&grid_stride_bindings)
        .expect("a four-lane launch covers the sixteen-element extent");
    assert_eq!(tiled.ir().entry.logical_shape, [4, 1, 1]);
    assert!(
        tiled
            .ir()
            .ops
            .iter()
            .any(|op| matches!(op, PcuDispatchOp::GridStrideLoop { extent: 16, .. }))
    );
}

#[test]
fn equal_extent_grid_stride_covers_each_logical_lane_once_and_excludes_padding() {
    let count = NonZeroU32::new(16).expect("nonzero test count");
    for id in 0..count.get() {
        assert!(renamed_pcu::PcuDispatchContext::new(id, count).is_some());
    }
    assert!(renamed_pcu::PcuDispatchContext::new(count.get(), count).is_none());
}

#[test]
fn specialized_ir_is_admitted_by_both_current_compute_lowerers() {
    let bindings = matrix_map_bindings();
    let builder = matrix_map::<32, 64>(&bindings).expect("macro builds specialized IR");
    let kernel = builder.ir();

    let hip = fusion_pcu_rocm::lower_dispatch_to_hip_source(&kernel)
        .expect("HIP lowerer admits the specialized kernel");
    assert!(hip.contains("fusion_gid >= 2048u"));

    let mut spirv = fusion_pcu_spirv::PcuSpirvFixedSink::<2048>::new();
    fusion_pcu_spirv::lower_dispatch_to_spirv(
        &kernel,
        fusion_pcu_spirv::PcuSpirvLoweringOptions::minimal_shader(),
        &mut spirv,
    )
    .expect("SPIR-V lowerer admits the specialized kernel");
    assert!(!spirv.is_empty());
}

struct SynchronousProbe;

// SAFETY: This probe only accesses its borrowed slices inside the call and returns after all
// accesses finish. It does not launch device work or retain pointers.
unsafe impl PcuSynchronousHostDispatchBackend<f32> for SynchronousProbe {
    type Error = ();

    fn run_host_direct(
        &self,
        _submission: PcuDispatchSubmission<'_>,
        bindings: &mut [PcuHostScalarBinding<'_, f32>],
        _parameters: PcuInvocationParameters<'_>,
    ) -> Result<(), Self::Error> {
        let [input, output] = bindings else {
            return Err(());
        };
        let PcuHostScalarSlice::Read(input) = &input.slice else {
            return Err(());
        };
        let PcuHostScalarSlice::ReadWrite(output) = &mut output.slice else {
            return Err(());
        };
        for (source, destination) in input.iter().zip(output.iter_mut()) {
            *destination = *source * 2.0;
        }
        Ok(())
    }
}

#[test]
fn scoped_host_bindings_enforce_reference_permissions_and_extent() {
    let descriptors = matrix_map_bindings();
    let builder = matrix_map::<2, 2>(&descriptors).expect("valid map");
    let kernel = builder.ir();
    let submission = PcuDispatchSubmission {
        kernel: &kernel,
        shape: PcuInvocationShape::invocations(core::num::NonZeroU32::new(4).expect("nonzero")),
    };
    let input = [1.0, 2.0, 3.0, 4.0];
    let mut output = [0.0; 4];
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
    SynchronousProbe
        .run_host(submission, &mut bindings, PcuInvocationParameters::empty())
        .expect("synchronous host profile runs");
    assert_eq!(
        output.map(f32::to_bits),
        [2.0_f32, 4.0, 6.0, 8.0].map(f32::to_bits)
    );

    let mut readonly_output = [0.0; 4];
    let bindings = [
        PcuHostScalarBinding {
            target: PcuBindingRef::new(0, 0),
            slice: PcuHostScalarSlice::Read(&input),
        },
        PcuHostScalarBinding {
            target: PcuBindingRef::new(0, 1),
            slice: PcuHostScalarSlice::Read(&readonly_output),
        },
    ];
    assert_eq!(
        pcu_alias::validate_host_scalar_bindings::<f32, ()>(submission, &bindings),
        Err(PcuHostDispatchError::AccessMismatch(PcuBindingRef::new(
            0, 1
        )))
    );
    readonly_output[0] = 1.0;
    assert_eq!(readonly_output[0].to_bits(), 1.0_f32.to_bits());
}

#[test]
fn cpu_reference_executes_macro_ir_and_preflights_unsupported_work() {
    let descriptors = matrix_map_bindings();
    let builder = matrix_map::<2, 3>(&descriptors).expect("valid map");
    let kernel = builder.ir();
    let submission = PcuDispatchSubmission {
        kernel: &kernel,
        shape: PcuInvocationShape::invocations(core::num::NonZeroU32::new(6).expect("nonzero")),
    };
    let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut output = [0.0; 6];
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
        .expect("reference executes the same IR as the GPU lowerers");
    assert_eq!(
        output.map(f32::to_bits),
        [2.0_f32, 4.0, 6.0, 8.0, 10.0, 12.0].map(f32::to_bits)
    );

    let invalid_ops = [PcuDispatchOp::Arithmetic(PcuDispatchAluOp::Min)];
    let invalid_kernel = pcu_alias::PcuDispatchKernelIr {
        ops: &invalid_ops,
        ..kernel
    };
    let invalid_submission = PcuDispatchSubmission {
        kernel: &invalid_kernel,
        shape: submission.shape,
    };
    let mut untouched = [7.0; 6];
    let mut invalid_bindings = [
        PcuHostScalarBinding {
            target: PcuBindingRef::new(0, 0),
            slice: PcuHostScalarSlice::Read(&input),
        },
        PcuHostScalarBinding {
            target: PcuBindingRef::new(0, 1),
            slice: PcuHostScalarSlice::ReadWrite(&mut untouched),
        },
    ];
    assert!(
        PcuF32Reference
            .run_host(
                invalid_submission,
                &mut invalid_bindings,
                PcuInvocationParameters::empty()
            )
            .is_err()
    );
    assert!(
        untouched
            .iter()
            .all(|value| value.to_bits() == 7.0_f32.to_bits())
    );

    let mut short_output = [9.0; 5];
    let mut short_bindings = [
        PcuHostScalarBinding {
            target: PcuBindingRef::new(0, 0),
            slice: PcuHostScalarSlice::Read(&input),
        },
        PcuHostScalarBinding {
            target: PcuBindingRef::new(0, 1),
            slice: PcuHostScalarSlice::ReadWrite(&mut short_output),
        },
    ];
    assert_eq!(
        PcuF32Reference.run_host_direct(
            submission,
            &mut short_bindings,
            PcuInvocationParameters::empty()
        ),
        Err(PcuF32ReferenceError::InvalidSubmission)
    );
    assert!(
        short_output
            .iter()
            .all(|value| value.to_bits() == 9.0_f32.to_bits())
    );
}

#[test]
fn cpu_reference_executes_grid_stride_iterations_beyond_the_launch_count() {
    let descriptors = tiled_grid_stride_map_bindings();
    let builder = tiled_grid_stride_map::<16>(&descriptors).expect("valid tiled map");
    let kernel = builder.ir();
    let submission = PcuDispatchSubmission {
        kernel: &kernel,
        shape: PcuInvocationShape::invocations(core::num::NonZeroU32::new(4).expect("nonzero")),
    };
    let input = [
        1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let mut output = [0.0; 16];
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
        .expect("CPU reference repeats each lane by the launch stride");
    assert_eq!(
        output.map(f32::to_bits),
        [
            2.0_f32, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0,
            30.0, 32.0,
        ]
        .map(f32::to_bits)
    );
}

#[test]
fn bounded_grid_stride_vectors_execute_and_lower_consistently() {
    // These finite, exact multiply-by-two values exercise coverage only; they do not claim
    // cross-backend parity for exceptional or rounding-sensitive f32 values.
    // The final case also keeps storage beyond the semantic extent as a padded-lane sentinel.
    let cases = [(4_u32, 16_u32), (8_u32, 8_u32), (8_u32, 5_u32)];
    for (width, extent) in cases {
        let descriptors = if width == 4 {
            tiled_grid_stride_map_bindings()
        } else {
            padded_grid_stride_map_bindings()
        };
        let builder = if width == 4 {
            tiled_grid_stride_map::<16>(&descriptors).expect("four-lane IR builds")
        } else if extent == 8 {
            padded_grid_stride_map::<8>(&descriptors).expect("equal-width IR builds")
        } else {
            padded_grid_stride_map::<5>(&descriptors).expect("padded-lane IR builds")
        };
        let kernel = builder.ir();
        assert_eq!(kernel.entry.logical_shape, [width, 1, 1]);
        assert_eq!(
            kernel.minimum_binding_elements(width),
            extent,
            "semantic extent differs from launch width in the test vector"
        );
        pcu_alias::validate_f32_map_kernel(&kernel).expect("shared validator admits the vector");
        let hip = fusion_pcu_rocm::lower_dispatch_to_hip_source(&kernel)
            .expect("HIP structural lowering accepts the vector");
        assert!(hip.contains(&format!("fusion_idx < {extent}u")));
        let mut spirv = fusion_pcu_spirv::PcuSpirvFixedSink::<4096>::new();
        fusion_pcu_spirv::lower_dispatch_to_spirv(
            &kernel,
            fusion_pcu_spirv::PcuSpirvLoweringOptions::minimal_shader(),
            &mut spirv,
        )
        .expect("SPIR-V structural lowering accepts the vector");
        assert!(!spirv.is_empty());

        let input = [
            1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ];
        let mut output = [f32::NAN; 16];
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
        let submission = PcuDispatchSubmission {
            kernel: &kernel,
            shape: PcuInvocationShape::invocations(
                core::num::NonZeroU32::new(width).expect("nonzero vector width"),
            ),
        };
        PcuF32Reference
            .run_host(submission, &mut bindings, PcuInvocationParameters::empty())
            .expect("CPU oracle executes every logical element");
        assert_eq!(
            output[..extent as usize]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            input[..extent as usize]
                .iter()
                .map(|value| (value * 2.0).to_bits())
                .collect::<Vec<_>>()
        );
        assert!(
            output[extent as usize..]
                .iter()
                .all(|value| value.to_bits() == f32::NAN.to_bits())
        );
    }
}

#[test]
fn grid_stride_extent_near_u32_limit_is_checked_without_host_allocation() {
    let descriptors = wide_grid_stride_map_bindings();
    let builder = wide_grid_stride_map::<{ u32::MAX as usize }>(&descriptors)
        .expect("maximum u32 extent is representable by the IR");
    let kernel = builder.ir();
    assert_eq!(kernel.minimum_binding_elements(250), u32::MAX);
    pcu_alias::validate_f32_map_kernel(&kernel).expect("maximum extent passes structural checks");
    let hip = fusion_pcu_rocm::lower_dispatch_to_hip_source(&kernel)
        .expect("HIP uses a wide loop index for the maximum extent");
    assert!(hip.contains("fusion_idx < 4294967295ull"));
    assert!(hip.contains("fusion_idx += 250ull"));
    let mut spirv = fusion_pcu_spirv::PcuSpirvFixedSink::<4096>::new();
    fusion_pcu_spirv::lower_dispatch_to_spirv(
        &kernel,
        fusion_pcu_spirv::PcuSpirvLoweringOptions::minimal_shader(),
        &mut spirv,
    )
    .expect("SPIR-V lowers the maximum extent structurally");
    assert!(!spirv.is_empty());
}

#[test]
fn malformed_grid_stride_ir_is_rejected_before_execution_or_lowering() {
    let descriptors = tiled_grid_stride_map_bindings();
    let builder = tiled_grid_stride_map::<16>(&descriptors).expect("valid grid-stride map");
    let kernel = builder.ir();
    let invalid_ops = [
        PcuDispatchOp::GridStrideLoop {
            extent: 0,
            body: match kernel.ops[0] {
                PcuDispatchOp::GridStrideLoop { body, .. } => body,
                _ => unreachable!("macro emitted a grid-stride loop"),
            },
        },
        PcuDispatchOp::Control(pcu_alias::PcuDispatchControlOp::Return),
    ];
    let invalid_kernel = pcu_alias::PcuDispatchKernelIr {
        ops: &invalid_ops,
        ..kernel
    };
    assert!(pcu_alias::validate_f32_map_kernel(&invalid_kernel).is_err());
    assert!(fusion_pcu_rocm::lower_dispatch_to_hip_source(&invalid_kernel).is_err());
    let mut spirv = fusion_pcu_spirv::PcuSpirvFixedSink::<128>::new();
    assert!(
        fusion_pcu_spirv::lower_dispatch_to_spirv(
            &invalid_kernel,
            fusion_pcu_spirv::PcuSpirvLoweringOptions::minimal_shader(),
            &mut spirv,
        )
        .is_err()
    );
    let mut output = [7.0_f32; 16];
    let mut bindings = [
        PcuHostScalarBinding {
            target: PcuBindingRef::new(0, 0),
            slice: PcuHostScalarSlice::Read(&[1.0_f32; 16]),
        },
        PcuHostScalarBinding {
            target: PcuBindingRef::new(0, 1),
            slice: PcuHostScalarSlice::ReadWrite(&mut output),
        },
    ];
    let submission = PcuDispatchSubmission {
        kernel: &invalid_kernel,
        shape: PcuInvocationShape::invocations(
            core::num::NonZeroU32::new(4).expect("nonzero width"),
        ),
    };
    assert!(
        PcuF32Reference
            .run_host(submission, &mut bindings, PcuInvocationParameters::empty())
            .is_err()
    );
    assert_eq!(output.map(f32::to_bits), [7.0_f32; 16].map(f32::to_bits));
}
