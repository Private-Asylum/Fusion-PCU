//! Matched source/IR/native views: fresh RAM owners versus already-cached owners.
//!
//! Both boundaries include native `MatMul`, a native-to-encoded output view,
//! explicit terminal readback and output Drop. Fresh includes two encoded input
//! uploads and their first views; cached retains and primes inputs before timing.
//! These are different workload boundaries, not interchangeable speedup claims.
use super::activity;
#[rustfmt::skip]
use fusion_pcu::{
    dialect::tensor::{
        Graph,
        TensorArithmeticCapability,
        TensorArithmeticRewritePolicy,
        TensorPointwiseGroupingPolicy,
    },
    global::{
        self,
        PcuBackendChoice,
        PcuExecutionPolicy,
        PcuSourceShape,
    },
    PcuCompoundArithmeticPolicy,
    PcuImplementationRequirements,
    PcuNumericalOptions,
    PcuPrecisionPolicy,
    PcuScalarType,
};
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxCheckedProgramInput,
    MlxProgramInput,
    MlxRuntime,
};
#[rustfmt::skip]
use std::{
    hint::black_box,
    sync::Arc,
};
use criterion::Criterion;
use super::source;

#[allow(clippy::too_many_lines)] // Freeze four matched plans, then share the exact measured execution boundary.
#[cfg_attr(feature = "mlx-view-census", allow(clippy::needless_pass_by_ref_mut))] // Shared Criterion entry; census registers no timed samples.
pub fn run<const N: usize>(criterion: &mut Criterion) {
    #[cfg(feature = "mlx-view-census")]
    let _ = criterion;
    activity::gpu_idle_guard();
    global::configure(PcuExecutionPolicy {
        backend: PcuBackendChoice::Mlx,
        #[cfg(feature = "mlx-view-census")]
        score_device: super::census::score_device,
        ..Default::default()
    })
    .unwrap();
    let session = MlxRuntime::load_default().unwrap().open_gpu(0).unwrap();
    let shape = PcuSourceShape::FixedMatrix {
        rows: N,
        columns: N,
    };
    let requirements = PcuImplementationRequirements::default();
    let identity_capture = global::__pcu_capture_tensor_program(
        [shape],
        requirements.float_underflow,
        requirements.numerical_mode,
        requirements.numerical_options,
        source::identity_matrix::__pcu_capture_entry::<N, N>,
    )
    .unwrap();
    let captured_identity = session
        .prepare_checked_program(Arc::clone(identity_capture.program()), requirements)
        .unwrap();
    let matrix_capture = global::__pcu_capture_tensor_program(
        [shape; 2],
        requirements.float_underflow,
        requirements.numerical_mode,
        requirements.numerical_options,
        source::matrix::__pcu_capture_entry::<N, N, N>,
    )
    .unwrap();
    let captured_matrix = session
        .prepare_program(Arc::clone(matrix_capture.program()))
        .unwrap();
    let mut identity_graph = Graph::try_new().unwrap();
    let input = identity_graph.input([N, N], PcuScalarType::F32).unwrap();
    let identity_program = identity_graph
        .into_selected_program(
            &[input],
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::Disabled,
        )
        .unwrap();
    let explicit_identity = session
        .prepare_checked_program(Arc::new(identity_program), requirements)
        .unwrap();
    let mut matrix_graph = Graph::try_new().unwrap();
    matrix_graph.set_numerical_options(PcuNumericalOptions {
        compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
        precision: PcuPrecisionPolicy::BackendOptimized,
        ..Default::default()
    });
    let a = matrix_graph.input([N, N], PcuScalarType::F32).unwrap();
    let b = matrix_graph.input([N, N], PcuScalarType::F32).unwrap();
    let product = matrix_graph.matmul(a, b).unwrap();
    let explicit_matrix = session
        .prepare_matmul(&matrix_graph, matrix_graph.node(product).unwrap())
        .unwrap();
    let native_matrix = session
        .prepare_native_matmul_control([N, N], [N, N])
        .unwrap();
    // Small integer products are exactly representable; these three literal
    // factors form an oracle independent of native contraction/reduction order.
    let banks = [1.0_f32, 2.0, 3.0].map(|phase| ([[phase; N]; N], [[4.0 - phase; N]; N]));
    let source_owners = banks.each_ref().map(|(left, right)| {
        (
            source::identity_matrix(left).unwrap(),
            source::identity_matrix(right).unwrap(),
        )
    });
    let native_owners = banks.each_ref().map(|(left, right)| {
        (
            session.upload_encoded(left.as_flattened()).unwrap(),
            session.upload_encoded(right.as_flattened()).unwrap(),
        )
    });
    let mut output = vec![77.0_f32; N * N + 1];
    for fresh in [false, true] {
        let mut execute = |route: usize, bank: usize, verify: bool| {
            let (left, right) = &banks[bank];
            if route == 0 {
                let staged;
                let (a, b) = if fresh {
                    staged = (
                        source::identity_matrix(left).unwrap(),
                        source::identity_matrix(right).unwrap(),
                    );
                    (&staged.0, &staged.1)
                } else {
                    (&source_owners[bank].0, &source_owners[bank].1)
                };
                let result = source::matrix::<N, N, N>(a, b).unwrap();
                source::consume_identity(result)
                    .unwrap()
                    .read_into(&mut output)
                    .unwrap();
            } else {
                let identity = if route == 1 {
                    &captured_identity
                } else {
                    &explicit_identity
                };
                let staged;
                let (a, b) = if fresh {
                    staged = if route == 3 {
                        (
                            session.upload_encoded(left.as_flattened()).unwrap(),
                            session.upload_encoded(right.as_flattened()).unwrap(),
                        )
                    } else {
                        (
                            identity.execute_host(left.as_flattened()).unwrap(),
                            identity.execute_host(right.as_flattened()).unwrap(),
                        )
                    };
                    (&staged.0, &staged.1)
                } else {
                    (&native_owners[bank].0, &native_owners[bank].1)
                };
                let left_view = a.native_f32_view([N, N]).unwrap();
                let right_view = b.native_f32_view([N, N]).unwrap();
                let result = match route {
                    1 => {
                        let [a, b] = captured_matrix.matmul().plan().inputs();
                        captured_matrix
                            .execute_mixed::<f32>(&[
                                (a, MlxProgramInput::Resident(&left_view)),
                                (b, MlxProgramInput::Resident(&right_view)),
                            ])
                            .unwrap()
                    }
                    2 => session
                        .execute_matmul(&explicit_matrix, &left_view, &right_view)
                        .unwrap(),
                    3 => session
                        .execute_native_matmul_control(&native_matrix, &left_view, &right_view)
                        .unwrap(),
                    _ => unreachable!("four registered matched routes"),
                };
                let encoded = result.encoded_f32_view().unwrap();
                if route == 3 {
                    encoded.read_into(&mut output).unwrap();
                } else {
                    identity
                        .execute_mixed(&[(
                            identity.plan().input(),
                            MlxCheckedProgramInput::Resident(&encoded),
                        )])
                        .unwrap()
                        .read_into(&mut output)
                        .unwrap();
                }
            }
            if verify {
                let factor = [3.0_f32, 4.0, 3.0][bank];
                let expected = f32::from(u16::try_from(N).unwrap()) * factor;
                assert!(
                    output[..N * N]
                        .iter()
                        .all(|value| value.to_bits() == expected.to_bits())
                );
                assert_eq!(output[N * N].to_bits(), 77.0_f32.to_bits());
            }
            black_box(&output);
        };
        let boundary = if fresh {
            "fresh_host_first_views"
        } else {
            "cached_resident_views"
        };
        #[cfg(not(feature = "mlx-view-census"))]
        let mut group = criterion.benchmark_group(format!("mlx_source_views/{N}/{boundary}"));
        for (route, name) in [
            "ordinary_pcu",
            "captured_pcu",
            "explicit_graph",
            "native_retained",
        ]
        .into_iter()
        .enumerate()
        {
            // All three cached banks must be primed, even when Criterion filters
            // to one route. Every independently filtered route gets a full oracle.
            for bank in 0..banks.len() {
                execute(route, bank, true);
            }
            #[cfg(feature = "mlx-view-census")]
            super::census::report(&format!("{N}/{boundary}/{name}"), fresh, |bank| {
                execute(route, bank, false);
            });
            #[cfg(not(feature = "mlx-view-census"))]
            {
                let mut bank = 0;
                group.bench_function(name, |bencher| {
                    bencher.iter(|| {
                        bank = (bank + 1) % banks.len();
                        execute(route, bank, false);
                    });
                });
            }
            for bank in 0..banks.len() {
                execute(route, bank, true);
            }
        }
        #[cfg(not(feature = "mlx-view-census"))]
        group.finish();
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
    activity::gpu_post_guard();
}
