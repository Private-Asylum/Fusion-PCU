//! Strict source/graph/native compounds preserve ordered checks under independent permissions.
#[path = "../low_precision/ffi/ffi.rs"]
mod ffi;
#[path = "../checked_tensor/native/native.rs"]
#[allow(dead_code)]
mod owner;
#[path = "source/source.rs"]
#[allow(dead_code)]
mod source;
extern crate pcu_facade as fusion_pcu;
#[global_allocator]
static ALLOCATOR: ffi::CountingAllocator = ffi::CountingAllocator;
#[rustfmt::skip]use pcu_facade::{global,PcuScalar,PcuCheckedFloat,PcuNumericalMode,PcuNumericalOptions,PcuCompoundArithmeticPolicy,PcuPrecisionPolicy,PcuFloatUnderflowPolicy,PcuScalarType,dialect::tensor::{Graph,TensorElement,TensorScalarValue}};
#[rustfmt::skip]use criterion::{Criterion,criterion_group,criterion_main};
use fusion_pcu_cpu::PcuCpuPreparedTensorGraph;
use std::{
    rc::Rc,
    sync::atomic::{AtomicUsize, Ordering},
};
static SCORES: AtomicUsize = AtomicUsize::new(0);
fn score(_: &global::PcuInvocationCandidate<'_>) -> i128 {
    SCORES.fetch_add(1, Ordering::Relaxed);
    0
}
fn scalar<T: TensorElement + PcuCheckedFloat>(value: f32) -> T {
    T::as_scalar(if T::TYPE == PcuScalarType::F32 {
        TensorScalarValue::F32(value)
    } else {
        TensorScalarValue::F64(f64::from(value))
    })
    .unwrap()
}
fn same<T: PcuScalar>(actual: &[T], expected: &[T]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert_eq!(a.encode_le().as_ref(), b.encode_le().as_ref());
    }
}
fn native<T: TensorElement + PcuCheckedFloat>(
    op: usize,
    left: &[[T; 2]; 2],
    right: &[[T; 2]; 2],
    policy: PcuFloatUnderflowPolicy,
) -> Vec<T> {
    let zero: T = scalar(0.0);
    let mut result = vec![zero; if op == 1 { 1 } else { 4 }];
    match op {
        0 => {
            for (output_row, left_row) in result.chunks_mut(2).zip(left) {
                for (column, output) in output_row.iter_mut().enumerate() {
                    for (&left, right_row) in left_row.iter().zip(right) {
                        let product = left
                            .pcu_checked_mul_with_policy(right_row[column], policy)
                            .unwrap();
                        *output = output.pcu_checked_add_with_policy(product, policy).unwrap();
                    }
                }
            }
        }
        1 => {
            for (&left, &right) in left.as_flattened().iter().zip(right.as_flattened()) {
                let difference = left.pcu_checked_sub_with_policy(right, policy).unwrap();
                let square = difference
                    .pcu_checked_mul_with_policy(difference, policy)
                    .unwrap();
                result[0] = result[0]
                    .pcu_checked_add_with_policy(square, policy)
                    .unwrap();
            }
            result[0] = result[0]
                .pcu_checked_div_with_policy(scalar(4.0), policy)
                .unwrap();
        }
        _ => {
            for (index, (&weight, &gradient)) in left
                .as_flattened()
                .iter()
                .zip(right.as_flattened())
                .enumerate()
            {
                let product = gradient
                    .pcu_checked_mul_with_policy(scalar(0.5), policy)
                    .unwrap();
                result[index] = weight.pcu_checked_sub_with_policy(product, policy).unwrap();
            }
        }
    }
    result
}
#[allow(clippy::too_many_lines)] // Three ordered operations retain the same fresh-owner/read/drop/census boundary for every policy tuple.
fn width<T: TensorElement + PcuCheckedFloat>(
    criterion: &mut Criterion,
    options: PcuNumericalOptions,
    policy: PcuFloatUnderflowPolicy,
) {
    let banks = [[[1.0, 2.0], [3.0, 4.0]], [[2.0, 4.0], [6.0, 8.0]]]
        .map(|matrix| matrix.map(|row| row.map(scalar::<T>)));
    for op in 0..3 {
        let right = match op {
            0 => [[1.0, 0.0], [0.0, 1.0]].map(|row| row.map(scalar::<T>)),
            _ => [[scalar(0.0); 2]; 2],
        };
        let wants: [Vec<T>; 2] = std::array::from_fn(|bank| match op {
            0 => banks[bank].as_flattened().to_vec(),
            1 => vec![scalar(if bank == 0 { 7.5 } else { 30.0 })],
            _ => (if bank == 0 {
                [0.5, 1.0, 1.5, 2.0]
            } else {
                [1.0, 2.0, 3.0, 4.0]
            })
            .map(scalar::<T>)
            .to_vec(),
        });
        let mut graph = Graph::default();
        graph.set_numerical_mode(PcuNumericalMode::Strict);
        graph.set_numerical_options(options);
        let left = graph.input([2, 2], T::TYPE).unwrap();
        let rhs = graph.input([2, 2], T::TYPE).unwrap();
        let output = match op {
            0 => graph.matmul(left, rhs),
            1 => graph.mean_squared_error(left, rhs),
            _ => graph.sgd_update(left, rhs, 0.5),
        }
        .unwrap();
        graph
            .set_value_float_underflow_policy(output, policy)
            .unwrap();
        let mut plan = PcuCpuPreparedTensorGraph::<T>::prepare(&graph, &[output]).unwrap();
        let shape: Rc<[usize]> = Rc::from(if op == 1 { &[][..] } else { &[2, 2][..] });
        let sentinel = scalar(17.0);
        let mut observed = [sentinel; 7];
        let mut run = |route: usize, bank: usize| {
            let left = &banks[bank];
            let rhs = if op == 2 { left } else { &right };
            observed.fill(sentinel);
            match route {
                0 => {
                    let owner = match op {
                        0 => source::product(left, rhs),
                        1 => source::loss(left, rhs),
                        _ => source::update(left, rhs),
                    }
                    .unwrap();
                    owner.read_into(&mut observed).unwrap();
                    drop(owner);
                }
                1 => {
                    plan.execute(&[left.as_flattened(), rhs.as_flattened()])
                        .unwrap();
                    let owner = owner::HostOutput::new(plan.output(0).unwrap().to_vec(), &shape);
                    observed[..owner.data().len()].copy_from_slice(owner.data());
                    drop(owner);
                }
                _ => {
                    let owner = owner::HostOutput::new(native(op, left, rhs, policy), &shape);
                    observed[..owner.data().len()].copy_from_slice(owner.data());
                    drop(owner);
                }
            }
            same(&observed[..wants[bank].len()], &wants[bank]);
            same(
                &observed[wants[bank].len()..],
                &[sentinel; 7][wants[bank].len()..],
            );
        };
        for route in 0..3 {
            run(route, 0);
            run(route, 1);
        }
        let scores = SCORES.load(Ordering::Relaxed);
        let mut group = criterion.benchmark_group(format!(
            "cpu_compound_permissions/{:?}/{:?}/{:?}/{policy:?}/{op}",
            T::TYPE,
            options.compound_arithmetic,
            options.precision
        ));
        for (route, label) in [
            "ordinary_annotated_owner_read",
            "explicit_graph_owner_read",
            "native_ordered_owner_read",
        ]
        .into_iter()
        .enumerate()
        {
            let counts = ffi::count_heap(|| {
                for call in 0..64 {
                    run(route, call % 2);
                }
            });
            assert_eq!(
                (counts.allocations, counts.reallocations, counts.frees),
                (64, 0, 64)
            );
            assert_eq!(SCORES.load(Ordering::Relaxed), scores);
            println!(
                "census {:?}/{:?}/{:?}/{policy:?}/{op}/{label}:64 changing calls 64alloc0realloc64free",
                T::TYPE,
                options.compound_arithmetic,
                options.precision
            );
            group.bench_function(label, |bench| {
                let mut bank = 0;
                bench.iter(|| {
                    bank ^= 1;
                    run(route, std::hint::black_box(bank));
                });
            });
        }
        group.finish();
    }
}
fn benchmarks(criterion: &mut Criterion) {
    for compound_arithmetic in [
        PcuCompoundArithmeticPolicy::Checked,
        PcuCompoundArithmeticPolicy::BackendDefined,
    ] {
        for precision in [
            PcuPrecisionPolicy::Preserve,
            PcuPrecisionPolicy::BackendOptimized,
        ] {
            for policy in [
                PcuFloatUnderflowPolicy::IeeeAfterRounding,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                PcuFloatUnderflowPolicy::RejectSubnormalResult,
            ] {
                let options = PcuNumericalOptions {
                    compound_arithmetic,
                    precision,
                    ..Default::default()
                };
                global::configure(global::PcuExecutionPolicy {
                    backend: global::PcuBackendChoice::Cpu,
                    numerical_options: options,
                    float_underflow: policy,
                    score_invocation: Some(score),
                    ..Default::default()
                })
                .unwrap();
                global::clear_thread_cache().unwrap();
                width::<f32>(criterion, options, policy);
                width::<f64>(criterion, options, policy);
            }
        }
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
criterion_group!(benches, benchmarks);
criterion_main!(benches);
