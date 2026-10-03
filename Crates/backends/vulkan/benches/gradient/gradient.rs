//! Complete tiny-model forward/loss/target-selected reverse/update source and native peers.
//! Native fuses the same checked scalar steps and packs samples+targets in one binding;
//! this valid dyadic control is not an independent arithmetic or fault-provenance oracle.
extern crate pcu_facade as fusion_pcu;
#[path = "../owned_leaf/bytes/bytes.rs"]
mod bytes;
#[path = "../../tests/scalar_transport/device/device.rs"]
mod device;
#[path = "../owned_leaf/ffi/ffi.rs"]
#[allow(dead_code)] // Separate transfer/pointwise/compound controls retain complete owning targets.
mod ffi;
#[path = "../../../cpu/benches/low_precision/ffi/ffi.rs"]
mod heap;
#[path = "../checked_compound/oracle/oracle.rs"]
#[allow(dead_code)] // Independent compound fault and rounding fixtures live in their owning test.
mod oracle;
#[path = "source/source.rs"]
mod source;
#[global_allocator]
static ALLOCATOR: heap::CountingAllocator = heap::CountingAllocator;
#[rustfmt::skip]
use criterion::{Criterion, BenchmarkId, criterion_group, criterion_main};
#[rustfmt::skip]
use pcu_facade::{global, PcuFloatUnderflowPolicy, PcuNumericalMode, PcuNumericalOptions,
 PcuCompoundArithmeticPolicy, PcuPrecisionPolicy, PcuExecutionFaultKind};
use pcu_facade::dialect::tensor::Graph;
#[rustfmt::skip]
use fusion_pcu_vulkan::{PcuVulkanPreparedTensorGraph, PcuVulkanTensorInput};
static SCORES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
fn score(_: &pcu_facade::PcuDeviceDescriptor<'_>, _: u64) -> i128 {
    SCORES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    0
}
fn read<T: oracle::Format>(actual: &[T], expected: &[T]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert_eq!(a.bits(), b.bits());
    }
}
fn plan<T: oracle::Format>(
    backend: &fusion_pcu_vulkan::PcuVulkanBackend,
    policy: PcuFloatUnderflowPolicy,
    options: PcuNumericalOptions,
) -> PcuVulkanPreparedTensorGraph<T> {
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    graph.set_numerical_options(options);
    let input = graph.input([2, 2], T::TYPE).unwrap();
    let weights = graph.input([2, 1], T::TYPE).unwrap();
    let target = graph.input([2, 1], T::TYPE).unwrap();
    let projected = graph.matmul(input, weights).unwrap();
    let predicted = graph.relu(projected).unwrap();
    let loss = graph
        .mean_squared_error_typed(
            graph.typed_view::<T>(predicted).unwrap(),
            graph.typed_view::<T>(target).unwrap(),
        )
        .unwrap()
        .erase();
    // Captured source policies govern the introduced reverse nodes, not mutable defaults.
    for value in [projected, predicted, loss] {
        graph
            .set_value_float_underflow_policy(value, policy)
            .unwrap();
    }
    let gradient = graph.backward_mse_for(loss, weights).unwrap();
    let output = graph.sgd_update(weights, gradient, 0.5).unwrap();
    graph
        .set_value_float_underflow_policy(output, policy)
        .unwrap();
    PcuVulkanPreparedTensorGraph::<T>::prepare(backend, &graph, &[output]).unwrap()
}
#[allow(clippy::too_many_lines, clippy::significant_drop_tightening)] // Six complete two-step owner boundaries remain adjacent to independent output/census assertions.
fn run<T: oracle::Format>(
    c: &mut Criterion,
    backend: &fusion_pcu_vulkan::PcuVulkanBackend,
    identity: pcu_facade::PcuStableDeviceIdentity,
    policy: PcuFloatUnderflowPolicy,
    options: PcuNumericalOptions,
) {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        float_underflow: policy,
        numerical_options: options,
        score_device: score,
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
    let input = [
        [T::value(1.0), T::value(0.0)],
        [T::value(0.0), T::value(1.0)],
    ];
    let target = [[T::value(1.0)], [T::value(0.0)]];
    let weights = [
        [[T::value(3.0)], [T::value(-1.0)]],
        [[T::value(4.0)], [T::value(-1.0)]],
    ];
    // Independent dyadic oracle: w1=(w0+1)/2 and w2=(w0+3)/4; inactive weight remains -1.
    let expected_first = [[2.0, -1.0], [2.5, -1.0]].map(|v| v.map(T::value));
    let expected_second = [[1.5, -1.0], [1.75, -1.0]].map(|v| v.map(T::value));
    let source_input = source::retain_input(&input).unwrap();
    let source_target = source::retain_vector(&target).unwrap();
    let source_weights = weights
        .each_ref()
        .map(|w| source::retain_vector(w).unwrap());
    let graph_input = backend.upload_owned(input.as_flattened()).unwrap();
    let graph_target = backend.upload_owned(target.as_flattened()).unwrap();
    let graph_weights = weights
        .each_ref()
        .map(|w| backend.upload_owned(w.as_flattened()).unwrap());
    let mut prepared = plan::<T>(backend, policy, options);
    let mut native = ffi::pointwise::NativePointwise::training(
        identity,
        T::TYPE,
        match policy {
            PcuFloatUnderflowPolicy::IeeeAfterRounding => 0,
            PcuFloatUnderflowPolicy::RejectSubnormalResult => 1,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow => 2,
        },
    )
    .unwrap();
    let packed = [
        T::value(1.0),
        T::value(0.0),
        T::value(0.0),
        T::value(1.0),
        T::value(1.0),
        T::value(0.0),
    ];
    let native_input = native.upload(bytes::read(&packed)).unwrap();
    let native_weights = weights
        .each_ref()
        .map(|w| native.upload(bytes::read(w.as_flattened())).unwrap());
    // Forward MSE must fault on MAX squared, despite the finite half-rate update otherwise.
    let invalid = [[T::maximum()], [T::value(-1.0)]];
    assert_eq!(
        source::train(&input, &invalid, &target)
            .unwrap_err()
            .arithmetic_fault()
            .unwrap()
            .kind,
        PcuExecutionFaultKind::ArithmeticOverflow
    );
    assert!(matches!(
        prepared
            .execute_owned(&[
                PcuVulkanTensorInput::Host(input.as_flattened()),
                PcuVulkanTensorInput::Host(invalid.as_flattened()),
                PcuVulkanTensorInput::Host(target.as_flattened()),
            ])
            .err()
            .unwrap(),
        fusion_pcu_vulkan::PcuVulkanTensorError::Graph(
            pcu_facade::dialect::tensor::TensorError::CompoundArithmeticFault {
                kind: PcuExecutionFaultKind::ArithmeticOverflow,
                ..
            }
        )
    ));
    assert_eq!(
        native
            .host(bytes::read(&packed), bytes::read(invalid.as_flattened()))
            .err()
            .unwrap()
            .kind,
        PcuExecutionFaultKind::ArithmeticOverflow
    );
    let sentinel = T::value(117.0);
    let mut observed_first = [sentinel; 5];
    let mut observed_second = [sentinel; 5];
    let mut invoke = |route, bank: usize| {
        match route {
            0 | 3 => {
                let first = if route == 0 {
                    source::train(&input, &weights[bank], &target)
                } else {
                    source::train(&source_input, &source_weights[bank], &source_target)
                }
                .unwrap();
                let second = if route == 0 {
                    source::train(&input, &first, &target)
                } else {
                    source::train(&source_input, &first, &source_target)
                }
                .unwrap();
                first.read_into(&mut observed_first).unwrap();
                second.read_into(&mut observed_second).unwrap();
                drop(second);
                drop(first);
            }
            1 | 4 => {
                let first = prepared
                    .execute_owned(&if route == 1 {
                        [
                            PcuVulkanTensorInput::Host(input.as_flattened()),
                            PcuVulkanTensorInput::Host(weights[bank].as_flattened()),
                            PcuVulkanTensorInput::Host(target.as_flattened()),
                        ]
                    } else {
                        [
                            PcuVulkanTensorInput::Owned(&graph_input),
                            PcuVulkanTensorInput::Owned(&graph_weights[bank]),
                            PcuVulkanTensorInput::Owned(&graph_target),
                        ]
                    })
                    .unwrap();
                let second = prepared
                    .execute_owned(&if route == 1 {
                        [
                            PcuVulkanTensorInput::Host(input.as_flattened()),
                            PcuVulkanTensorInput::Owned(&first),
                            PcuVulkanTensorInput::Host(target.as_flattened()),
                        ]
                    } else {
                        [
                            PcuVulkanTensorInput::Owned(&graph_input),
                            PcuVulkanTensorInput::Owned(&first),
                            PcuVulkanTensorInput::Owned(&graph_target),
                        ]
                    })
                    .unwrap();
                first.read_into(&mut observed_first).unwrap();
                second.read_into(&mut observed_second).unwrap();
                drop(second);
                drop(first);
            }
            2 | 5 => {
                let first = if route == 2 {
                    native.host(
                        bytes::read(&packed),
                        bytes::read(weights[bank].as_flattened()),
                    )
                } else {
                    native.borrowed(&native_input, &native_weights[bank])
                }
                .unwrap();
                let second = if route == 2 {
                    native.mixed(bytes::read(&packed), &first)
                } else {
                    native.borrowed(&native_input, &first)
                }
                .unwrap();
                first.read(bytes::write(&mut observed_first[..2]));
                second.read(bytes::write(&mut observed_second[..2]));
                drop(second);
                drop(first);
            }
            _ => unreachable!(),
        }
        read(&observed_first[..2], &expected_first[bank]);
        read(&observed_second[..2], &expected_second[bank]);
        read(&observed_first[2..], &[sentinel; 3]);
        read(&observed_second[2..], &[sentinel; 3]);
    };
    for route in 0..6 {
        invoke(route, 0);
        invoke(route, 1);
    }
    let scores = SCORES.load(std::sync::atomic::Ordering::Relaxed);
    let mut group = c.benchmark_group(format!(
        "vulkan_requested_gradient/{:?}/{policy:?}/{options:?}",
        T::TYPE
    ));
    for (route, label) in [
        "source_host_two_steps",
        "graph_host_two_steps",
        "native_host_two_steps",
        "source_borrowed_two_steps",
        "graph_borrowed_two_steps",
        "native_borrowed_two_steps",
    ]
    .into_iter()
    .enumerate()
    {
        let counts = heap::count_heap(|| {
            for bank in 0..64 {
                invoke(route, bank % 2);
            }
        });
        assert_eq!(
            (counts.allocations, counts.reallocations, counts.frees),
            (0, 0, 0)
        );
        assert_eq!(SCORES.load(std::sync::atomic::Ordering::Relaxed), scores);
        println!(
            "census {:?}/{policy:?}/{options:?}/{label}:64changing two-step calls0Rustheap; two fresh outputs/read/drop and forward MSE retained; SDK unknown",
            T::TYPE
        );
        let mut bank = 0;
        group.bench_function(BenchmarkId::new(label, 2), |b| {
            b.iter(|| {
                bank ^= 1;
                invoke(route, bank);
            });
        });
    }
    group.finish();
}
fn benchmark(c: &mut Criterion) {
    let (backend, identity) = device::selected();
    for compound_arithmetic in [
        PcuCompoundArithmeticPolicy::Checked,
        PcuCompoundArithmeticPolicy::BackendDefined,
    ] {
        for precision in [
            PcuPrecisionPolicy::Preserve,
            PcuPrecisionPolicy::BackendOptimized,
        ] {
            let options = PcuNumericalOptions {
                compound_arithmetic,
                precision,
                ..Default::default()
            };
            for policy in [
                PcuFloatUnderflowPolicy::IeeeAfterRounding,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                PcuFloatUnderflowPolicy::RejectSubnormalResult,
            ] {
                run::<f32>(c, &backend, identity, policy, options);
                run::<f64>(c, &backend, identity, policy, options);
            }
        }
    }
}
criterion_group!(benches, benchmark);
criterion_main!(benches);
