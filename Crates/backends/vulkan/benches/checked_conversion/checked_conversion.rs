//! Matched annotated prepared/ordinary, explicit graph and native transactional conversion controls.
#[path = "../../tests/scalar_transport/device/device.rs"]
mod device;
#[path = "ffi/ffi.rs"]
mod ffi;
#[path = "../../tests/checked_conversion/oracle/oracle.rs"]
mod oracle;
#[path = "../../../cpu/tests/prepared_conversion/source/source.rs"]
#[allow(dead_code)]
mod source;
#[global_allocator]
static ALLOCATOR: ffi::CountingAllocator = ffi::CountingAllocator;
#[rustfmt::skip]
use criterion::{Criterion,BenchmarkId,Throughput,criterion_group,criterion_main};
#[rustfmt::skip]
use pcu_facade::{global,PcuScalar,PcuFloatUnderflowPolicy as Policy,PcuExecutionFault,PcuHostArgument,PcuBindingRef,PcuHostKernelBackend,PcuPreparedHostKernel,PcuImplementationRequirements,PcuDispatchOp,PcuDispatchDataOp};
use fusion_pcu_vulkan::{PcuVulkanBackend, PcuVulkanError};
use std::sync::atomic::{AtomicUsize, Ordering};
static SCORES: AtomicUsize = AtomicUsize::new(0);
fn score(_: &pcu_facade::PcuDeviceDescriptor<'_>, _: u64) -> i128 {
    SCORES.fetch_add(1, Ordering::Relaxed);
    0
}
fn score_invocation(_: &global::PcuInvocationCandidate<'_>) -> i128 {
    SCORES.fetch_add(1, Ordering::Relaxed);
    0
}
fn fault(error: &PcuVulkanError) -> PcuExecutionFault {
    let PcuVulkanError::Fault(fault) = error else {
        panic!("native failure {error:?}")
    };
    *fault
}
fn compare<T: PcuScalar + PartialEq, U: PcuScalar + PartialEq, const N: usize>(
    criterion: &mut Criterion,
    label: &str,
    banks: &[Vec<T>; 2],
    expected: &[(Vec<U>, Option<PcuExecutionFault>); 2],
    sentinel: U,
    mut run: impl FnMut(usize, &[T], &mut [U]) -> Result<(), PcuExecutionFault>,
) {
    let mut output = vec![sentinel; N + 3];
    let mut invoke = |route: usize, bank: usize| {
        output.fill(sentinel);
        let result = run(route, &banks[bank], &mut output);
        assert_eq!(result.err(), expected[bank].1);
        if expected[bank].1.is_none_or(|f| f.recovered) {
            for (a, b) in output[..N].iter().zip(&expected[bank].0) {
                assert_eq!(a.encode_le().as_ref(), b.encode_le().as_ref());
            }
        } else {
            assert!(output[..N].iter().all(|v| *v == sentinel));
        }
        assert!(output[N..].iter().all(|v| *v == sentinel));
    };
    for route in 0..4 {
        invoke(route, 0);
        invoke(route, 1);
    }
    let scores = SCORES.load(Ordering::Relaxed);
    let mut group = criterion.benchmark_group(label);
    group.throughput(Throughput::Elements(u64::try_from(N).unwrap()));
    for (route, name) in [
        "prepared_annotated",
        "ordinary_annotated",
        "explicit_graph",
        "native_checked",
    ]
    .into_iter()
    .enumerate()
    {
        let counts = ffi::count_heap(|| {
            for i in 0..64 {
                invoke(route, i % 2);
            }
        });
        assert_eq!(
            (counts.allocations, counts.reallocations, counts.frees),
            (0, 0, 0)
        );
        assert_eq!(SCORES.load(Ordering::Relaxed), scores);
        println!("census {label}/{N}/{name}:64 changing calls0 allocations0 reallocations0 frees");
        group.bench_function(BenchmarkId::new(name, N), |bench| {
            let mut bank = 0;
            bench.iter(|| {
                bank ^= 1;
                invoke(route, std::hint::black_box(bank));
            });
        });
    }
    group.finish();
}
fn narrow<const N: usize>(
    criterion: &mut Criterion,
    policy: Policy,
    backend: &PcuVulkanBackend,
    identity: pcu_facade::PcuStableDeviceIdentity,
) {
    for workload in 0..3 {
        let mut banks = [vec![1.25; N], vec![-2.5; N]];
        for input in &mut banks {
            if workload > 0 {
                input[0] = f64::MAX;
                input[N - 1] = if workload == 2 { f64::NAN } else { -f64::MAX };
            }
        }
        let expected = std::array::from_fn(|bank| {
            let mut first = None;
            let mut fatal = None;
            let data = banks[bank]
                .iter()
                .enumerate()
                .map(|(i, v)| match oracle::narrow(v.to_bits(), policy) {
                    Ok((bits, notice)) => {
                        if first.is_none() {
                            first = notice.map(|kind| PcuExecutionFault {
                                kind,
                                invocation_id: u64::try_from(i).unwrap(),
                                recovered: true,
                            });
                        }
                        f32::from_bits(bits)
                    }
                    Err(kind) => {
                        if fatal.is_none() {
                            fatal = Some(PcuExecutionFault {
                                kind,
                                invocation_id: u64::try_from(i).unwrap(),
                                recovered: false,
                            });
                        }
                        0.0
                    }
                })
                .collect();
            (data, fatal.or(first))
        });
        let mut prepared =
            source::narrow_clamp_prepare::<N, _>(&Configured(policy, backend)).unwrap();
        let bindings = source::narrow_clamp_bindings();
        let builder = source::narrow_clamp_ir::<N>(&bindings).unwrap();
        let kernel = builder.ir();
        let mut ops = kernel.ops.to_vec();
        for op in &mut ops {
            if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatConvert {
                underflow_policy,
                ..
            }) = op
            {
                *underflow_policy = policy;
            }
        }
        let kernel = pcu_facade::PcuDispatchKernelIr {
            ops: &ops,
            numerical_requirements: PcuImplementationRequirements {
                float_underflow: policy,
                ..kernel.numerical_requirements
            },
            ..kernel
        };
        let mut graph = backend.prepare_host_kernel(&kernel).unwrap();

        let mut native = ffi::NativeConversion::new(
            identity,
            u32::try_from(N).unwrap(),
            1,
            policy_code(policy),
            true,
        )
        .unwrap();
        compare::<f64, f32, N>(
            criterion,
            &format!("vulkan_checked_conversion/narrow/{policy:?}/{workload}"),
            &banks,
            &expected,
            99.0,
            |route, input, output| match route {
                0 => prepared(input, output).map_err(|error| fault(&error)),
                1 => source::narrow_clamp::<N>(input, output)
                    .map_err(|e| e.arithmetic_fault().unwrap()),
                2 => graph
                    .call(&mut [
                        PcuHostArgument::read(PcuBindingRef::new(0, 0), input),
                        PcuHostArgument::read_write(PcuBindingRef::new(0, 1), output),
                    ])
                    .map_err(|error| fault(&error)),
                _ => native
                    .call(ffi::bytes(input), ffi::bytes_mut(&mut output[..N]))
                    .unwrap()
                    .map_or(Ok(()), Err),
            },
        );
    }
}
struct Configured<'a>(Policy, &'a PcuVulkanBackend);
impl PcuHostKernelBackend for Configured<'_> {
    type Prepared = fusion_pcu_vulkan::PcuVulkanPreparedHost;
    type Error = PcuVulkanError;
    fn prepare_host_kernel(
        &self,
        k: &pcu_facade::PcuDispatchKernelIr<'_>,
    ) -> Result<Self::Prepared, Self::Error> {
        let mut ops = k.ops.to_vec();
        for op in &mut ops {
            if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatConvert {
                underflow_policy,
                ..
            }) = op
            {
                *underflow_policy = self.0;
            }
        }
        self.1
            .prepare_host_kernel(&pcu_facade::PcuDispatchKernelIr {
                ops: &ops,
                numerical_requirements: PcuImplementationRequirements {
                    float_underflow: self.0,
                    ..k.numerical_requirements
                },
                ..*k
            })
    }
}
fn widen<const N: usize>(
    criterion: &mut Criterion,
    policy: Policy,
    backend: &PcuVulkanBackend,
    identity: pcu_facade::PcuStableDeviceIdentity,
) {
    let banks = [vec![1.25_f32; N], vec![-2.5; N]];
    let expected = std::array::from_fn(|bank| {
        (
            banks[bank]
                .iter()
                .map(|v| f64::from_bits(oracle::widen(v.to_bits()).unwrap()))
                .collect(),
            None,
        )
    });
    let mut prepared = source::widen_clamp_prepare::<N, _>(&Configured(policy, backend)).unwrap();
    let bindings = source::widen_clamp_bindings();
    let builder = source::widen_clamp_ir::<N>(&bindings).unwrap();
    let mut graph = Configured(policy, backend)
        .prepare_host_kernel(&builder.ir())
        .unwrap();
    let mut native = ffi::NativeConversion::new(
        identity,
        u32::try_from(N).unwrap(),
        0,
        policy_code(policy),
        true,
    )
    .unwrap();
    compare::<f32, f64, N>(
        criterion,
        &format!("vulkan_checked_conversion/widen/{policy:?}"),
        &banks,
        &expected,
        99.0,
        |route, input, output| match route {
            0 => prepared(input, output).map_err(|error| fault(&error)),
            1 => source::widen_clamp::<N>(input, output).map_err(|e| e.arithmetic_fault().unwrap()),
            2 => graph
                .call(&mut [
                    PcuHostArgument::read(PcuBindingRef::new(0, 0), input),
                    PcuHostArgument::read_write(PcuBindingRef::new(0, 1), output),
                ])
                .map_err(|error| fault(&error)),
            _ => native
                .call(ffi::bytes(input), ffi::bytes_mut(&mut output[..N]))
                .unwrap()
                .map_or(Ok(()), Err),
        },
    );
}
const fn policy_code(policy: Policy) -> u32 {
    match policy {
        Policy::IeeeAfterRounding => 0,
        Policy::RejectSubnormalResult => 1,
        Policy::AllowGradualUnderflow => 2,
    }
}
fn benchmarks(criterion: &mut Criterion) {
    let (backend, identity) = device::selected();
    for policy in [
        Policy::IeeeAfterRounding,
        Policy::RejectSubnormalResult,
        Policy::AllowGradualUnderflow,
    ] {
        global::configure(global::PcuExecutionPolicy {
            backend: global::PcuBackendChoice::Vulkan,
            float_underflow: policy,
            score_device: score,
            score_invocation: Some(score_invocation),
            ..Default::default()
        })
        .unwrap();
        global::clear_thread_cache().unwrap();
        narrow::<1>(criterion, policy, &backend, identity);
        narrow::<65>(criterion, policy, &backend, identity);
        widen::<1>(criterion, policy, &backend, identity);
        widen::<65>(criterion, policy, &backend, identity);
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
criterion_group!(benches, benchmarks);
criterion_main!(benches);
