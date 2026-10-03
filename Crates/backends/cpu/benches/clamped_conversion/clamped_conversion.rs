//! Matched annotated prepared/ordinary, explicit graph and native transactional conversion controls.
#[path = "../low_precision/ffi/ffi.rs"]
mod ffi;
#[path = "../../tests/clamped_conversion/oracle/oracle.rs"]
mod oracle;
#[path = "../../tests/prepared_conversion/source/source.rs"]
#[allow(dead_code)]
mod source;
#[global_allocator]
static ALLOCATOR: ffi::CountingAllocator = ffi::CountingAllocator;
#[rustfmt::skip]
use criterion::{Criterion,BenchmarkId,Throughput,criterion_group,criterion_main};
#[rustfmt::skip]
use pcu_facade::{global,PcuScalar,PcuClampedError,PcuClampedFloatConversion,PcuCheckedFloatWidening,PcuFloatUnderflowPolicy as Policy,PcuExecutionFault,PcuHostArgument,PcuBindingRef,PcuHostKernelBackend,PcuPreparedHostKernel,PcuImplementationRequirements,PcuDispatchOp,PcuDispatchDataOp};
use fusion_pcu_cpu::PcuCpuHostBackend;
use std::sync::atomic::{AtomicUsize, Ordering};
static SCORES: AtomicUsize = AtomicUsize::new(0);
fn score(_: &pcu_facade::PcuDeviceDescriptor<'_>, _: u64) -> i128 {
    SCORES.fetch_add(1, Ordering::Relaxed);
    0
}
fn native<T: Copy, U: Copy>(
    input: &[T],
    output: &mut [U],
    evaluate: impl Fn(T) -> Result<U, PcuClampedError<U>>,
) -> Result<(), PcuExecutionFault> {
    let mut first = None;
    for (i, &value) in input.iter().enumerate() {
        match evaluate(value) {
            Ok(_) => {}
            Err(PcuClampedError::Range(fault)) => {
                if first.is_none() {
                    first = Some(PcuExecutionFault {
                        kind: fault.kind(),
                        invocation_id: u64::try_from(i).unwrap(),
                        recovered: true,
                    });
                }
            }
            Err(PcuClampedError::Fatal(kind)) => {
                return Err(PcuExecutionFault {
                    kind,
                    invocation_id: u64::try_from(i).unwrap(),
                    recovered: false,
                });
            }
        }
    }
    for (&value, out) in input.iter().zip(output) {
        *out = match evaluate(value) {
            Ok(value) => value,
            Err(PcuClampedError::Range(fault)) => fault.clamped_value(),
            Err(PcuClampedError::Fatal(_)) => unreachable!("native preflight"),
        };
    }
    first.map_or(Ok(()), Err)
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
fn narrow<const N: usize>(criterion: &mut Criterion, policy: Policy) {
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
        let mut prepared = source::narrow_clamp_prepare::<N, _>(&Configured(policy)).unwrap();
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
        let mut graph = PcuCpuHostBackend::scalar()
            .prepare_host_kernel(&kernel)
            .unwrap();

        compare::<f64, f32, N>(
            criterion,
            &format!("cpu_clamped_conversion/narrow/{policy:?}/{workload}"),
            &banks,
            &expected,
            99.0,
            |route, input, output| match route {
                0 => prepared(input, output).map_err(|e| e.fault().unwrap()),
                1 => source::narrow_clamp::<N>(input, output)
                    .map_err(|e| e.arithmetic_fault().unwrap()),
                2 => graph
                    .call(&mut [
                        PcuHostArgument::read(PcuBindingRef::new(0, 0), input),
                        PcuHostArgument::read_write(PcuBindingRef::new(0, 1), output),
                    ])
                    .map_err(|e| e.fault().unwrap()),
                _ => native(input, &mut output[..N], |v| {
                    v.pcu_clamped_to_f32_with_policy(policy)
                }),
            },
        );
    }
}
struct Configured(Policy);
impl PcuHostKernelBackend for Configured {
    type Prepared = fusion_pcu_cpu::PcuCpuPreparedHost;
    type Error = fusion_pcu_cpu::PcuCpuHostError;
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
        PcuCpuHostBackend::scalar().prepare_host_kernel(&pcu_facade::PcuDispatchKernelIr {
            ops: &ops,
            numerical_requirements: PcuImplementationRequirements {
                float_underflow: self.0,
                ..k.numerical_requirements
            },
            ..*k
        })
    }
}
fn widen<const N: usize>(criterion: &mut Criterion, policy: Policy) {
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
    let mut prepared = source::widen_clamp_prepare::<N, _>(&Configured(policy)).unwrap();
    let bindings = source::widen_clamp_bindings();
    let builder = source::widen_clamp_ir::<N>(&bindings).unwrap();
    let mut graph = Configured(policy)
        .prepare_host_kernel(&builder.ir())
        .unwrap();
    compare::<f32, f64, N>(
        criterion,
        &format!("cpu_clamped_conversion/widen/{policy:?}"),
        &banks,
        &expected,
        99.0,
        |route, input, output| match route {
            0 => prepared(input, output).map_err(|e| e.fault().unwrap()),
            1 => source::widen_clamp::<N>(input, output).map_err(|e| e.arithmetic_fault().unwrap()),
            2 => graph
                .call(&mut [
                    PcuHostArgument::read(PcuBindingRef::new(0, 0), input),
                    PcuHostArgument::read_write(PcuBindingRef::new(0, 1), output),
                ])
                .map_err(|e| e.fault().unwrap()),
            _ => native(input, &mut output[..N], |v| {
                v.pcu_checked_to_f64().map_err(PcuClampedError::Fatal)
            }),
        },
    );
}
fn benchmarks(criterion: &mut Criterion) {
    for policy in [
        Policy::IeeeAfterRounding,
        Policy::RejectSubnormalResult,
        Policy::AllowGradualUnderflow,
    ] {
        global::configure(global::PcuExecutionPolicy {
            backend: global::PcuBackendChoice::Cpu,
            float_underflow: policy,
            score_device: score,
            ..Default::default()
        })
        .unwrap();
        global::clear_thread_cache().unwrap();
        narrow::<1>(criterion, policy);
        narrow::<65>(criterion, policy);
        widen::<1>(criterion, policy);
        widen::<65>(criterion, policy);
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
criterion_group!(benches, benchmarks);
criterion_main!(benches);
