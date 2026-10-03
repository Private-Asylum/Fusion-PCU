//! Genuine overwritten-effect source, hand-authored SSA and fixed native checked Add.
#[path = "graph/graph.rs"]
mod graph;
#[rustfmt::skip]
use super::{
    black_box,
    fault,
    ffi,
    global,
    oracle::Format,
    Criterion,
    PcuBindingRef,
    PcuExecutionFault,
    PcuFloatUnderflowPolicy as Uf,
    PcuHostArgument,
    PcuPreparedHostKernel,
    PcuRangePolicy as Range,
    PcuVulkanBackend,
    SCORES,
};
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuCheckedFloat,
    PcuExecutionFaultKind,
    PcuHostKernelBackend,
    PcuImplementationRequirements,
    PcuNumericalMode,
    PcuStableDeviceIdentity,
};
use std::sync::{atomic::Ordering, Mutex};
static EXPECTED: Mutex<Option<(PcuImplementationRequirements, usize)>> = Mutex::new(None);

// Scoring runs only during cold capture; retained calls never acquire this lock.
pub fn assert_candidate(kernel: &pcu_facade::PcuDispatchKernelIr<'_>) {
    let expected = *EXPECTED.lock().unwrap();
    if let Some((requirements, count)) = expected {
        assert_complete_policy(kernel, requirements, count);
    }
}

pub fn expect(requirements: PcuImplementationRequirements, count: usize) {
    *EXPECTED.lock().unwrap() = Some((requirements, count));
}
pub fn clear_expected() {
    *EXPECTED.lock().unwrap() = None;
}
pub fn verified(
    backend: &PcuVulkanBackend,
    requirements: PcuImplementationRequirements,
    count: usize,
) -> VerifiedBackend<'_> {
    expect(requirements, count);
    VerifiedBackend {
        backend,
        requirements,
        count,
    }
}
pub struct VerifiedBackend<'a> {
    pub backend: &'a PcuVulkanBackend,
    pub requirements: PcuImplementationRequirements,
    pub count: usize,
}
impl PcuHostKernelBackend for VerifiedBackend<'_> {
    type Prepared = <PcuVulkanBackend as PcuHostKernelBackend>::Prepared;
    type Error = <PcuVulkanBackend as PcuHostKernelBackend>::Error;
    fn prepare_host_kernel(
        &self,
        kernel: &pcu_facade::PcuDispatchKernelIr<'_>,
    ) -> Result<Self::Prepared, Self::Error> {
        assert_complete_policy(kernel, self.requirements, self.count);
        self.backend.prepare_host_kernel(kernel)
    }
}
fn assert_policy(
    kernel: &pcu_facade::PcuDispatchKernelIr<'_>,
    requirements: PcuImplementationRequirements,
) {
    assert_complete_policy(kernel, requirements, 1);
}
fn assert_complete_policy(
    kernel: &pcu_facade::PcuDispatchKernelIr<'_>,
    requirements: PcuImplementationRequirements,
    expected_count: usize,
) {
    assert_eq!(kernel.numerical_requirements, requirements);
    let mut count = 0;
    for op in kernel.ops {
        if let pcu_facade::PcuDispatchOp::Data(
            pcu_facade::PcuDispatchDataOp::CheckedFloatBinary {
                range_policy,
                underflow_policy,
                ..
            },
        ) = op
        {
            assert_eq!(*range_policy, requirements.range_policy);
            assert_eq!(*underflow_policy, requirements.float_underflow);
            count += 1;
        }
    }
    assert_eq!(count, expected_count);
}

#[pcu(crate_path=::pcu_facade,invocations=N)]
fn overwritten_add<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    let mut value = input[id];
    value += value;
    value = input[id];
    output[id] = value;
}

fn verify<T: Format, const N: usize>(
    entry: &mut impl FnMut(&[T], &mut [T]) -> Result<(), PcuExecutionFault>,
    input: &mut [T],
    output: &mut [T],
    range: Range,
) {
    for phase in 0..3 {
        for (lane, value) in input.iter_mut().enumerate() {
            *value = T::from(
                T::ONE
                    | if (lane + phase).is_multiple_of(2) {
                        0
                    } else {
                        T::SIGN
                    },
            );
        }
        entry(input, output).unwrap();
        assert!(
            output[..N]
                .iter()
                .zip(input.iter())
                .all(|(a, b)| a.bits() == b.bits())
        );
        assert!(output[N..].iter().all(|x| x.bits() == T::ONE + 1));
    }
    input[0] = T::from(T::MAX);
    let before = output.iter().map(|x| x.bits()).collect::<Vec<_>>();
    assert_eq!(
        entry(input, output),
        Err(PcuExecutionFault {
            invocation_id: 0,
            kind: PcuExecutionFaultKind::ArithmeticOverflow,
            recovered: range == Range::Clamp
        })
    );
    if range == Range::Reject {
        assert_eq!(output.iter().map(|x| x.bits()).collect::<Vec<_>>(), before);
    } else {
        assert_eq!(output[0].bits(), T::MAX);
    }
    input[0] = T::from(T::SIGN - 1);
    let before = output.iter().map(|x| x.bits()).collect::<Vec<_>>();
    assert_eq!(
        entry(input, output),
        Err(PcuExecutionFault {
            invocation_id: 0,
            kind: PcuExecutionFaultKind::InvalidFloatingOperand,
            recovered: false
        })
    );
    assert_eq!(output.iter().map(|x| x.bits()).collect::<Vec<_>>(), before);
    input.fill(T::from(T::ONE));
    entry(input, output).unwrap();
}

fn measure<T: Format, const N: usize>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    label: &str,
    uf: Uf,
    range: Range,
    mut entry: impl FnMut(&[T], &mut [T]) -> Result<(), PcuExecutionFault>,
) {
    let mut input = vec![T::from(T::ONE); N];
    let mut output = vec![T::from(T::ONE + 1); N + 3];
    verify::<T, N>(&mut entry, &mut input, &mut output, range);
    let scores = SCORES.load(Ordering::Relaxed);
    let counters = ffi::count_heap(|| {
        for phase in 0..64 {
            for (lane, value) in input.iter_mut().enumerate() {
                *value = T::from(
                    T::ONE
                        | if (lane + phase).is_multiple_of(2) {
                            0
                        } else {
                            T::SIGN
                        },
                );
            }
            entry(&input, &mut output).unwrap();
            assert!(
                output[..N]
                    .iter()
                    .zip(input.iter())
                    .all(|(a, b)| a.bits() == b.bits())
            );
        }
    });
    assert_eq!(
        (counters.allocations, counters.reallocations, counters.frees),
        (0, 0, 0)
    );
    assert_eq!(SCORES.load(Ordering::Relaxed), scores);
    println!(
        "vulkan_one_effect {:?}/{uf:?}/{range:?}/{N}/{label}:64 changing calls, zero warm Rust heap/rescore",
        T::TYPE
    );
    let mut phase = 0_usize;
    group.bench_function(label, |bench| {
        bench.iter(|| {
            phase = phase.wrapping_add(1);
            input[0] = T::from(T::ONE | if phase.is_multiple_of(2) { 0 } else { T::SIGN });
            entry(black_box(&input), black_box(&mut output)).unwrap();
            black_box(&output);
        });
    });
}

pub fn width<T: Format, const N: usize>(
    criterion: &mut Criterion,
    backend: &PcuVulkanBackend,
    identity: PcuStableDeviceIdentity,
) {
    for (uf, policy) in [
        (Uf::IeeeAfterRounding, 0),
        (Uf::RejectSubnormalResult, 1),
        (Uf::AllowGradualUnderflow, 2),
    ] {
        for range in [Range::Reject, Range::Clamp] {
            global::configure(global::PcuExecutionPolicy {
                backend: global::PcuBackendChoice::Vulkan,
                numerical_mode: PcuNumericalMode::Strict,
                float_underflow: uf,
                range_policy: range,
                score_invocation: Some(super::score),
                ..Default::default()
            })
            .unwrap();
            let requirements = PcuImplementationRequirements {
                numerical_mode: PcuNumericalMode::Strict,
                float_underflow: uf,
                range_policy: range,
                ..Default::default()
            };
            let bindings = overwritten_add_bindings::<T>();
            let mut prepared = __overwritten_add_ir_with_float_underflow_policy::<T, N>(
                &bindings,
                uf,
                range,
                requirements,
            )
            .unwrap()
            .with_ir(|kernel| {
                assert_policy(kernel, requirements);
                backend.prepare_host_kernel(kernel).unwrap()
            });
            expect(requirements, 1);
            let mut graph = graph::prepare::<T, N, _>(backend, requirements);
            let mut native = ffi::NativeComposed::one_effect(
                identity,
                u32::try_from(N).unwrap(),
                policy,
                u32::from(range == Range::Clamp),
                T::TYPE,
            )
            .unwrap();
            native.assert_policy(T::TYPE, uf, range);
            let mut group = criterion.benchmark_group(format!(
                "vulkan_one_effect/{:?}/{uf:?}/{range:?}/{N}",
                T::TYPE
            ));
            measure::<T, N>(
                &mut group,
                "source_prepared",
                uf,
                range,
                move |input, output| {
                    prepared
                        .call(&mut [
                            PcuHostArgument::read(PcuBindingRef::new(0, 0), input),
                            PcuHostArgument::read_write(PcuBindingRef::new(0, 1), output),
                        ])
                        .map_err(fault)
                },
            );
            measure::<T, N>(&mut group, "source_ordinary", uf, range, |input, output| {
                overwritten_add::<T, N>(input, output)
                    .map_err(|error| error.arithmetic_fault().unwrap())
            });
            measure::<T, N>(
                &mut group,
                "independent_graph",
                uf,
                range,
                move |input, output| {
                    graph
                        .call(&mut [
                            PcuHostArgument::read(PcuBindingRef::new(0, 0), input),
                            PcuHostArgument::read_write(PcuBindingRef::new(0, 1), output),
                        ])
                        .map_err(fault)
                },
            );
            measure::<T, N>(
                &mut group,
                "native_checked",
                uf,
                range,
                move |input, output| {
                    native
                        .call(ffi::bytes(input), ffi::bytes_mut(output))
                        .unwrap()
                        .map_or(Ok(()), Err)
                },
            );
            clear_expected();
            group.finish();
        }
    }
}
