//! Source, explicit graph and handwritten packed shader preserve exact publication laws.
use criterion::Criterion;
use pcu_facade::dialect::tensor::{Graph, Tensor, TensorElement, TensorError};
#[rustfmt::skip]
use pcu_facade::{global, PcuScalar, PcuExecutionFault, PcuNumericalOptions,
    PcuNumericalMode, PcuCompoundArithmeticPolicy, PcuPrecisionPolicy,
    PcuFloatUnderflowPolicy, PcuStableDeviceIdentity, PcuF16Bits, PcuBf16Bits,
    PcuF8E4M3FnBits, PcuF8E5M2Bits};
#[rustfmt::skip]
use fusion_pcu_vulkan::{PcuVulkanBackend, PcuVulkanPreparedTensorGraph,
    PcuVulkanTensorInput, PcuVulkanTensorError};
use super::{
    ffi, oracle,
    source::{self, ProducerSource},
};
type Status = Result<Option<(u32, u32, bool)>, ()>;
fn fault(fault: PcuExecutionFault) -> (u32, u32, bool) {
    (
        u32::try_from(fault.invocation_id).unwrap(),
        oracle::code(fault.kind),
        fault.recovered,
    )
}
fn graph_fault(error: &PcuVulkanTensorError) -> Status {
    match error {
        PcuVulkanTensorError::Graph(TensorError::ArithmeticFault {
            element_index,
            kind,
            ..
        }) => Ok(Some((
            u32::try_from(*element_index).unwrap(),
            oracle::code(*kind),
            false,
        ))),
        _ => Err(()),
    }
}
fn check<T: PcuScalar>(actual: &[T], expected: &[T], sentinel: T) {
    oracle::bits(&actual[..expected.len()], expected);
    oracle::bits(&actual[expected.len()..], &[sentinel; 2]);
}
#[allow(clippy::too_many_lines)] // Three exact routes share the same fault, publication and ownership witnesses.
fn case<T: ProducerSource + TensorElement, const N: usize, const HALF: bool>(
    backend: &PcuVulkanBackend,
    identity: PcuStableDeviceIdentity,
    options: PcuNumericalOptions,
    mode: PcuNumericalMode,
    policy: PcuFloatUnderflowPolicy,
) {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        numerical_mode: mode,
        numerical_options: options,
        float_underflow: policy,
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
    let constant: Vec<T> = (0..N).map(oracle::payload).collect();
    let uniform = vec![oracle::factor::<T>(HALF); N];
    let banks = [
        vec![oracle::small::<T>(1); N],
        vec![oracle::small::<T>(2); N],
    ];
    let expected = banks
        .each_ref()
        .map(|input| oracle::pipeline(input, &constant, &uniform, policy).unwrap());
    let sentinel = T::sentinel();
    let mut observed = vec![sentinel; N + 2];
    let captured = T::capture::<N, HALF>(mode, options, policy).unwrap();
    assert_eq!(captured.argument_indices(), [0]);
    PcuVulkanPreparedTensorGraph::<T>::assess(
        captured.program().graph(),
        captured.program().output_values(),
    )
    .unwrap();
    let mut graph = Graph::default();
    graph.set_numerical_options(options);
    graph.set_numerical_mode(mode);
    let input = graph.input([N], T::TYPE).unwrap();
    let c = graph.constant_typed(Tensor::new([N], constant.clone()).unwrap());
    let u = graph.uniform_typed([N], uniform[0]).unwrap();
    let sum = graph.add(input, c.erase()).unwrap();
    let output = graph.mul(sum, u.erase()).unwrap();
    for value in [sum, output] {
        graph
            .set_value_float_underflow_policy(value, policy)
            .unwrap();
    }
    let mut plan = PcuVulkanPreparedTensorGraph::<T>::prepare(backend, &graph, &[output]).unwrap();
    drop(graph);
    let mut native = ffi::NativeComposed::new(
        identity,
        u32::try_from(N).unwrap(),
        match policy {
            PcuFloatUnderflowPolicy::IeeeAfterRounding => 0,
            PcuFloatUnderflowPolicy::RejectSubnormalResult => 1,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow => 2,
        },
        0,
        T::TYPE,
    )
    .unwrap();
    native.assert_policy(T::TYPE, policy, pcu_facade::PcuRangePolicy::Reject);
    // Fresh selected source outputs never alias the cached immutable producer storage.
    let mut literal = T::literal::<N>(&[]).unwrap();
    literal.read_into(&mut observed).unwrap();
    check(&observed, &constant, sentinel);
    source::overwrite::<T, N>(&[oracle::small::<T>(3)], &mut literal).unwrap();
    let consumed = source::consume(literal).unwrap();
    consumed.read_into(&mut observed).unwrap();
    check(&observed, &vec![oracle::small::<T>(3); N], sentinel);
    let replay = T::literal::<N>(&[]).unwrap();
    replay.read_into(&mut observed).unwrap();
    check(&observed, &constant, sentinel);
    T::uniform::<HALF>(&banks[0])
        .unwrap()
        .read_into(&mut observed)
        .unwrap();
    check(&observed, &uniform, sentinel);
    global::clear_thread_cache().unwrap();
    consumed.read_into(&mut observed).unwrap();
    check(&observed, &vec![oracle::small::<T>(3); N], sentinel);
    drop(consumed);
    drop(replay);
    // A detached selected graph literal keeps its own output alive after graph/plan drop.
    let mut literal_graph = Graph::default();
    literal_graph.set_numerical_options(options);
    let selected = literal_graph.constant_typed(Tensor::new([N], constant.clone()).unwrap());
    let mut literal_plan =
        PcuVulkanPreparedTensorGraph::<T>::prepare(backend, &literal_graph, &[selected.erase()])
            .unwrap();
    drop(literal_graph);
    let escaped = literal_plan.execute_owned(&[]).unwrap();
    let again = literal_plan.execute_owned(&[]).unwrap();
    escaped.read_into(&mut observed).unwrap();
    check(&observed, &constant, sentinel);
    drop(literal_plan);
    again.read_into(&mut observed).unwrap();
    check(&observed, &constant, sentinel);
    drop(escaped);
    drop(again);
    // Provider preflights are exercised directly, independently of the shared route guard.
    assert!(T::pipeline::<N, HALF>(&banks[0][..N - 1]).is_err());
    assert!(
        plan.execute_owned(&[PcuVulkanTensorInput::Host(&banks[0][..N - 1])])
            .is_err()
    );
    let previous = observed.clone();
    assert!(
        native
            .call(
                ffi::bytes(&banks[0][..N - 1]),
                ffi::bytes(&constant),
                ffi::bytes(&uniform),
                ffi::bytes_mut(&mut observed)
            )
            .is_err()
    );
    assert!(
        native
            .call(
                ffi::bytes(&banks[0]),
                ffi::bytes(&constant),
                ffi::bytes(&uniform),
                ffi::bytes_mut(&mut observed[..N - 1])
            )
            .is_err()
    );
    oracle::bits(&observed, &previous);
    let owner = plan
        .execute_owned(&[PcuVulkanTensorInput::Host(&banks[0])])
        .unwrap();
    assert!(owner.read_into(&mut observed[..N - 1]).is_err());
    oracle::bits(&observed, &previous);
    drop(owner);
    for route in 0..3 {
        let mut call = |input: &[T], out: &mut [T]| -> Status {
            if input.len() < N || out.len() < N {
                return Err(());
            }
            match route {
                0 => match T::pipeline::<N, HALF>(input) {
                    Ok(owner) => {
                        owner.read_into(out).map_err(|_| ())?;
                        Ok(None)
                    }
                    Err(error) => error
                        .arithmetic_fault()
                        .map_or(Err(()), |value| Ok(Some(fault(value)))),
                },
                1 => match plan.execute_owned(&[PcuVulkanTensorInput::Host(input)]) {
                    Ok(owner) => {
                        owner.read_into(out).map_err(|_| ())?;
                        Ok(None)
                    }
                    Err(error) => graph_fault(&error),
                },
                _ => native
                    .call(
                        ffi::bytes(input),
                        ffi::bytes(&constant),
                        ffi::bytes(&uniform),
                        ffi::bytes_mut(out),
                    )
                    .map_err(|_| ())
                    .map(|status| status.map(fault)),
            }
        };
        for phase in 0..2 {
            assert_eq!(call(&banks[phase], &mut observed), Ok(None));
            check(&observed, &expected[phase], sentinel);
        }
        let mut bad = banks[0].clone();
        bad[7] = T::from(T::MAX + 1);
        let mut fixtures = vec![bad.clone()];
        bad[3] = T::from(T::MAX + 1);
        fixtures.push(bad.clone());
        bad[3] = T::from(T::MAX);
        fixtures.push(bad.clone());
        bad[7] = T::from(T::MAX);
        fixtures.push(bad);
        let mut tiny = banks[0].clone();
        tiny[1] = T::zero();
        fixtures.push(tiny.clone());
        tiny[2] = T::from(1 << T::FRACTION);
        fixtures.push(tiny);
        let mut tiny = banks[0].clone();
        tiny[2] = T::from(1 << T::FRACTION);
        fixtures.push(tiny);
        for values in fixtures {
            let previous = observed.clone();
            let want = oracle::pipeline(&values, &constant, &uniform, policy);
            assert_eq!(
                call(&values, &mut observed),
                Ok(want.as_ref().err().copied())
            );
            if let Ok(want) = want {
                check(&observed, &want, sentinel);
            } else {
                oracle::bits(&observed, &previous);
            }
            assert_eq!(call(&banks[1], &mut observed), Ok(None));
            check(&observed, &expected[1], sentinel);
        }
        let previous = observed.clone();
        assert_eq!(call(&banks[0][..N - 1], &mut observed), Err(()));
        assert_eq!(call(&banks[0], &mut observed[..N - 1]), Err(()));
        oracle::bits(&observed, &previous);
        assert_eq!(call(&banks[0], &mut observed), Ok(None));
        check(&observed, &expected[0], sentinel);
        let name = format!(
            "{mode:?}/{:?}/{:?}/{policy:?}/half={HALF}/{}/{N}/{route}",
            options.compound_arithmetic,
            options.precision,
            T::LABEL
        );
        println!(
            "vulkan-low-producer-semantic/{name}: exact source/graph/independent bits/fault phases/rollback/retry/escaped owner/tails PASS"
        );
        #[cfg(feature = "insights")]
        {
            let mut phase = 0;
            super::census::warm(&name, route, || {
                let heap = ffi::count_heap(|| {
                    for _ in 0..64 {
                        phase ^= 1;
                        assert_eq!(call(&banks[phase], &mut observed), Ok(None));
                        check(&observed, &expected[phase], sentinel);
                    }
                });
                if route == 2 {
                    assert_eq!(
                        (heap.allocations, heap.reallocations, heap.frees),
                        (0, 0, 0)
                    );
                }
                println!("vulkan-low-producer-heap/{name}:64-changing-calls {heap:?}");
            });
        }
    }
}
fn matrix(backend: &PcuVulkanBackend, identity: PcuStableDeviceIdentity) {
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
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
                    PcuFloatUnderflowPolicy::RejectSubnormalResult,
                    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                ] {
                    let options = PcuNumericalOptions {
                        compound_arithmetic,
                        precision,
                        ..Default::default()
                    };
                    macro_rules! width {
                        ($ty:ty) => {
                            case::<$ty, 65, false>(backend, identity, options, mode, policy);
                            case::<$ty, 65, true>(backend, identity, options, mode, policy);
                            case::<$ty, 4096, false>(backend, identity, options, mode, policy);
                            case::<$ty, 4096, true>(backend, identity, options, mode, policy);
                        };
                    }
                    width!(PcuF16Bits);
                    width!(PcuBf16Bits);
                    width!(PcuF8E4M3FnBits);
                    width!(PcuF8E5M2Bits);
                }
            }
        }
    }
}
pub fn run(_: &mut Criterion) {
    assert!(
        std::env::args().any(|arg| arg == "--test"),
        "correctness-only target; no uncontrolled timing"
    );
    #[cfg(feature = "insights")]
    super::census::run(|| {
        let (backend, identity) = super::device::selected();
        matrix(&backend, identity);
    });
    #[cfg(not(feature = "insights"))]
    {
        let (backend, identity) = super::device::selected();
        matrix(&backend, identity);
    }
}
