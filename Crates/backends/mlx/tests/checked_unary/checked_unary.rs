//! Real MLX-owned integer checked kernels; tensor-native `MatMul` is a separate permission scope.
#[path = "graph/graph.rs"]
mod graph;
#[path = "oracle/oracle.rs"]
mod oracle;
#[path = "prefix/prefix.rs"]
mod prefix;
#[path = "source/source.rs"]
mod source;
#[rustfmt::skip]
use fusion_pcu_mlx::{MlxRuntime,MlxError,MlxCheckedUnaryPlan};
#[rustfmt::skip]
use pcu_facade::{PcuBindingRef,PcuHostArgument,PcuHostKernelBackend,PcuPreparedHostKernel,
    PcuHostDispatchError,PcuExecutionFault,PcuExecutionFaultKind,PcuDispatchFloatUnaryOp as Op,
    PcuFloatUnderflowPolicy as Policy,PcuRangePolicy as Range,PcuReproducibility,
    PcuF16Bits,PcuBf16Bits,PcuF8E4M3FnBits,PcuF8E5M2Bits};
use oracle::Low;
fn call<T: Low>(
    prepared: &mut impl PcuPreparedHostKernel<Error = fusion_pcu_mlx::MlxHostKernelError>,
    input: &[T],
    output: &mut [T],
) -> Result<(), PcuExecutionFault> {
    prepared
        .call(&mut [
            PcuHostArgument::read_write(PcuBindingRef::new(4, 1), output),
            PcuHostArgument::read(PcuBindingRef::new(2, 3), input),
        ])
        .map_err(|error| match error {
            PcuHostDispatchError::Backend(MlxError::Arithmetic(fault)) => fault,
            other => panic!("unexpected operational/schema error {other:?}"),
        })
}
// Keep the fault, retry and both shape profiles adjacent for this lifecycle qualification.
#[allow(clippy::too_many_lines)]
fn lifecycle<T: Low>() {
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    for op in [Op::Neg, Op::Relu] {
        for policy in [
            Policy::IeeeAfterRounding,
            Policy::RejectSubnormalResult,
            Policy::AllowGradualUnderflow,
        ] {
            for range in [Range::Reject, Range::Clamp] {
                for grid in [false, true] {
                    let mut prepared = graph::fixture_profile::<T, _>(
                        3,
                        op,
                        policy,
                        range,
                        grid,
                        false,
                        |kernel| session.prepare_host_kernel(kernel),
                    )
                    .unwrap();
                    for input in [
                        [T::from_bits(1), T::from_bits(T::SIGN), T::from_bits(T::MAX)],
                        [
                            T::from_bits(T::SIGN | 1),
                            T::from_bits(1 << T::FRACTION),
                            T::from_bits(0),
                        ],
                    ] {
                        let mut output = [T::from_bits(17); 5];
                        let mut expected = output;
                        let actual = call(&mut prepared, &input, &mut output);
                        let reference =
                            oracle::native::<T, 3>(&input, &mut expected, op, policy, range);
                        assert_eq!(actual, reference);
                        assert_eq!(output, expected);
                    }
                    let input = [
                        T::from_bits(1),
                        T::from_bits(T::SIGN - 1),
                        T::from_bits(T::MAX),
                    ];
                    let mut output = [T::from_bits(17); 5];
                    let before = output;
                    let fault = call(&mut prepared, &input, &mut output).unwrap_err();
                    assert_eq!(output, before);
                    // Reject can stop at an earlier range fault; Clamp must never hide this later fatal.
                    if range == Range::Clamp {
                        assert_eq!(
                            fault,
                            PcuExecutionFault {
                                invocation_id: 1,
                                kind: PcuExecutionFaultKind::InvalidFloatingOperand,
                                recovered: false
                            }
                        );
                    }
                    call(
                        &mut prepared,
                        &[T::from_bits(1 << T::FRACTION); 3],
                        &mut output,
                    )
                    .unwrap();
                    assert_eq!(&output[3..], &before[3..]);
                    let invalid = prepared.call(&mut [
                        PcuHostArgument::read(PcuBindingRef::new(2, 3), &input[..1]),
                        PcuHostArgument::read_write(PcuBindingRef::new(4, 1), &mut output),
                    ]);
                    assert!(invalid.is_err());
                    assert!(!prepared.last_call_may_have_written());
                    assert!(!prepared.last_call_completion_uncertain());
                    let mut broadcast = graph::fixture_profile::<T, _>(
                        3,
                        op,
                        policy,
                        range,
                        grid,
                        true,
                        |kernel| session.prepare_host_kernel(kernel),
                    )
                    .unwrap();
                    for bits in [
                        0,
                        T::SIGN,
                        1,
                        T::SIGN | 1,
                        1 << T::FRACTION,
                        T::MAX,
                        T::SIGN - 1,
                    ] {
                        let input = [T::from_bits(bits)];
                        let mut output = [T::from_bits(17); 5];
                        let mut expected = output;
                        let reference = oracle::native::<T, 3>(
                            &[input[0]; 3],
                            &mut expected,
                            op,
                            policy,
                            range,
                        );
                        assert_eq!(call(&mut broadcast, &input, &mut output), reference);
                        assert_eq!(output, expected);
                    }
                }
            }
        }
    }
    source_calls::<T>(&session);
}
fn source_calls<T: Low>(session: &fusion_pcu_mlx::MlxSession) {
    let input = [
        T::from_bits(T::SIGN | 1),
        T::from_bits(0),
        T::from_bits(T::MAX),
    ];
    let mut output = [T::from_bits(17); 5];
    source::negate_prepare::<T, 3, _>(session).unwrap()(&input, &mut output).unwrap();
    source::relu_prepare::<T, 3, _>(session).unwrap()(&input, &mut output).unwrap();
    source::negate_tight_clamp_prepare::<T, 3, _>(session).unwrap()(&input, &mut output)
        .unwrap_err();
    source::relu_tight_clamp_prepare::<T, 3, _>(session).unwrap()(
        &[T::from_bits(1), input[1], input[2]],
        &mut output,
    )
    .unwrap_err();
}
macro_rules! native {($name:ident,$ty:ty)=>{
    #[test]#[ignore="Requires actual pinned MLX-owned GPU custom kernel, checked statuses and terminal host publication."]
    fn $name(){lifecycle::<$ty>();}
};}
native!(f16_checked_unary_lifecycle, PcuF16Bits);
native!(bf16_checked_unary_lifecycle, PcuBf16Bits);
native!(e4m3fn_checked_unary_lifecycle, PcuF8E4M3FnBits);
native!(e5m2_checked_unary_lifecycle, PcuF8E5M2Bits);
#[test]
fn cold_profiles_are_detached_and_portable_unary_retains_exact_tuple() {
    for range in [Range::Reject, Range::Clamp] {
        for broadcast in [false, true] {
            for grid in [false, true] {
                graph::fixture_profile::<PcuF16Bits, _>(
                    3,
                    Op::Relu,
                    Policy::RejectSubnormalResult,
                    range,
                    grid,
                    broadcast,
                    |kernel| {
                        let _plan = MlxCheckedUnaryPlan::assess(kernel).unwrap();
                        let mut unsupported = *kernel;
                        unsupported
                            .numerical_requirements
                            .numerical_options
                            .reproducibility = PcuReproducibility::PortableV1;
                        assert_eq!(
                            MlxCheckedUnaryPlan::assess(&unsupported)
                                .unwrap()
                                .requirements(),
                            unsupported.numerical_requirements
                        );
                        let mut mismatch = *kernel;
                        mismatch.numerical_requirements.float_underflow = Policy::IeeeAfterRounding;
                        assert!(MlxCheckedUnaryPlan::assess(&mismatch).is_err());
                    },
                );
            }
        }
    }
    graph::fixture::<pcu_facade::PcuF128Bits, _>(
        3,
        Op::Neg,
        Policy::IeeeAfterRounding,
        false,
        |kernel| {
            assert!(MlxCheckedUnaryPlan::assess(kernel).is_err());
        },
    );
    graph::fixture::<f64, _>(
        u32::MAX / 2,
        Op::Neg,
        Policy::IeeeAfterRounding,
        false,
        |kernel| {
            assert!(MlxCheckedUnaryPlan::assess(kernel).is_err());
        },
    );
}

#[path = "ordinary/ordinary.rs"]
mod ordinary;

#[path = "portable/portable.rs"]
mod portable;

#[path = "normal/normal.rs"]
mod normal;
