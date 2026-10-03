//! Exact independent unary encoding proof and source publication on the actual provider.
extern crate pcu_facade as fusion_pcu;
#[path = "../../benches/low_unary/oracle/oracle.rs"]
#[allow(dead_code)]
mod oracle;
#[path = "../../benches/low_unary/source/source.rs"]
#[allow(dead_code)]
mod source;
#[rustfmt::skip]
use fusion_pcu::{global,PcuExecutionError,PcuExecutionFaultKind,PcuFloatUnderflowPolicy,PcuRangePolicy,PcuTensor,PcuBindingRef,PcuHostArgument,PcuHostKernelBackend,PcuPreparedHostKernel,PcuF16Bits,PcuBf16Bits,PcuF8E4M3FnBits,PcuF8E5M2Bits};
use oracle::Format;
#[path = "../../benches/strict_matmul/selection.rs"]
mod selection;
fn selected() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Rocm,
        device: Some(0),
        block_size: 256,
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
}
#[allow(clippy::needless_pass_by_value)] // Classify one terminal execution result.
fn fault(
    result: Result<(), PcuExecutionError>,
    index: u64,
    kind: PcuExecutionFaultKind,
    recovered: bool,
) {
    assert!(
        matches!(&result,Err(PcuExecutionError::ArithmeticFault(f)) if f.invocation_id==index&&f.kind==kind&&f.recovered==recovered),
        "unexpected {result:?}"
    );
}
fn host<T: Format, const N: usize>(
    input: &[T],
    output: &mut [T],
    op: u32,
    policy: PcuFloatUnderflowPolicy,
    range: PcuRangePolicy,
) -> Result<(), PcuExecutionError> {
    match (op, range, policy) {
        (0, PcuRangePolicy::Reject, PcuFloatUnderflowPolicy::IeeeAfterRounding) => {
            source::neg_reject_ieee::<T, N>(input, output)
        }
        (0, PcuRangePolicy::Reject, PcuFloatUnderflowPolicy::AllowGradualUnderflow) => {
            source::neg_reject_gradual::<T, N>(input, output)
        }
        (0, PcuRangePolicy::Reject, PcuFloatUnderflowPolicy::RejectSubnormalResult) => {
            source::neg_reject_strict::<T, N>(input, output)
        }
        (0, PcuRangePolicy::Clamp, PcuFloatUnderflowPolicy::IeeeAfterRounding) => {
            source::neg_clamp_ieee::<T, N>(input, output)
        }
        (0, PcuRangePolicy::Clamp, PcuFloatUnderflowPolicy::AllowGradualUnderflow) => {
            source::neg_clamp_gradual::<T, N>(input, output)
        }
        (0, PcuRangePolicy::Clamp, PcuFloatUnderflowPolicy::RejectSubnormalResult) => {
            source::neg_clamp_strict::<T, N>(input, output)
        }
        (1, PcuRangePolicy::Reject, PcuFloatUnderflowPolicy::IeeeAfterRounding) => {
            source::relu_reject_ieee::<T, N>(input, output)
        }
        (1, PcuRangePolicy::Reject, PcuFloatUnderflowPolicy::AllowGradualUnderflow) => {
            source::relu_reject_gradual::<T, N>(input, output)
        }
        (1, PcuRangePolicy::Reject, PcuFloatUnderflowPolicy::RejectSubnormalResult) => {
            source::relu_reject_strict::<T, N>(input, output)
        }
        (1, PcuRangePolicy::Clamp, PcuFloatUnderflowPolicy::IeeeAfterRounding) => {
            source::relu_clamp_ieee::<T, N>(input, output)
        }
        (1, PcuRangePolicy::Clamp, PcuFloatUnderflowPolicy::AllowGradualUnderflow) => {
            source::relu_clamp_gradual::<T, N>(input, output)
        }
        (1, PcuRangePolicy::Clamp, PcuFloatUnderflowPolicy::RejectSubnormalResult) => {
            source::relu_clamp_strict::<T, N>(input, output)
        }
        _ => unreachable!(),
    }
}
fn resident<T: Format, const N: usize>(
    input: &PcuTensor<T>,
    output: &mut PcuTensor<T>,
    op: u32,
    policy: PcuFloatUnderflowPolicy,
    range: PcuRangePolicy,
) -> Result<(), PcuExecutionError> {
    match (op, range, policy) {
        (0, PcuRangePolicy::Reject, PcuFloatUnderflowPolicy::IeeeAfterRounding) => {
            source::neg_reject_ieee::<T, N>(input, output)
        }
        (0, PcuRangePolicy::Reject, PcuFloatUnderflowPolicy::AllowGradualUnderflow) => {
            source::neg_reject_gradual::<T, N>(input, output)
        }
        (0, PcuRangePolicy::Reject, PcuFloatUnderflowPolicy::RejectSubnormalResult) => {
            source::neg_reject_strict::<T, N>(input, output)
        }
        (0, PcuRangePolicy::Clamp, PcuFloatUnderflowPolicy::IeeeAfterRounding) => {
            source::neg_clamp_ieee::<T, N>(input, output)
        }
        (0, PcuRangePolicy::Clamp, PcuFloatUnderflowPolicy::AllowGradualUnderflow) => {
            source::neg_clamp_gradual::<T, N>(input, output)
        }
        (0, PcuRangePolicy::Clamp, PcuFloatUnderflowPolicy::RejectSubnormalResult) => {
            source::neg_clamp_strict::<T, N>(input, output)
        }
        (1, PcuRangePolicy::Reject, PcuFloatUnderflowPolicy::IeeeAfterRounding) => {
            source::relu_reject_ieee::<T, N>(input, output)
        }
        (1, PcuRangePolicy::Reject, PcuFloatUnderflowPolicy::AllowGradualUnderflow) => {
            source::relu_reject_gradual::<T, N>(input, output)
        }
        (1, PcuRangePolicy::Reject, PcuFloatUnderflowPolicy::RejectSubnormalResult) => {
            source::relu_reject_strict::<T, N>(input, output)
        }
        (1, PcuRangePolicy::Clamp, PcuFloatUnderflowPolicy::IeeeAfterRounding) => {
            source::relu_clamp_ieee::<T, N>(input, output)
        }
        (1, PcuRangePolicy::Clamp, PcuFloatUnderflowPolicy::AllowGradualUnderflow) => {
            source::relu_clamp_gradual::<T, N>(input, output)
        }
        (1, PcuRangePolicy::Clamp, PcuFloatUnderflowPolicy::RejectSubnormalResult) => {
            source::relu_clamp_strict::<T, N>(input, output)
        }
        _ => unreachable!(),
    }
}

#[allow(clippy::too_many_lines)] // One format's complete source/prepared and publication law remains together.
fn complete<T: Format>() {
    const N: usize = 65536;
    const M: usize = 65;
    // Explicit full-width enumeration includes both signs; FP8 repeats its entire byte space.
    let input = (0..N)
        .map(|i| {
            let bits = u16::try_from(i % (usize::from(T::SIGN) * 2)).unwrap();
            T::from(if bits & (T::SIGN - 1) > T::MAX {
                0
            } else {
                bits
            })
        })
        .collect::<Vec<_>>();
    for op in [0, 1] {
        for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
            for policy in [
                PcuFloatUnderflowPolicy::IeeeAfterRounding,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                PcuFloatUnderflowPolicy::RejectSubnormalResult,
            ] {
                let expected = input
                    .iter()
                    .copied()
                    .map(|v| {
                        let (bits, underflow) = oracle::evaluate(v, op, policy).unwrap();
                        let reference = if op == 0 {
                            v.pcu_clamped_neg_with_policy(policy)
                        } else {
                            v.pcu_clamped_relu_with_policy(policy)
                        };
                        match reference {
                            Ok(value) => {
                                assert!(!underflow);
                                assert_eq!(value, bits);
                            }
                            Err(fusion_pcu::PcuClampedError::Range(f)) => {
                                assert!(underflow);
                                assert_eq!(f.kind(), PcuExecutionFaultKind::ArithmeticUnderflow);
                                assert_eq!(f.clamped_value(), bits);
                            }
                            Err(fusion_pcu::PcuClampedError::Fatal(kind)) => {
                                panic!("finite reference failure {kind:?}")
                            }
                        }
                        bits
                    })
                    .collect::<Vec<_>>();
                let first = input
                    .iter()
                    .position(|v| oracle::evaluate(*v, op, policy).unwrap().1);
                let mut output = vec![T::sentinel(); N + 2];
                let result = host::<T, N>(&input, &mut output, op, policy, range);
                if let Some(index) = first {
                    fault(
                        result,
                        u64::try_from(index).unwrap(),
                        PcuExecutionFaultKind::ArithmeticUnderflow,
                        range == PcuRangePolicy::Clamp,
                    );
                    if range == PcuRangePolicy::Reject {
                        assert!(output.iter().all(|v| *v == T::sentinel()));
                        continue;
                    }
                } else {
                    result.unwrap();
                }
                oracle::verify(&expected, &output);
            }
        }
    }
    // Every nonfinite encoding independently executes through a cold-prepared unary graph.
    let (_, backend, _) = selection::selected_device();
    let bindings = source::neg_reject_ieee_bindings::<T>();
    let builder = source::neg_reject_ieee_ir::<T, 1>(&bindings).unwrap();
    for operation in [
        fusion_pcu::PcuDispatchFloatUnaryOp::Neg,
        fusion_pcu::PcuDispatchFloatUnaryOp::Relu,
    ] {
        let mut ir = builder.ir();
        let mut ops = ir.ops.to_vec();
        for instruction in &mut ops {
            if let fusion_pcu::PcuDispatchOp::Data(
                fusion_pcu::PcuDispatchDataOp::CheckedFloatUnary { op, .. },
            ) = instruction
            {
                *op = operation;
            }
        }
        ir.ops = &ops;
        let mut prepared = backend.prepare_host_kernel(&ir).unwrap();
        for bits in 0..u32::from(T::SIGN) * 2 {
            let bits = u16::try_from(bits).unwrap();
            if bits & (T::SIGN - 1) <= T::MAX {
                continue;
            }
            let input = [T::from(bits)];
            let mut output = [T::sentinel(); 3];
            fault(
                prepared
                    .call(&mut [
                        PcuHostArgument::read(PcuBindingRef::new(0, 0), &input),
                        PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output),
                    ])
                    .map_err(|error| match error {
                        fusion_pcu_rocm::RocmHostKernelError::CheckedExecutionFault(f) => {
                            PcuExecutionError::ArithmeticFault(f)
                        }
                        other => panic!("unexpected preflight {other:?}"),
                    }),
                0,
                PcuExecutionFaultKind::InvalidFloatingOperand,
                false,
            );
            assert!(output.iter().all(|v| *v == T::sentinel()));
        }
    }
    let mut input = vec![T::one(); M];
    let mut output = vec![T::sentinel(); M + 2];
    input.fill(T::from(T::SIGN | 1));
    source::relu_reject_strict::<T, M>(&input, &mut output).unwrap();
    oracle::verify(&[T::zero(); M], &output);
    input[5] = T::from(T::SIGN | (T::MAX + 1));
    output.fill(T::sentinel());
    fault(
        source::relu_clamp_strict::<T, M>(&input, &mut output),
        5,
        PcuExecutionFaultKind::InvalidFloatingOperand,
        false,
    );
    assert!(output.iter().all(|v| *v == T::sentinel()));
    input.fill(T::one());
    input[5] = T::from(1);
    input[41] = T::from(2);
    fault(
        source::grid_neg::<T, M>(&input, &mut output),
        5,
        PcuExecutionFaultKind::ArithmeticUnderflow,
        true,
    );
    let expected = input
        .iter()
        .copied()
        .map(|v| {
            oracle::evaluate(v, 0, PcuFloatUnderflowPolicy::RejectSubnormalResult)
                .unwrap()
                .0
        })
        .collect::<Vec<_>>();
    oracle::verify(&expected, &output);
    output.fill(T::sentinel());
    fault(
        source::broadcast_relu::<T, M>(&T::from(1), &mut output),
        0,
        PcuExecutionFaultKind::ArithmeticUnderflow,
        true,
    );
    oracle::verify(&[T::from(1); M], &output);
    let tiny = source::identity(input.as_slice()).unwrap();
    let mut recovered = source::identity(output.as_slice()).unwrap();
    fault(
        source::neg_clamp_strict::<T, M>(&tiny, &mut recovered),
        5,
        PcuExecutionFaultKind::ArithmeticUnderflow,
        true,
    );
    recovered.read_into(&mut output).unwrap();
    oracle::verify(&expected, &output);
    let mut permissions = vec![T::sentinel(); M + 2];
    source::permitted_neg::<T, M>(&input, &mut permissions).unwrap();
    oracle::verify(&expected, &permissions);
    input[41] = T::from(T::MAX + 1);
    output.fill(T::sentinel());
    fault(
        source::grid_neg::<T, M>(&input, &mut output),
        41,
        PcuExecutionFaultKind::InvalidFloatingOperand,
        false,
    );
    assert!(output.iter().all(|v| *v == T::sentinel()));
    let resident_input = source::identity(input.as_slice()).unwrap();
    let mut resident_output = source::identity(output.as_slice()).unwrap();
    fault(
        resident::<T, M>(
            &resident_input,
            &mut resident_output,
            0,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
            PcuRangePolicy::Clamp,
        ),
        41,
        PcuExecutionFaultKind::InvalidFloatingOperand,
        false,
    );
    assert!(matches!(
        resident_output.read_into(&mut output),
        Err(PcuExecutionError::Argument(
            global::PcuArgumentError::ResidentValueDiscarded
        ))
    ));
    assert!(source::identity(&resident_output).is_err());
    assert!(
        resident::<T, M>(
            &resident_input,
            &mut resident_output,
            0,
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuRangePolicy::Reject
        )
        .is_err()
    );
    input.fill(T::one());
    let good = source::identity(input.as_slice()).unwrap();
    let mut fresh = source::identity(output.as_slice()).unwrap();
    resident::<T, M>(
        &good,
        &mut fresh,
        0,
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuRangePolicy::Reject,
    )
    .unwrap();
    fresh.read_into(&mut output).unwrap();
    oracle::verify(&[T::from(T::ONE | T::SIGN); M], &output);
    let mut mixed = source::identity(output.as_slice()).unwrap();
    input[5] = T::from(T::MAX + 1);
    fault(
        source::neg_reject_ieee::<T, M>(&input, &mut mixed),
        5,
        PcuExecutionFaultKind::InvalidFloatingOperand,
        false,
    );
    assert!(mixed.read_into(&mut output).is_err());
    input.fill(T::one());
    let before = output.clone();
    assert!(
        host::<T, M>(
            &input[..M - 1],
            &mut output,
            0,
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuRangePolicy::Reject
        )
        .is_err()
    );
    assert_eq!(output, before);
    let mut ready = source::identity(before.as_slice()).unwrap();
    assert!(source::neg_reject_ieee::<T, M>(&input[..M - 1], &mut ready).is_err());
    ready.read_into(&mut output).unwrap();
    assert_eq!(output, before);
    source::portable_neg::<T, M>(&input, &mut output).unwrap();
    oracle::verify(&[T::from(T::ONE | T::SIGN); M], &output);
    global::clear_thread_cache().unwrap();
}
macro_rules! fixtures {($($name:ident,$ty:ty;)+)=>{$(#[test] #[ignore="requires authorized actual GPU; correctness only, run serially"] fn $name(){selected();complete::<$ty>();})+};}
fixtures!(half,PcuF16Bits;brain,PcuBf16Bits;e4,PcuF8E4M3FnBits;e5,PcuF8E5M2Bits;);

#[test]
fn cold_unary_admission_retains_exact_layout_and_requested_portable_profile() {
    fn profile<T: Format>() {
        let bindings = source::neg_reject_ieee_bindings::<T>();
        let builder = source::neg_reject_ieee_ir::<T, 65>(&bindings).unwrap();
        let mut ir = builder.ir();
        let code = fusion_pcu_rocm::lower_dispatch_to_hip_source(&ir).unwrap();
        assert!(code.contains("fusion_checked_low_unary"));
        assert!(code.contains(if T::SIGN == 0x80 {
            "const unsigned char*"
        } else {
            "const unsigned short*"
        }));
        assert!(!code.contains("__half"));
        assert!(!code.contains("__int128"));
        ir.numerical_requirements.numerical_options.reproducibility =
            fusion_pcu::PcuReproducibility::PortableV1;
        assert!(fusion_pcu::describe_portable_v1_unary_map(&ir).is_ok());
        assert!(fusion_pcu_rocm::lower_dispatch_to_hip_source(&ir).is_ok());
    }
    profile::<PcuF16Bits>();
    profile::<PcuBf16Bits>();
    profile::<PcuF8E4M3FnBits>();
    profile::<PcuF8E5M2Bits>();
}
