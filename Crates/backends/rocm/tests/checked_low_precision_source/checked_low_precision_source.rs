//! Actual four-format source maps, independent output oracle and checked publication.
extern crate pcu_facade as fusion_pcu;
#[path = "../../benches/checked_low_precision/oracle/oracle.rs"]
#[allow(dead_code)]
mod oracle;
#[path = "../../benches/checked_low_precision/source/source.rs"]
#[allow(dead_code)]
mod source;
#[rustfmt::skip]
use fusion_pcu::{ global, PcuExecutionError, PcuExecutionFaultKind, PcuFloatUnderflowPolicy, PcuClampedError, PcuF16Bits, PcuBf16Bits, PcuF8E4M3FnBits, PcuF8E5M2Bits };
use oracle::Format;
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
#[allow(clippy::needless_pass_by_value)] // Consume one terminal execution result for classification.
fn check_fault(
    result: Result<(), PcuExecutionError>,
    index: u64,
    kind: PcuExecutionFaultKind,
    recovered: bool,
) {
    assert!(
        matches!(&result,Err(PcuExecutionError::ArithmeticFault(f)) if f.invocation_id==index&&f.kind==kind&&f.recovered==recovered),
        "unexpected execution result: {result:?}"
    );
}
#[allow(clippy::too_many_lines)] // Exact full-output and publication contract for one format.
fn profile<T: Format>() {
    const N: usize = 65;
    macro_rules! operation {
        ($module:ident,$op:expr) => {{
            fn case<T: Format>() {
                let (mut left, mut right, expected) = oracle::inputs::<T>(N, 37, $op);
                let mut output = vec![T::sentinel(); N + 2];
                source::$module::direct::<T, N>(&left, &right, &mut output).unwrap();
                oracle::verify(&expected, &output);
                source::$module::grid::<T, N>(&left, &right, &mut output).unwrap();
                oracle::verify(&expected, &output);
                source::$module::strict::<T, N>(&left, &right, &mut output).unwrap();
                oracle::verify(&expected, &output);
                let before = output.clone();
                left[5] = T::from(T::MAX + 1);
                left[41] = T::from(T::MAX + 1);
                check_fault(
                    source::$module::grid::<T, N>(&left, &right, &mut output),
                    5,
                    PcuExecutionFaultKind::InvalidFloatingOperand,
                    false,
                );
                assert_eq!(output, before);
                let input = source::identity(left.as_slice()).unwrap();
                let divisor = source::identity(right.as_slice()).unwrap();
                let mut resident = source::identity(before.as_slice()).unwrap();
                check_fault(
                    source::$module::direct::<T, N>(&input, &divisor, &mut resident),
                    5,
                    PcuExecutionFaultKind::InvalidFloatingOperand,
                    false,
                );
                assert!(matches!(
                    resident.read_into(&mut output),
                    Err(PcuExecutionError::Argument(
                        global::PcuArgumentError::ResidentValueDiscarded
                    ))
                ));
                assert!(source::identity::<T>(&resident).is_err());
                assert!(source::$module::direct::<T, N>(&input, &divisor, &mut resident).is_err());
                left.fill(T::one());
                right.fill(T::one());
                output.fill(T::sentinel());
                source::$module::broadcast::<T, N>(&left, &T::one(), &mut output).unwrap();
                let expected = vec![
                    oracle::expected(
                        T::one(),
                        T::one(),
                        $op,
                        PcuFloatUnderflowPolicy::IeeeAfterRounding
                    )
                    .0;
                    N
                ];
                oracle::verify(&expected, &output);
            }
            case::<T>();
        }};
    }
    operation!(add, 0);
    operation!(sub, 1);
    operation!(mul, 2);
    operation!(div, 3);
    let mut left = vec![T::one(); N];
    let mut right = left.clone();
    let mut output = vec![T::sentinel(); N + 2];
    left[2] = T::from(T::MAX);
    right[2] = T::from(T::MAX);
    right[41] = T::zero();
    check_fault(
        source::div::clamp::<T, N>(&left, &right, &mut output),
        41,
        PcuExecutionFaultKind::DivideByZero,
        false,
    );
    assert!(output.iter().all(|v| *v == T::sentinel()));
    right.fill(T::one());
    left.fill(T::from(T::MAX));
    right.fill(T::from(T::ONE + (1 << T::FRACTION)));
    let input = source::identity(left.as_slice()).unwrap();
    let divisor = source::identity(right.as_slice()).unwrap();
    let mut resident = source::identity(output.as_slice()).unwrap();
    check_fault(
        source::mul::clamp::<T, N>(&input, &divisor, &mut resident),
        0,
        PcuExecutionFaultKind::ArithmeticOverflow,
        true,
    );
    resident.read_into(&mut output).unwrap();
    let expected = vec![T::from(T::MAX); N];
    oracle::verify(&expected, &output);
    assert!(source::identity::<T>(&resident).is_ok());
    // Exact subnormal allowed by IEEE, rejected by the tighter policy; tiny half-ulp
    // rounds to zero and faults under IEEE, and allow-gradual publishes it without fault.
    left.fill(T::from(1));
    right.fill(T::one());
    output.fill(T::sentinel());
    source::mul::direct::<T, N>(&left, &right, &mut output).unwrap();
    check_fault(
        source::mul::tight::<T, N>(&left, &right, &mut output),
        0,
        PcuExecutionFaultKind::ArithmeticUnderflow,
        false,
    );
    right.fill(T::from(T::ONE - (1 << T::FRACTION)));
    check_fault(
        source::mul::direct::<T, N>(&left, &right, &mut output),
        0,
        PcuExecutionFaultKind::ArithmeticUnderflow,
        false,
    );
    source::mul::allow::<T, N>(&left, &right, &mut output).unwrap();
    assert!(output[..N].iter().all(|v| v.bits() == 0));
    assert!(source::unsupported_compound::<T, N>(&left, &right, &mut output).is_err());
    let bindings = source::add::direct_bindings::<T>();
    let builder = source::add::direct_ir::<T, N>(&bindings).unwrap();
    let mut ir = builder.ir();
    ir.numerical_requirements.numerical_options.reproducibility =
        fusion_pcu::PcuReproducibility::PortableV1;
    assert!(fusion_pcu_rocm::lower_dispatch_to_hip_source(&ir).is_ok());
    left.fill(T::from(T::MAX));
    right.fill(T::from(T::ONE + (1 << T::FRACTION)));
    right[41] = T::from(T::MAX + 1);
    output.fill(T::sentinel());
    check_fault(
        source::mul::clamp::<T, N>(&left, &right, &mut output),
        41,
        PcuExecutionFaultKind::InvalidFloatingOperand,
        false,
    );
    assert!(output.iter().all(|v| *v == T::sentinel()));
    left.fill(T::one());
    right.fill(T::one());
    let input = source::identity(left.as_slice()).unwrap();
    let divisor = source::identity(right.as_slice()).unwrap();
    let mut short = source::identity(&vec![T::sentinel(); N - 1]).unwrap();
    assert!(source::mul::direct::<T, N>(&input, &divisor, &mut short).is_err());
    let mut unchanged = vec![T::zero(); N - 1];
    short.read_into(&mut unchanged).unwrap();
    assert!(unchanged.iter().all(|v| *v == T::sentinel()));
    global::clear_thread_cache().unwrap();
}
#[allow(clippy::too_many_lines)] // One exhaustive output fixture per named format and operation.
fn encoding_pairs<T: Format>() {
    const N: usize = 65536;
    macro_rules! operation {
        ($module:ident,$op:expr) => {{
            fn case<T: Format>() {
                let mut left = Vec::with_capacity(N);
                let mut right = Vec::with_capacity(N);
                for i in 0..N {
                    let a = if T::SIGN == 0x80 {
                        u16::try_from(i / 256).unwrap()
                    } else {
                        u16::try_from(i).unwrap()
                    };
                    let b = if T::SIGN == 0x80 {
                        u16::try_from(i % 256).unwrap()
                    } else {
                        u16::try_from((i.wrapping_mul(32749) + 17) % 65536).unwrap()
                    };
                    let (a, b) = (T::from(a), T::from(b));
                    let (a, b) = if matches!(
                        oracle::reference(
                            a,
                            b,
                            $op,
                            PcuFloatUnderflowPolicy::AllowGradualUnderflow
                        ),
                        Err(PcuClampedError::Fatal(_))
                    ) {
                        (T::one(), T::one())
                    } else {
                        (a, b)
                    };
                    left.push(a);
                    right.push(b);
                }
                for policy in [
                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    PcuFloatUnderflowPolicy::RejectSubnormalResult,
                    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                ] {
                    let mut expected = Vec::with_capacity(N);
                    let mut first = None;
                    for (index, (&a, &b)) in left.iter().zip(&right).enumerate() {
                        let (v, f) = oracle::expected(a, b, $op, policy);
                        expected.push(v);
                        if first.is_none() {
                            first = f.map(|kind| (u64::try_from(index).unwrap(), kind));
                        }
                    }
                    let mut output = vec![T::sentinel(); N + 2];
                    let result = match policy {
                        PcuFloatUnderflowPolicy::IeeeAfterRounding => {
                            source::$module::clamp::<T, N>(&left, &right, &mut output)
                        }
                        PcuFloatUnderflowPolicy::RejectSubnormalResult => {
                            source::$module::clamp_tight::<T, N>(&left, &right, &mut output)
                        }
                        PcuFloatUnderflowPolicy::AllowGradualUnderflow => {
                            source::$module::clamp_allow::<T, N>(&left, &right, &mut output)
                        }
                    };
                    if let Some((index, kind)) = first {
                        check_fault(result, index, kind, true);
                    } else {
                        result.unwrap();
                    }
                    oracle::verify(&expected, &output);
                }
            }
            case::<T>();
        }};
    }
    operation!(add, 0);
    operation!(sub, 1);
    operation!(mul, 2);
    operation!(div, 3);
    global::clear_thread_cache().unwrap();
}
macro_rules! fixtures {($($name:ident,$ty:ty;)+)=>{$(
    mod $name {use super::*;
        #[test] #[ignore="requires an authorized GPU; correctness only, run serially"] fn source_contract(){selected();profile::<$ty>();}
        #[test] #[ignore="requires an authorized GPU; correctness only, run serially"] fn encoding_pairs(){selected();super::encoding_pairs::<$ty>();}
    }
)+};}
fixtures!(half,PcuF16Bits;brain,PcuBf16Bits;e4,PcuF8E4M3FnBits;e5,PcuF8E5M2Bits;);

#[test]
fn cold_admission_preserves_named_layout_and_independent_requirement_identity() {
    fn profile<T: Format>() {
        let bindings = source::add::direct_bindings::<T>();
        let builder = source::add::direct_ir::<T, 65>(&bindings).unwrap();
        let mut ir = builder.ir();
        let ordinary = fusion_pcu_rocm::lower_dispatch_to_hip_source(&ir).unwrap();
        assert!(ordinary.contains(if T::SIGN == 0x80 {
            "const unsigned char*"
        } else {
            "const unsigned short*"
        }));
        assert!(ordinary.contains("fusion_checked_low_binary"));
        assert!(!ordinary.contains("__half"));
        assert!(!ordinary.contains("__int128"));
        let body = ordinary.split_once('\n').unwrap().1;
        ir.numerical_requirements.numerical_mode = fusion_pcu::PcuNumericalMode::Strict;
        ir.numerical_requirements
            .numerical_options
            .compound_arithmetic = fusion_pcu::PcuCompoundArithmeticPolicy::BackendDefined;
        ir.numerical_requirements.numerical_options.precision =
            fusion_pcu::PcuPrecisionPolicy::BackendOptimized;
        let stronger = fusion_pcu_rocm::lower_dispatch_to_hip_source(&ir).unwrap();
        assert_ne!(ordinary, stronger);
        assert_eq!(body, stronger.split_once('\n').unwrap().1);
        ir.numerical_requirements.numerical_options.reproducibility =
            fusion_pcu::PcuReproducibility::PortableV1;
        assert!(fusion_pcu_rocm::lower_dispatch_to_hip_source(&ir).is_ok());
        ir.numerical_requirements.range_policy = fusion_pcu::PcuRangePolicy::Clamp;
        assert!(fusion_pcu_rocm::lower_dispatch_to_hip_source(&ir).is_err());
        let bindings = source::unsupported_compound_bindings::<T>();
        let builder = source::unsupported_compound_ir::<T, 65>(&bindings).unwrap();
        assert!(fusion_pcu_rocm::lower_dispatch_to_hip_source(&builder.ir()).is_err());
    }
    profile::<PcuF16Bits>();
    profile::<PcuBf16Bits>();
    profile::<PcuF8E4M3FnBits>();
    profile::<PcuF8E5M2Bits>();
}
