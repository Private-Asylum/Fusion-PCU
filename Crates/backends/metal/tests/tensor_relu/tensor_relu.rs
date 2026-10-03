//! Native ordinary six-format `ReLU` qualification; no ignored-device fallback.
#[path = "source/source.rs"]
mod source;
#[rustfmt::skip]
use pcu_facade::{
    global,
    PcuScalar,
    PcuTensor,
    PcuExecutionError,
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuCompoundArithmeticPolicy,
    PcuPrecisionPolicy,
    PcuReproducibility,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
};
trait Sample: PcuScalar {
    const SIGN: u64;
    const ONE: u64;
    const NORMAL: u64;
    const NAN: u64;
    fn raw(bits: u64) -> Self;
}
macro_rules! samples {($($ty:ty,$sign:expr,$one:expr,$normal:expr,$nan:expr;)+)=>{$(
    impl Sample for $ty {
        const SIGN:u64=$sign;const ONE:u64=$one;const NORMAL:u64=$normal;const NAN:u64=$nan;
        fn raw(bits:u64)->Self{Self::from_bits(bits.try_into().unwrap())}
    }
)+};}
samples! {PcuF16Bits,0x8000,0x3c00,0x0400,0x7e01;PcuBf16Bits,0x8000,0x3f80,0x0080,0x7fc1;PcuF8E4M3FnBits,0x80,0x38,0x08,0x7f;PcuF8E5M2Bits,0x80,0x3c,0x04,0x7f;f32,0x8000_0000,0x3f80_0000,0x0080_0000,0x7fc1_2345;}
impl Sample for f64 {
    const SIGN: u64 = 0x8000_0000_0000_0000;
    const ONE: u64 = 0x3ff0_0000_0000_0000;
    const NORMAL: u64 = 0x0010_0000_0000_0000;
    const NAN: u64 = 0x7ff8_1234_5678_9abc;
    fn raw(bits: u64) -> Self {
        Self::from_bits(bits)
    }
}
fn equal<T: PcuScalar>(a: &[T], b: &[T]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert_eq!(a.encode_le().as_ref(), b.encode_le().as_ref());
    }
}
fn read<T: Sample>(owner: &PcuTensor<T>, expected: &[T]) {
    assert_eq!(owner.shape(), &[expected.len()]);
    let mut actual = vec![T::raw(T::ONE); expected.len() + 3];
    owner.read_into(&mut actual).unwrap();
    equal(&actual[..expected.len()], expected);
    equal(&actual[expected.len()..], &[T::raw(T::ONE); 3]);
    let mut short = [T::raw(T::ONE); 1];
    assert!(owner.read_into(&mut short).is_err());
    equal(&short, &[T::raw(T::ONE)]);
}
fn fault<T: PcuScalar>(
    result: Result<PcuTensor<T>, PcuExecutionError>,
    kind: PcuExecutionFaultKind,
) {
    let error = result.unwrap_err();
    let fault = error.arithmetic_fault().unwrap();
    assert_eq!(fault.kind, kind);
    assert_eq!(fault.invocation_id, 2);
    assert!(!fault.recovered);
}
fn exceptional<T: Sample>(old: &PcuTensor<T>, expected: &[T], policy: PcuFloatUnderflowPolicy) {
    let mut bad = [T::raw(T::ONE); 65];
    bad[2] = T::raw(T::NAN);
    bad[6] = T::raw(T::NAN);
    let retained = source::retain(&bad).unwrap();
    fault(
        source::relu::<T>(&retained),
        PcuExecutionFaultKind::InvalidFloatingOperand,
    );
    fault(
        source::unused::<T>(&retained),
        PcuExecutionFaultKind::InvalidFloatingOperand,
    );
    read(&retained, &bad);
    read(old, expected);
    bad[2] = T::raw(1);
    bad[6] = T::raw(T::ONE);
    let tiny = source::relu(&bad);
    if policy == PcuFloatUnderflowPolicy::RejectSubnormalResult {
        fault(tiny, PcuExecutionFaultKind::ArithmeticUnderflow);
    } else {
        read(&tiny.unwrap(), &bad);
    }
    read(&source::allow(&bad).unwrap(), &bad);
    fault(
        source::tight(&bad),
        PcuExecutionFaultKind::ArithmeticUnderflow,
    );
    read(
        &source::relu(&[T::raw(T::ONE); 65]).unwrap(),
        &[T::raw(T::ONE); 65],
    );
}
fn case<T: Sample>(policy: PcuFloatUnderflowPolicy) {
    for phase in 0..3 {
        let codes = [T::SIGN | T::ONE, T::SIGN, 0, T::ONE, T::NORMAL, T::SIGN | 1];
        let mut input = (0..65)
            .map(|i| T::raw(codes[(i + phase) % 6]))
            .collect::<Vec<_>>();
        let expected = (0..65)
            .map(|i| {
                let code = codes[(i + phase) % 6];
                T::raw(if code & T::SIGN != 0 { 0 } else { code })
            })
            .collect::<Vec<_>>();
        let original = input.clone();
        let retained = source::retain(&input).unwrap();
        let host = source::relu(&input).unwrap();
        let resident = source::relu::<T>(&retained).unwrap();
        let unused = source::unused::<T>(&retained).unwrap();
        input.fill(T::raw(T::ONE));
        read(&host, &expected);
        read(&resident, &expected);
        read(&unused, &original);
        let consumed = source::consume(retained).unwrap();
        read(&consumed, &expected);
        exceptional(&host, &expected, policy);
        global::clear_thread_cache().unwrap();
        drop(resident);
        drop(unused);
        drop(consumed);
        read(&host, &expected);
    }
}
fn width<T: Sample>() {
    for numerical_mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for compound_arithmetic in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for precision in [
                PcuPrecisionPolicy::Preserve,
                PcuPrecisionPolicy::BackendOptimized,
            ] {
                for float_underflow in [
                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    PcuFloatUnderflowPolicy::RejectSubnormalResult,
                    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                ] {
                    global::configure(global::PcuExecutionPolicy {
                        backend: global::PcuBackendChoice::Metal,
                        numerical_mode,
                        float_underflow,
                        numerical_options: PcuNumericalOptions {
                            compound_arithmetic,
                            precision,
                            reproducibility: PcuReproducibility::Unspecified,
                        },
                        ..Default::default()
                    })
                    .unwrap();
                    case::<T>(float_underflow);
                }
            }
        }
    }
}
fn changed_hint_uses_authentic_session() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Metal,
        block_size: 32,
        ..Default::default()
    })
    .unwrap();
    let input = [-1.0_f64, -0.0, 1.0];
    let retained = source::retain(&input).unwrap();
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Metal,
        block_size: 64,
        ..Default::default()
    })
    .unwrap();
    let output = source::relu::<f64>(&retained).unwrap();
    drop(retained);
    global::clear_thread_cache().unwrap();
    let sibling = source::relu::<f64>(&output).unwrap();
    drop(output);
    read(&sibling, &[0.0, 0.0, 1.0]);
}
#[test]
#[ignore = "Requires actual Metal ordinary six-format owned ReLU, effects and lifetimes."]
fn ordinary_six_format_relu_effects_and_escaped_owners() {
    width::<PcuF16Bits>();
    width::<PcuBf16Bits>();
    width::<PcuF8E4M3FnBits>();
    width::<PcuF8E5M2Bits>();
    width::<f32>();
    width::<f64>();
    changed_hint_uses_authentic_session();
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
