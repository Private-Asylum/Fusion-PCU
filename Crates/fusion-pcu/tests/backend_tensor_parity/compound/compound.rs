//! Shared ordinary-source acceptance for stronger checked compound implementations.
#[rustfmt::skip]
use fusion_pcu::{
    global,
    PcuCompoundArithmeticPolicy,
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuPrecisionPolicy,
    PcuScalar,
    PcuTensor,
};
use super::contract::POLICY_LOCK;

#[path = "source/source.rs"]
mod source;

fn bits<T: PcuScalar>(actual: &[T], expected: &[T]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.encode_le().as_ref(), expected.encode_le().as_ref());
    }
}

fn read<T: PcuScalar>(owner: &PcuTensor<T>, expected: &[T], sentinel: T) {
    let mut stack = [sentinel; 9];
    owner.read_into(&mut stack).unwrap();
    bits(&stack[..expected.len()], expected);
    bits(&stack[expected.len()..], &[sentinel; 9][expected.len()..]);
}

fn format<T: PcuScalar>(
    value: impl Fn(f32) -> T + Copy,
    minimum: T,
    policy: PcuFloatUnderflowPolicy,
) {
    let sentinel = value(16.0);
    let left = [[1.0, 2.0], [3.0, 4.0]].map(|row| row.map(value));
    let identity = [[1.0, 0.0], [0.0, 1.0]].map(|row| row.map(value));
    let product = source::product(&left, &identity).unwrap();
    read(&product, &[1.0, 2.0, 3.0, 4.0].map(value), sentinel);
    let composed = source::product::<T>(&product, &identity).unwrap();
    global::clear_thread_cache().unwrap();
    drop(product);
    read(&composed, &[1.0, 2.0, 3.0, 4.0].map(value), sentinel);
    let zeros = [[value(0.0); 2]; 2];
    let loss = source::loss::<T>(&composed, &zeros).unwrap();
    assert_eq!(loss.len(), 1);
    read(&loss, &[value(7.5)], sentinel);
    let updated = source::update::<T>(&composed, &composed).unwrap();
    drop(loss);
    drop(composed);
    read(&updated, &[0.5, 1.0, 1.5, 2.0].map(value), sentinel);

    // IEEE tininess/inexactness is observed at the separately rounded Strict
    // multiply, despite the final dot product being finite and normal.
    let tiny_left = [[minimum, value(1.0)]];
    let right = [[value(0.5)], [value(1.0)]];
    match policy {
        PcuFloatUnderflowPolicy::AllowGradualUnderflow => {
            let result = source::dot(&tiny_left, &right).unwrap();
            read(&result, &[value(1.0)], sentinel);
        }
        PcuFloatUnderflowPolicy::IeeeAfterRounding
        | PcuFloatUnderflowPolicy::RejectSubnormalResult => {
            let Err(error) = source::dot(&tiny_left, &right) else {
                panic!("Strict compound must report its tiny inexact multiply");
            };
            let fault = error.arithmetic_fault().expect("compound underflow");
            assert_eq!(fault.kind, PcuExecutionFaultKind::ArithmeticUnderflow);
            assert_eq!(fault.invocation_id, 0);
            assert!(!fault.recovered);
        }
    }
    // A failed private result must preserve earlier escaped owners and permit retry.
    read(&updated, &[0.5, 1.0, 1.5, 2.0].map(value), sentinel);
    read(
        &source::dot(&[[value(1.0), value(2.0)]], &right).unwrap(),
        &[value(2.5)],
        sentinel,
    );
}

pub fn verify(backend: global::PcuBackendChoice) {
    let _guard = POLICY_LOCK.lock().unwrap();
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
                PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                PcuFloatUnderflowPolicy::RejectSubnormalResult,
            ] {
                global::configure(global::PcuExecutionPolicy {
                    backend,
                    // The helper selects Strict locally, independently of the
                    // caller's default Boundary setting and other permissions.
                    numerical_mode: PcuNumericalMode::Boundary,
                    numerical_options: PcuNumericalOptions {
                        compound_arithmetic,
                        precision,
                        ..Default::default()
                    },
                    float_underflow,
                    ..Default::default()
                })
                .unwrap();
                global::clear_thread_cache().unwrap();
                format::<f32>(|v| v, f32::from_bits(1), float_underflow);
                format::<f64>(f64::from, f64::from_bits(1), float_underflow);
            }
        }
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
