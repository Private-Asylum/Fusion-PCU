//! Ordinary conversion borrows keep source/output widths and escaped owners distinct.

#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuExecutionError,
    PcuExecutionFaultKind,
    PcuScalar,
    PcuTensor,
};
use super::bits;

#[pcu]
fn retain<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}

#[pcu(invocations = 7)]
fn narrow(input: &[f64], output: &mut [f32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] as f32;
}

#[pcu(invocations = 7)]
fn widen(input: &[f32], output: &mut [f64]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] as f64;
}

#[pcu(invocations = 3)]
fn grid(input: &[f64], output: &mut [f32]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < 7 {
        output[id] = input[id] as f32;
        id += stride;
    }
}

#[pcu(invocations = 7)]
fn broadcast(input: &f64, output: &mut [f32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = *input as f32;
}

#[pcu(invocations = 7)]
fn broadcast_element_zero(input: &[f64], output: &mut [f32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[0] as f32;
}

#[pcu(invocations = 7, flag(clamp_range))]
fn clamp(input: &[f64], output: &mut [f32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] as f32;
}

fn read<T: PcuScalar, const N: usize>(value: &PcuTensor<T>, expected: &[T; N]) {
    let mut stack = [expected[0]; N];
    value.read_into(&mut stack).unwrap();
    bits(&stack, expected);
}

pub(super) fn verify(backend: global::PcuBackendChoice) {
    // This new mixed cohort qualifies the Apple facade bridge. Other providers
    // retain their separately qualified host and owned conversion profiles.
    let apple = match backend {
        #[cfg(feature = "metal")]
        global::PcuBackendChoice::Metal => true,
        #[cfg(feature = "mlx")]
        global::PcuBackendChoice::Mlx => true,
        _ => false,
    };
    if !apple {
        return;
    }
    let input = [0.0, -0.0, 1.0, -1.5, 2.0, 4.0, 8.0];
    let expected = [0.0_f32, -0.0, 1.0, -1.5, 2.0, 4.0, 8.0];
    let resident = retain(&input).unwrap();
    let sibling = retain(&resident).unwrap();
    let mut stack = [23.0_f32; 9];
    narrow(&resident, &mut stack).unwrap();
    bits(&stack[..7], &expected);
    bits(&stack[7..], &[23.0; 2]);
    grid(&resident, &mut stack).unwrap();
    bits(&stack[..7], &expected);

    let mut output = retain(&[29.0_f32; 9]).unwrap();
    narrow(&input, &mut output).unwrap();
    read(&output, &[0.0, -0.0, 1.0, -1.5, 2.0, 4.0, 8.0, 29.0, 29.0]);
    narrow(&resident, &mut output).unwrap();
    read(&output, &[0.0, -0.0, 1.0, -1.5, 2.0, 4.0, 8.0, 29.0, 29.0]);
    // Widen consumes an ordinary borrow of a different encoded scalar owner;
    // no RAM roundtrip is required between the conversion calls.
    let narrow_owner = retain(&expected).unwrap();
    let mut widened = [37.0_f64; 9];
    widen(&narrow_owner, &mut widened).unwrap();
    bits(&widened[..7], &input);
    bits(&widened[7..], &[37.0; 2]);

    let scalar = retain(&[3.0_f64]).unwrap();
    // A rank-one owner is not a scalar merely because it contains one element.
    let scalar_before = stack;
    assert!(broadcast(&scalar, &mut stack).is_err());
    bits(&stack, &scalar_before);
    broadcast_element_zero(&scalar, &mut stack).unwrap();
    bits(&stack[..7], &[3.0; 7]);
    bits(&stack[7..], &[23.0; 2]);

    // Native capacity and the logical read span are separate. Exceptional bits
    // in unread capacity must not be checked as arithmetic operands or discarded.
    let extended = retain(&[0.0, -0.0, 1.0, -1.5, 2.0, 4.0, 8.0, f64::NAN, f64::INFINITY]).unwrap();
    narrow(&extended, &mut stack).unwrap();
    bits(&stack[..7], &expected);
    bits(&stack[7..], &[23.0; 2]);
    grid(&extended, &mut output).unwrap();
    read(&output, &[0.0, -0.0, 1.0, -1.5, 2.0, 4.0, 8.0, 29.0, 29.0]);
    broadcast_element_zero(&extended, &mut stack).unwrap();
    bits(&stack[..7], &[0.0; 7]);
    narrow(&resident, &mut stack).unwrap();
    read(
        &extended,
        &[0.0, -0.0, 1.0, -1.5, 2.0, 4.0, 8.0, f64::NAN, f64::INFINITY],
    );

    let short_before = stack;
    assert!(narrow(&resident, &mut stack[..6]).is_err());
    bits(&stack, &short_before);
    assert!(narrow(&[1.0; 6], &mut stack).is_err());
    bits(&stack, &short_before);

    let invalid = retain(&[1.0, 2.0, f64::MAX, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let output_before = [0.0_f32, -0.0, 1.0, -1.5, 2.0, 4.0, 8.0, 29.0, 29.0];
    let old_output = retain(&output).unwrap();
    let error = narrow(&invalid, &mut output).unwrap_err();
    assert!(!error.arithmetic_fault().unwrap().recovered);
    read(&output, &output_before);
    let error = clamp(&invalid, &mut output).unwrap_err();
    assert!(error.arithmetic_fault().unwrap().recovered);
    read(
        &output,
        &[1.0, 2.0, f32::MAX, 3.0, 4.0, 5.0, 6.0, 29.0, 29.0],
    );
    read(&old_output, &output_before);
    narrow(&resident, &mut output).unwrap();
    read(&output, &output_before);

    let error = narrow(&invalid, &mut stack).unwrap_err();
    let fault = error.arithmetic_fault().unwrap();
    assert_eq!(fault.kind, PcuExecutionFaultKind::ArithmeticOverflow);
    assert_eq!(fault.invocation_id, 2);
    assert!(!fault.recovered);
    bits(&stack, &short_before);
    let error = clamp(&invalid, &mut stack).unwrap_err();
    assert!(error.arithmetic_fault().unwrap().recovered);
    bits(&stack[..7], &[1.0, 2.0, f32::MAX, 3.0, 4.0, 5.0, 6.0]);
    bits(&stack[7..], &[23.0; 2]);
    narrow(&resident, &mut stack).unwrap();
    bits(&stack[..7], &expected);
    read(&resident, &input);
    read(&sibling, &input);
    read(&invalid, &[1.0, 2.0, f64::MAX, 3.0, 4.0, 5.0, 6.0]);
}
