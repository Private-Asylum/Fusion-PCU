//! Genuine original normal-header unary source with unread declarations and useful Clamp notice.
#[rustfmt::skip]
use pcu_facade::{
    global,
    pcu,
    PcuCheckedFloat,
    PcuExecutionError,
    PcuExecutionFaultKind,
    PcuScalar,
    PcuTensor,
};
#[pcu(crate_path = ::pcu_facade)]
fn retain<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
#[pcu(invocations = 3, flag(clamp_range), flag(reject_subnormal_result), crate_path = ::pcu_facade)]
fn activate<T: PcuCheckedFloat>(unread: &[T], output: &mut [T], seed: &T, input: &[T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[id]);
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Mlx,
        ..Default::default()
    })?;
    let input = [f64::from_bits(1), -0.0, -2.0, f64::NAN, f64::INFINITY];
    let owner = retain(&input)?;
    let mut output = [91.0; 5];
    let error = activate(&[] as &[f64], &mut output, &f64::NAN, &owner).unwrap_err();
    let fault = error
        .arithmetic_fault()
        .expect("exact subnormal selection notice");
    assert!(fault.recovered);
    assert_eq!(fault.kind, PcuExecutionFaultKind::ArithmeticUnderflow);
    assert_eq!(fault.invocation_id, 0);
    assert_eq!(
        output.map(f64::to_bits),
        [1, 0, 0, 91.0_f64.to_bits(), 91.0_f64.to_bits()]
    );
    global::clear_thread_cache()?;
    let mut original = [0.0; 7];
    owner.read_into(&mut original)?;
    assert_eq!(
        original[..5]
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        input.map(f64::to_bits)
    );
    println!(
        "normal unary actual roles: {:?}; recovered subnormal notice",
        output.map(f64::to_bits)
    );
    Ok(())
}
