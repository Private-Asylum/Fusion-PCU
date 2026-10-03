//! Ordinary annotated source preserves useful clamped payload and rejects nonfinite input transactionally.
#[rustfmt::skip]
use pcu_facade::{pcu,PcuCheckedFloat,PcuF8E4M3FnBits,global,PcuExecutionFaultKind};
#[pcu(invocations=7,crate_path=::pcu_facade,flag(clamp_range),flag(reject_subnormal_result))]
fn relu<T: PcuCheckedFloat>(input: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = pcu::relu(input[id]);
}
fn main() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        ..Default::default()
    })
    .unwrap();
    let tiny = PcuF8E4M3FnBits::from_bits(1);
    let mut input = [tiny; 7];
    let sentinel = PcuF8E4M3FnBits::from_bits(17);
    let mut output = [sentinel; 10];
    let error = relu(&input, &mut output).unwrap_err();
    assert!(
        matches!(error,global::PcuExecutionError::ArithmeticFault(f) if f.recovered&&f.kind==PcuExecutionFaultKind::ArithmeticUnderflow&&f.invocation_id==0)
    );
    assert_eq!(&output[..7], &input);
    assert_eq!(&output[7..], &[sentinel; 3]);
    input[6] = PcuF8E4M3FnBits::from_bits(0x7f);
    let before = output;
    assert!(
        matches!(relu(&input,&mut output),Err(global::PcuExecutionError::ArithmeticFault(f)) if !f.recovered&&f.invocation_id==6)
    );
    assert_eq!(output, before);
    input.fill(PcuF8E4M3FnBits::from_bits(0x38));
    relu(&input, &mut output).unwrap();
    assert_eq!(&output[..7], &input);
    println!(
        "ordinary FP8 ReLU: useful recovered output, later fatal rollback, normal retry and tails pass"
    );
}
