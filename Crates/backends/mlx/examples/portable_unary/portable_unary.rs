//! Genuine deterministic source retains exact signed zero and gradual subnormal bits.
#[rustfmt::skip]
use pcu_facade::{
    global,
    pcu,
    PcuCheckedFloat,
};
#[pcu(invocations = 3, flag(deterministic), flag(allow_gradual_underflow), crate_path = ::pcu_facade)]
fn negate<T: PcuCheckedFloat>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}
fn main() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Mlx,
        ..Default::default()
    })
    .unwrap();
    let input = [f64::from_bits(1), -0.0, 2.0];
    let mut output = [91.0; 5];
    negate(&input, &mut output).unwrap();
    assert_eq!(
        output.map(f64::to_bits),
        [
            0x8000_0000_0000_0001,
            0,
            (-2.0_f64).to_bits(),
            91.0_f64.to_bits(),
            91.0_f64.to_bits()
        ]
    );
    global::clear_thread_cache().unwrap();
    println!("Portable exact unary bits {:?}", output.map(f64::to_bits));
}
