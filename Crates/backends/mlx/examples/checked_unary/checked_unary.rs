//! MLX owns and schedules the checked encoding kernel; ordinary source stages plain RAM.
#[rustfmt::skip]
use pcu_facade::{pcu,PcuF16Bits as Half,PcuCheckedFloat,global::{configure,PcuExecutionPolicy,PcuBackendChoice}};
#[pcu(invocations=3,flag(strict),flag(clamp_range),flag(reject_subnormal_result),crate_path=::pcu_facade)]
fn negate<T: PcuCheckedFloat>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}
fn main() {
    if !cfg!(target_os = "macos") {
        println!("SKIP: MLX GPU is unavailable");
        return;
    }
    configure(PcuExecutionPolicy {
        backend: PcuBackendChoice::Mlx,
        ..PcuExecutionPolicy::default()
    })
    .unwrap();
    let input = [
        Half::from_bits(1),
        Half::pcu_checked_from_f32(2.0).unwrap(),
        Half::from_bits(0x8000),
    ];
    let mut output = [Half::from_bits(0); 3];
    let recovery = negate(&input, &mut output).unwrap_err();
    println!("Recovered checked underflow: {recovery:?}; exact payload: {output:?}");
    let input64 = [f64::from_bits(1), 2.0_f64, -0.0_f64];
    let mut output64 = [91.0_f64; 5];
    let recovery64 = negate(&input64, &mut output64).unwrap_err();
    assert_eq!(
        output64.map(f64::to_bits),
        [f64::from_bits(0x8000_0000_0000_0001), -2.0, 0.0, 91.0, 91.0].map(f64::to_bits)
    );
    println!("Exact paired-U32 F64 checked payload: {output64:?}; {recovery64:?}");
}
