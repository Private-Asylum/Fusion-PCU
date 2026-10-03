//! Prepared annotated checked binary arithmetic with full host rollback on a terminal fault.
use fusion_pcu_cpu::PcuCpuHostBackend;
use pcu_facade::pcu;

#[pcu(invocations = N, crate_path = ::pcu_facade)]
fn divide<const N: usize>(lhs: &[f64], rhs: &[f64], output: &mut [f64]) {
    let id = context.global_invocation_id;
    output[id] = lhs[id] / rhs[id];
}

fn main() {
    let mut divide = divide_prepare::<3, _>(&PcuCpuHostBackend::scalar()).unwrap();
    let mut output = [99.0; 4];
    divide(&[6.0, 12.0, 20.0], &[2.0, 3.0, 4.0], &mut output).unwrap();
    assert_eq!(
        output.map(f64::to_bits),
        [3.0_f64, 4.0, 5.0, 99.0].map(f64::to_bits)
    );
    let fault = divide(&[6.0, 12.0, 20.0], &[2.0, 0.0, 4.0], &mut output).unwrap_err();
    assert_eq!(fault.fault().unwrap().invocation_id, 1);
    assert_eq!(
        output.map(f64::to_bits),
        [3.0_f64, 4.0, 5.0, 99.0].map(f64::to_bits)
    );
    println!("Checked f64 divide: {output:?}; zero denominator preserves every output.");
}
