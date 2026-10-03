//! Genuine authored full-width quotient/remainder on the explicitly selected GPU.
extern crate pcu_facade as fusion_pcu;
#[path = "../../benches/wide_div_rem/source/source.rs"]
#[allow(dead_code)] // Example uses the direct entry; grid and identity serve the paired suite.
mod source;
fn main() {
    fusion_pcu::global::configure(fusion_pcu::global::PcuExecutionPolicy {
        backend: fusion_pcu::global::PcuBackendChoice::Cuda,
        ..Default::default()
    })
    .unwrap();
    let lhs = [u128::MAX, 1_u128 << 100];
    let rhs = [3_u128, 7];
    let mut q = [0_u128; 2];
    let mut r = q;
    source::direct::<u128, 2>(&lhs, &rhs, &mut q, &mut r).unwrap();
    for i in 0..2 {
        assert_eq!((q[i], r[i]), (lhs[i] / rhs[i], lhs[i] % rhs[i]));
    }
    println!("cuda full-width joint quotient/remainder passed");
}
