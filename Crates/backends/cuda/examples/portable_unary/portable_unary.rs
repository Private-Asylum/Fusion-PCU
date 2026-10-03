//! Genuine requested Portable unary with exact encoding-preserving values.
extern crate pcu_facade as fusion_pcu;
#[path = "../../benches/portable_unary/source/source.rs"]
#[allow(dead_code)] // Other source geometries are covered by integration and paired benchmarks.
mod source;
fn main() {
    fusion_pcu::global::configure(fusion_pcu::global::PcuExecutionPolicy {
        backend: fusion_pcu::global::PcuBackendChoice::Cuda,
        device: Some(0),
        ..Default::default()
    })
    .unwrap();
    let input = [1.0_f64, -0.0, f64::from_bits(1), -2.0];
    let mut output = [7.0; 6];
    source::neg::<f64, 4>(&mut output, &input).unwrap();
    assert_eq!(
        std::array::from_fn::<_, 4, _>(|i| output[i].to_bits()),
        input.map(|v| v.to_bits() ^ (1 << 63))
    );
    assert_eq!(output[4..], [7.0; 2]);
    source::relu::<f64, 4>(&mut output, &input).unwrap();
    assert_eq!(
        std::array::from_fn::<_, 4, _>(|i| output[i].to_bits()),
        [1.0_f64.to_bits(), 0, 1, 0]
    );
    println!("requested Portable unary exact F64 bits and preserved tails");
}
