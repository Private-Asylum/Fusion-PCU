//! Actual wide integer repeated source, ignored declaration and independent index-zero operands.
#[path = "../../../cpu/benches/integer_operands/source/source.rs"]
#[allow(dead_code)] // The paired target exercises every companion source specialization.
mod source;
#[rustfmt::skip]
use pcu_facade::{
    global,
    PcuU512,
};
fn main() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        numerical_options: pcu_facade::PcuNumericalOptions {
            reproducibility: if std::env::var_os("PCU_INTEGER_PORTABLE").is_some() {
                pcu_facade::PcuReproducibility::PortableV1
            } else {
                pcu_facade::PcuReproducibility::Unspecified
            },
            ..Default::default()
        },
        ..Default::default()
    })
    .unwrap();
    let base = PcuU512::from_limbs_le([2, 0, 0, 1, 0, 0, 0, 0]);
    let input = [base; 7];
    let sentinel = PcuU512::from_limbs_le([91, 0, 0, 0, 0, 0, 0, 0]);
    let mut output = [sentinel; 10];
    source::doubled::<PcuU512, 7>(&mut output, &input).unwrap();
    assert_eq!(output[0].to_limbs_le(), [4, 0, 0, 2, 0, 0, 0, 0]);
    assert_eq!(output[7..], [sentinel; 3]);
    source::independent::<PcuU512, 7>(&input, &mut output).unwrap();
    assert_eq!(output[..7], [PcuU512::from_limbs_le([0; 8]); 7]);
    assert_eq!(output[7..], [sentinel; 3]);
    source::squared::<u32, 7>(&[], &mut [0; 7], &[3; 7]).unwrap();
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
    println!("Vulkan wide repeated/index-zero source and untouched tails PASS");
}
