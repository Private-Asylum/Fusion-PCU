//! Explicit CPU selection repeats one wide carrier through ordinary generic source.
#[rustfmt::skip]
use pcu_facade::{pcu,global,PcuScalar,PcuU512};
#[pcu(invocations=N,crate_path=::pcu_facade)]
fn repeat<T: PcuScalar, const N: usize>(input: &T, output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = *input;
}
fn main() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        ..Default::default()
    })
    .unwrap();
    let input = PcuU512::from_limbs_le([1, 2, 3, 4, 5, 6, 7, u64::MAX]);
    let sentinel = PcuU512::from_limbs_le([17; 8]);
    let mut output = [sentinel; 10];
    repeat::<_, 7>(&input, &mut output).unwrap();
    assert_eq!(output[..7], [input; 7]);
    assert_eq!(output[7..], [sentinel; 3]);
    println!(
        "CPU repeated one 512-bit carrier into seven outputs and preserved three tail elements"
    );
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
