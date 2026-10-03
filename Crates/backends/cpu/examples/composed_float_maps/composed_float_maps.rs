//! Actual ordinary checked nested expression with retained output tails.
#[rustfmt::skip]
use pcu_facade::{
    global,
    pcu,
};
#[pcu(crate_path=::pcu_facade,invocations=4)]
fn transform(input: &[f32], output: &mut [f32]) {
    let id = context.global_invocation_id;
    output[id] = (input[id] + input[id]) * input[id];
}
fn main() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        ..Default::default()
    })
    .unwrap();
    let mut output = [17_f32; 6];
    transform(&[0.5, 1., -1., 2.], &mut output).unwrap();
    assert_eq!(
        output.map(f32::to_bits),
        [0.5_f32, 2., 2., 8., 17., 17.].map(f32::to_bits)
    );
    println!("checked composed output: {output:?}");
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
