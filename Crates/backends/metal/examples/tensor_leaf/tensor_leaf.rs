//! Ordinary source ownership, retained exact Metal session and cache-independent readback.
#[path = "../../benches/tensor_leaf/source/source.rs"]
mod source;
use pcu_facade::global;
#[pcu_facade::pcu(invocations = 3, crate_path = ::pcu_facade)]
fn scalar_copy(input: &[u64], output: &mut [u64]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id];
}
fn main() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Metal,
        block_size: 32,
        ..Default::default()
    })
    .unwrap();
    let mut input = [
        0x8000_0000_0000_0000_u64,
        0xffff_ffff_ffff_ffff,
        0x1234_5678_9abc_def0,
    ];
    let owner = source::retain(&input).unwrap();
    input.fill(0);
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Metal,
        block_size: 64,
        ..Default::default()
    })
    .unwrap();
    let sibling = source::retain(&owner).unwrap();
    drop(owner);
    global::clear_thread_cache().unwrap();
    let mut output = [91_u64; 5];
    scalar_copy(&sibling, &mut output).unwrap();
    assert_eq!(
        output,
        [
            0x8000_0000_0000_0000,
            0xffff_ffff_ffff_ffff,
            0x1234_5678_9abc_def0,
            91,
            91
        ]
    );
    output.fill(91);
    sibling.read_into(&mut output).unwrap();
    assert_eq!(
        output,
        [
            0x8000_0000_0000_0000,
            0xffff_ffff_ffff_ffff,
            0x1234_5678_9abc_def0,
            91,
            91
        ]
    );
    println!("Retained ordinary Metal carrier: {output:x?}");
    global::use_defaults().unwrap();
}
