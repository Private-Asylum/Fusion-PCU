//! Ordinary generic 512-bit transport preserves the entire carrier and output tail.
#[rustfmt::skip]
use pcu_facade::{pcu,global,PcuScalar,PcuU512};
#[pcu(invocations=N,crate_path=::pcu_facade)]
fn repeat<T: PcuScalar, const N: usize>(input: &T, output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = *input;
}
fn main() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        ..Default::default()
    })
    .unwrap();
    let input = PcuU512::decode_le([0xa5; 64]);
    let tail = PcuU512::decode_le([0x17; 64]);
    let mut output = [tail; 68];
    repeat::<_, 65>(&input, &mut output).unwrap();
    assert!(
        output[..65]
            .iter()
            .all(|value| value.encode_le() == input.encode_le())
    );
    assert!(
        output[65..]
            .iter()
            .all(|value| value.encode_le() == tail.encode_le())
    );
    println!("ordinary Vulkan U512 scalar broadcast preserves all64 bytes and caller tail");
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
