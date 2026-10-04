//! Large model literals must not require an equally large caller stack.
#![cfg(all(feature = "cpu", feature = "tensor"))]
#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuExecutionError,
    PcuTensor,
};

#[pcu]
fn weights() -> Result<PcuTensor<u32>, PcuExecutionError> {
    pcu::constant(const { [[0x1234_5678_u32; 512]; 512] })
}

#[test]
fn megabyte_matrix_literal_captures_on_small_stack_and_retains_owned_storage() {
    std::thread::Builder::new()
        .name("pcu-small-stack-literal".into())
        .stack_size(128 * 1024)
        .spawn(|| {
            global::configure(global::PcuExecutionPolicy {
                backend: global::PcuBackendChoice::Cpu,
                ..Default::default()
            })
            .unwrap();
            let first = weights().unwrap();
            let replay = weights().unwrap();
            assert_eq!(first.shape(), [512, 512]);
            assert_eq!(first.len(), 512 * 512);
            global::clear_thread_cache().unwrap();
            let mut host = vec![0xfeed_face; first.len() + 2];
            for owner in [first, replay] {
                owner.read_into(&mut host).unwrap();
                assert!(
                    host[..owner.len()]
                        .iter()
                        .all(|&value| value == 0x1234_5678)
                );
                assert_eq!(&host[owner.len()..], &[0xfeed_face; 2]);
            }
            global::clear_thread_cache().unwrap();
        })
        .unwrap()
        .join()
        .unwrap();
}
