//! Ordinary MLX-owned scalar source. Hardware qualification requires the actual pinned image.
use fusion_pcu::pcu;
use fusion_pcu::PcuCheckedFloat;
use std::sync::Mutex;
static POLICY_LOCK: Mutex<()> = Mutex::new(());
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
#[rustfmt::skip]
use fusion_pcu::{
    global,
    PcuF16Bits,
};

#[pcu(invocations = N)]
fn neg<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
#[test]
fn explicit_mlx_refusal_preserves_host_output_and_does_not_substitute_cpu() {
    let _guard = POLICY_LOCK.lock().unwrap();
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Mlx,
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
    let input = [PcuF16Bits::from_bits(0x3c00); 2];
    let mut output = [PcuF16Bits::from_bits(0x4000); 2];
    let error = neg::<PcuF16Bits, 2>(&input, &mut output).unwrap_err();
    assert!(
        matches!(error, global::PcuExecutionError::NoCompatibleDevice {
        ref rejected, ref discovery,
    } if rejected.is_empty() && discovery.iter().any(|message| message.contains("MLX")))
    );
    assert_eq!(output.map(PcuF16Bits::to_bits), [0x4000; 2]);
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

// The fixtures use portable Rust APIs: compile/lint them on every host, while
// actual execution stays an explicit hardware-required ignored gate.
#[path = "native/native.rs"]
mod native;

#[test]
#[ignore = "requires authorized actual MLX GPU execution; no other provider substitutes"]
fn mlx_four_format_ordinary_source_publication_and_warm_reuse() {
    native::verify();
}

#[path = "owned/owned.rs"]
mod owned;

#[test]
#[ignore = "requires actual MLX GPU; required encoded owned source and effect gate"]
fn mlx_four_format_owned_source_liveness_and_local_policies() {
    owned::verify();
}

#[path = "invocation/invocation.rs"]
mod invocation;

#[test]
#[ignore = "requires actual MLX GPU; encoded invocation borrows and terminal output replacement"]
fn mlx_four_format_resident_invocation_borrows_and_publication() {
    invocation::verify();
}

#[path = "binary/binary.rs"]
mod binary;

#[test]
#[ignore = "requires actual MLX GPU; binary source mixed borrows, repeated inputs and publication"]
fn mlx_four_format_binary_invocation_borrows_and_publication() {
    binary::verify();
}

#[path = "transport/transport.rs"]
mod transport;

#[test]
#[ignore = "requires actual MLX GPU; ordered transport over host/resident borrows and joint publication"]
fn mlx_all_carrier_ordered_resident_transport_and_publication() {
    transport::verify();
}
