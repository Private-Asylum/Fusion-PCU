//! CPU reference conformance for ordinary tuple owners and composable tuple helpers.
#![cfg(all(feature = "cpu", feature = "tensor"))]
use fusion_pcu::pcu;
#[rustfmt::skip]
use fusion_pcu::{
    global,
    PcuExecutionError,
    PcuTensor,
};

#[pcu]
fn pair(input: &[f32]) -> Result<(PcuTensor<f32>, PcuTensor<f32>), PcuExecutionError> {
    let positive = pcu::relu(input)?;
    let doubled = pcu::add(&positive, &positive)?;
    Ok((positive, doubled))
}

#[pcu]
fn nested(input: &[f32]) -> Result<(PcuTensor<f32>,), PcuExecutionError> {
    let (positive, doubled) = pair(input)?;
    Ok((pcu::add(&positive, &doubled)?,))
}

#[pcu]
fn fault_pair(
    input: &[f32],
    denominator: &[f32],
) -> Result<(PcuTensor<f32>, PcuTensor<f32>), PcuExecutionError> {
    let valid = pcu::relu(input)?;
    let quotient = pcu::div(input, denominator)?;
    Ok((valid, quotient))
}

static BUILDS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static SITE: global::PcuHostCallSite = global::PcuHostCallSite::new();
fn counted_capture(
    capture: &mut global::PcuTensorGraphCapture,
    [input]: [global::PcuTensorGraphValue<f32>; 1],
) -> Result<[global::PcuTensorGraphValue<f32>; 2], PcuExecutionError> {
    BUILDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok([capture.identity(input)?, capture.relu(input)?])
}

fn read<const N: usize>(owner: &PcuTensor<f32>) -> [u32; N] {
    let mut values = [0.0; N];
    owner.read_into(&mut values).unwrap();
    values.map(f32::to_bits)
}

#[test]
fn ordinary_tuple_owners_retain_storage_replay_helpers_and_publish_only_on_success() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        ..global::PcuExecutionPolicy::default()
    })
    .unwrap();
    let (held, held_double) = pair(&[-2.0, 3.0]).unwrap();
    let (later, later_double) = pair(&[4.0, -1.0]).unwrap();
    assert_eq!(read::<2>(&held), [0.0_f32, 3.0].map(f32::to_bits));
    assert_eq!(read::<2>(&held_double), [0.0_f32, 6.0].map(f32::to_bits));
    assert_eq!(read::<2>(&later), [4.0_f32, 0.0].map(f32::to_bits));
    assert_eq!(read::<2>(&later_double), [8.0_f32, 0.0].map(f32::to_bits));
    let (combined,) = nested(&later).unwrap();
    assert_eq!(read::<2>(&combined), [12.0_f32, 0.0].map(f32::to_bits));
    assert!(fault_pair(&[2.0, 3.0], &[1.0, 0.0]).is_err());
    let (valid, quotient) = fault_pair(&[2.0, 3.0], &[1.0, 3.0]).unwrap();
    assert_eq!(read::<2>(&valid), [2.0_f32, 3.0].map(f32::to_bits));
    assert_eq!(read::<2>(&quotient), [2.0_f32, 1.0].map(f32::to_bits));
    for input in [[-1.0_f32, 2.0], [3.0, -4.0]] {
        let source = global::PcuTensorSource::<f32>::as_tensor_source(input.as_slice()).unwrap();
        let [identity, positive] = global::call_owned_tensors_capture::<f32, 1, 2, _>(
            &SITE,
            core::any::TypeId::of::<()>(),
            [source],
            counted_capture,
        )
        .unwrap();
        assert_eq!(read::<2>(&identity), input.map(f32::to_bits));
        assert_eq!(
            read::<2>(&positive),
            input.map(|value| value.max(0.0).to_bits())
        );
    }
    assert_eq!(BUILDS.load(std::sync::atomic::Ordering::Relaxed), 1);
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
