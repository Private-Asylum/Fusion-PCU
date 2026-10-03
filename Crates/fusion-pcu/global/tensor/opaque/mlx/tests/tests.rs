//! Genuine separate sessions must remain foreign despite equal device and shape.
use core::marker::PhantomData;
use std::rc::Rc;

#[rustfmt::skip]
use crate::{
    global,
    global::arguments::{MlxSourceRoot, TensorBacking},
    PcuDeviceActivation,
    PcuExecutionError,
    PcuTensor,
};
use fusion_pcu_mlx::MlxDiscovery;

#[crate::pcu(crate_path = crate, flag(non_strict), flag(non_deterministic), flag(native_compound), flag(backend_precision))]
fn product(
    left: &[[f32; 2]; 2],
    right: &[[f32; 2]; 2],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::matmul(left, right)
}

#[crate::pcu(crate_path = crate)]
fn identity(input: &[[f32; 2]; 2]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::identity(input)
}

#[crate::pcu(crate_path = crate)]
fn difference(
    left: &[[f32; 2]; 2],
    right: &[[f32; 2]; 2],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::sub(left, right)
}

#[crate::pcu(crate_path = crate)]
fn repeat_with_foreign_unused(
    _unused: &[[f32; 2]; 2],
    input: &[[f32; 2]; 2],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::add(input, input)
}

fn separate_owner(values: &[f32; 4]) -> PcuTensor<f32> {
    let discovery = MlxDiscovery::discover_default().unwrap();
    let reference = discovery.device_reference(0).unwrap();
    let session = discovery.open_device(reference).unwrap();
    let array = session.upload_f32([2, 2], values).unwrap();
    PcuTensor {
        backing: TensorBacking::Mlx {
            array,
            shape: [2, 2],
            root: Rc::new(MlxSourceRoot { discovery, session }),
            marker: PhantomData,
        },
    }
}

#[crate::pcu(crate_path = crate, invocations = 4)]
fn integer_sum<T: crate::PcuCheckedInteger>(left: &[T], output: &mut [T], right: &[T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] + right[id];
}

#[crate::pcu(crate_path = crate, invocations = 4, flag(clamp_range))]
fn integer_repeat<T: crate::PcuCheckedInteger>(unused: &[T], input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] + input[id];
}

fn separate_integer_owner(values: &[i128]) -> PcuTensor<i128> {
    let discovery = MlxDiscovery::discover_default().unwrap();
    let reference = discovery.device_reference(0).unwrap();
    let session = discovery.open_device(reference).unwrap();
    let array = session.upload_encoded(values).unwrap();
    PcuTensor {
        backing: TensorBacking::MlxEncoded {
            array,
            shape: Rc::from([values.len()]),
            root: Rc::new(MlxSourceRoot { discovery, session }),
            marker: PhantomData,
        },
    }
}

#[crate::pcu(crate_path = crate)]
fn owned_integer_sum<T: crate::PcuCheckedInteger>(
    left: &[T],
    right: &[T],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::add(left, right)
}

#[crate::pcu(crate_path = crate)]
fn owned_integer_repeat<T: crate::PcuCheckedInteger>(
    unused: &[T],
    input: &[T],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::add(input, input)
}

#[crate::pcu(crate_path = crate)]
fn owned_integer_checked_unused<T: crate::PcuCheckedInteger>(
    left: &[T],
    right: &[T],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let _checked = pcu::add(left, right);
    pcu::identity(left)
}

#[test]
#[cfg_attr(
    not(all(target_os = "macos", target_arch = "aarch64")),
    ignore = "requires actual MLX owned integer separate-session and unused-effect lifetime"
)]
fn integer_owned_calls_reject_used_foreign_sessions_and_preserve_checked_effects() {
    let _guard = global::policy::TEST_LOCK.lock().unwrap();
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Mlx,
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
    let values = [1_i128, 2, 3, 4];
    let selected = separate_integer_owner(&values);
    let foreign = separate_integer_owner(&[4_i128, 3, 2, 1]);
    let unused = separate_integer_owner(&[0_i128]);
    assert!(matches!(
        owned_integer_sum(&selected, &foreign),
        Err(PcuExecutionError::Argument(
            global::PcuArgumentError::SessionMismatch
        ))
    ));
    let peer = owned_integer_sum(&selected, &[0_i128; 4]).unwrap();
    let sum = owned_integer_sum(&selected, &peer).unwrap();
    let repeated = owned_integer_repeat(&unused, &selected).unwrap();
    let checked = owned_integer_checked_unused(&selected, &peer).unwrap();
    let mut output = [99_i128; 6];
    sum.read_into(&mut output).unwrap();
    assert_eq!(output, [2, 4, 6, 8, 99, 99]);
    repeated.read_into(&mut output).unwrap();
    assert_eq!(output, [2, 4, 6, 8, 99, 99]);
    let error =
        owned_integer_checked_unused(&selected, &[0_i128, i128::MAX, 0, i128::MAX]).unwrap_err();
    let fault = error.arithmetic_fault().unwrap();
    assert_eq!(fault.kind, crate::PcuExecutionFaultKind::ArithmeticOverflow);
    assert_eq!(fault.invocation_id, 1);
    assert!(!fault.recovered);
    selected.read_into(&mut output).unwrap();
    assert_eq!(output, [1, 2, 3, 4, 99, 99]);
    foreign.read_into(&mut output).unwrap();
    assert_eq!(output, [4, 3, 2, 1, 99, 99]);
    let retry = owned_integer_checked_unused(&selected, &peer).unwrap();
    global::clear_thread_cache().unwrap();
    drop(selected);
    drop(peer);
    drop(foreign);
    drop(unused);
    drop(repeated);
    checked.read_into(&mut output).unwrap();
    assert_eq!(output, [1, 2, 3, 4, 99, 99]);
    retry.read_into(&mut output).unwrap();
    assert_eq!(output, [1, 2, 3, 4, 99, 99]);
    sum.read_into(&mut output).unwrap();
    assert_eq!(output, [2, 4, 6, 8, 99, 99]);
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

#[test]
#[cfg_attr(
    not(all(target_os = "macos", target_arch = "aarch64")),
    ignore = "requires actual Apple silicon MLX integer separate-session ownership"
)]
fn integer_invocations_reject_used_foreign_sessions_but_ignore_unused_owners() {
    let _guard = global::policy::TEST_LOCK.lock().unwrap();
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Mlx,
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
    let values = [1_i128, 2, 3, i128::MAX];
    let selected = separate_integer_owner(&values);
    // Actual encoded arrays come from distinct native sessions, not fabricated
    // identities or matching device/shape surrogates. The unused owner is shorter
    // than the four-lane workload; an actual read would fail extent validation.
    let foreign = separate_integer_owner(&[4, 3, 2, 1]);
    let unused = separate_integer_owner(&[0]);
    let mut output = [99_i128; 6];
    assert!(matches!(
        integer_sum(&selected, &mut output, &foreign),
        Err(PcuExecutionError::Argument(
            global::PcuArgumentError::SessionMismatch
        ))
    ));
    assert_eq!(output, [99; 6]);
    integer_sum(&selected, &mut output, &[0_i128; 4]).unwrap();
    assert_eq!(output, [1, 2, 3, i128::MAX, 99, 99]);
    let fault = integer_repeat(&unused, &selected, &mut output).unwrap_err();
    let observed = fault
        .arithmetic_fault()
        .expect("completed clamp reports its fault");
    assert_eq!(
        observed.kind,
        crate::PcuExecutionFaultKind::ArithmeticOverflow
    );
    assert_eq!(observed.invocation_id, 3);
    assert!(observed.recovered);
    assert_eq!(output, [2, 4, 6, i128::MAX, 99, 99]);
    integer_repeat(&unused, &[1_i128, 2, 3, 4], &mut output).unwrap();
    assert_eq!(output, [2, 4, 6, 8, 99, 99]);
    let mut original = [77_i128; 5];
    selected.read_into(&mut original).unwrap();
    assert_eq!(original, [1, 2, 3, i128::MAX, 77]);
    foreign.read_into(&mut original).unwrap();
    assert_eq!(original, [4, 3, 2, 1, 77]);
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

#[test]
#[cfg_attr(
    not(all(target_os = "macos", target_arch = "aarch64")),
    ignore = "requires actual Apple silicon MLX separate-session ownership"
)]
fn separate_native_sessions_cannot_be_composed_by_matching_device_and_shape() {
    let _guard = global::policy::TEST_LOCK.lock().unwrap();
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Mlx,
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
    let left = [1.0_f32, 2.0, 3.0, 4.0];
    let right = [1.0_f32, 0.0, 0.0, 1.0];
    let left_owner = separate_owner(&left);
    let right_owner = separate_owner(&right);
    assert!(matches!(
        product(&left_owner, &right_owner),
        Err(PcuExecutionError::Argument(
            global::PcuArgumentError::SessionMismatch
        ))
    ));
    let encoded_left = identity(&left_owner).unwrap();
    let encoded_right = identity(&right_owner).unwrap();
    for (a, b) in [
        (&encoded_left, &encoded_right),
        (&encoded_left, &right_owner),
        (&left_owner, &encoded_right),
    ] {
        assert!(matches!(
            product(a, b),
            Err(PcuExecutionError::Argument(
                global::PcuArgumentError::SessionMismatch
            ))
        ));
        assert!(matches!(
            difference(a, b),
            Err(PcuExecutionError::Argument(
                global::PcuArgumentError::SessionMismatch
            ))
        ));
    }
    let mut stack = [99.0_f32; 5];
    left_owner.read_into(&mut stack).unwrap();
    assert_eq!(
        stack.map(f32::to_bits),
        [1.0_f32, 2.0, 3.0, 4.0, 99.0].map(f32::to_bits)
    );
    right_owner.read_into(&mut stack).unwrap();
    assert_eq!(
        stack.map(f32::to_bits),
        [1.0_f32, 0.0, 0.0, 1.0, 99.0].map(f32::to_bits)
    );
    encoded_left.read_into(&mut stack).unwrap();
    assert_eq!(
        stack.map(f32::to_bits),
        [1.0_f32, 2.0, 3.0, 4.0, 99.0].map(f32::to_bits)
    );
    encoded_right.read_into(&mut stack).unwrap();
    assert_eq!(
        stack.map(f32::to_bits),
        [1.0_f32, 0.0, 0.0, 1.0, 99.0].map(f32::to_bits)
    );
    // Only the actual selected parameter participates in affinity, even when
    // another declared Rust argument is a genuine foreign resident owner.
    let repeated = repeat_with_foreign_unused(&encoded_right, &encoded_left).unwrap();
    repeated.read_into(&mut stack).unwrap();
    assert_eq!(
        stack.map(f32::to_bits),
        [2.0_f32, 4.0, 6.0, 8.0, 99.0].map(f32::to_bits)
    );
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
