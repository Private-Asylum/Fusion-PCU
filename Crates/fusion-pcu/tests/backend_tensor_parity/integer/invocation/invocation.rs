//! CPU owners lend ordinary initialized RAM while retaining explicit CPU execution affinity.
use super::*;
#[rustfmt::skip]
use core::sync::atomic::{
    AtomicUsize,
    Ordering,
};
#[pcu(invocations = 5)]
fn direct<T: PcuCheckedInteger>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] + right[id];
}
#[pcu(invocations = 5)]
fn reordered<T: PcuCheckedInteger>(output: &mut [T], right: &[T], left: &[T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] + right[id];
}
#[pcu(invocations = 5, flag(clamp_range))]
fn clamp<T: PcuCheckedInteger>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] + right[id];
}
static SCORES: AtomicUsize = AtomicUsize::new(0);
fn score(_: &global::PcuInvocationCandidate<'_>) -> i128 {
    SCORES.fetch_add(1, Ordering::Relaxed);
    1
}
fn read<T: Sample>(owner: &PcuTensor<T>, expected: &[T; 7]) {
    let mut stack = [T::small(11); 9];
    owner.read_into(&mut stack).unwrap();
    bits(&stack[..7], expected);
    bits(&stack[7..], &[T::small(11); 2]);
}
fn format<T: Sample>(
    numerical_mode: PcuNumericalMode,
    numerical_options: fusion_pcu::PcuNumericalOptions,
) {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        numerical_mode,
        numerical_options,
        ..Default::default()
    })
    .unwrap();
    let left = [2, 3, 4, 5, 6, 9, 9].map(T::small);
    let right = [T::small(1); 7];
    let expected = [3, 4, 5, 6, 7, 9, 9].map(T::small);
    let source = retain(&left).unwrap();
    let rhs = retain(&right).unwrap();
    let sibling = retain(&left).unwrap();
    let mut destination = retain(&[T::small(9); 7]).unwrap();
    let mut host = [T::small(9); 7];
    direct::<T>(&source, &rhs, &mut host).unwrap();
    bits(&host, &expected);
    direct::<T>(&left, &right, &mut destination).unwrap();
    read(&destination, &expected);
    direct::<T>(&source, &right, &mut destination).unwrap();
    reordered::<T>(&mut destination, &rhs, &source).unwrap();
    read(&destination, &expected);
    read(&source, &left);
    read(&sibling, &left);
    let mut invalid = left;
    invalid[2] = T::MAX;
    let invalid_owner = retain(&invalid).unwrap();
    assert!(
        matches!(direct::<T>(&invalid_owner, &rhs, &mut destination),
        Err(PcuExecutionError::ArithmeticFault(fault)) if !fault.recovered
        && fault.invocation_id == 2 && fault.kind == PcuExecutionFaultKind::ArithmeticOverflow)
    );
    read(&destination, &expected);
    read(&invalid_owner, &invalid);
    let error = clamp::<T>(&invalid_owner, &rhs, &mut destination).unwrap_err();
    let fault = error.recovered_range_fault().unwrap();
    assert_eq!(fault.invocation_id, 2);
    assert_eq!(fault.kind, PcuExecutionFaultKind::ArithmeticOverflow);
    let mut clamped = expected;
    clamped[2] = T::MAX;
    read(&destination, &clamped);
    read(&sibling, &left);
    // Owned RAM pins CPU in Automatic mode. Discovery/SDK probing/scoring is cold-only
    // and unnecessary for this already-known affinity, including the initial call.
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Automatic,
        numerical_mode,
        numerical_options,
        score_invocation: Some(score),
        ..Default::default()
    })
    .unwrap();
    SCORES.store(0, Ordering::Relaxed);
    for phase in 0..64 {
        let changed = [T::small(1 + phase % 5); 7];
        direct::<T>(&source, &changed, &mut destination).unwrap();
    }
    assert_eq!(SCORES.load(Ordering::Relaxed), 0);
    drop(source);
    drop(rhs);
    drop(invalid_owner);
    global::clear_thread_cache().unwrap();
    read(&sibling, &left);
    let mut final_expected = expected;
    for (index, element) in final_expected.iter_mut().take(5).enumerate() {
        *element = T::small(u8::try_from(index).unwrap() + 6);
    }
    read(&destination, &final_expected);
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Rocm,
        ..Default::default()
    })
    .unwrap();
    assert!(matches!(
        direct::<T>(&sibling, &right, &mut destination),
        Err(PcuExecutionError::ResidentPolicyConflict)
    ));
    read(&destination, &final_expected);
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        ..Default::default()
    })
    .unwrap();
}
pub(super) fn verify() {
    let _guard = POLICY_LOCK.lock().unwrap();
    for numerical_mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for compound_arithmetic in [
            fusion_pcu::PcuCompoundArithmeticPolicy::Checked,
            fusion_pcu::PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for precision in [
                fusion_pcu::PcuPrecisionPolicy::Preserve,
                fusion_pcu::PcuPrecisionPolicy::BackendOptimized,
            ] {
                let numerical_options = fusion_pcu::PcuNumericalOptions {
                    compound_arithmetic,
                    precision,
                    ..Default::default()
                };
                format::<u8>(numerical_mode, numerical_options);
                format::<i8>(numerical_mode, numerical_options);
                format::<u16>(numerical_mode, numerical_options);
                format::<i16>(numerical_mode, numerical_options);
                format::<u32>(numerical_mode, numerical_options);
                format::<i32>(numerical_mode, numerical_options);
                format::<u64>(numerical_mode, numerical_options);
                format::<i64>(numerical_mode, numerical_options);
                format::<u128>(numerical_mode, numerical_options);
                format::<i128>(numerical_mode, numerical_options);
                format::<PcuU256>(numerical_mode, numerical_options);
                format::<PcuI256>(numerical_mode, numerical_options);
                format::<PcuU512>(numerical_mode, numerical_options);
                format::<PcuI512>(numerical_mode, numerical_options);
            }
        }
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
