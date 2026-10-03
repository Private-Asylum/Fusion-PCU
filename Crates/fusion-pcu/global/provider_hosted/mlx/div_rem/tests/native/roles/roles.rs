//! Actual unique resident reads share the canonical joint publication law.
use super::*;

#[path = "spans/spans.rs"]
mod spans;

#[crate::pcu(crate_path = crate, invocations = 4)]
fn repeated(q: &mut [i64], input: &[i64; 4], r: &mut [i64]) {
    let id = pcu::context::global_invocation_id();
    let (quotient, remainder) = pcu::checked_div_rem(input[id], input[id]);
    r[id] = remainder;
    q[id] = quotient;
}

#[crate::pcu(crate_path = crate, invocations = 4)]
fn unread(_unused: &[i64], r: &mut [i64], input: &[i64; 4], q: &mut [i64]) {
    let id = pcu::context::global_invocation_id();
    let (quotient, remainder) = pcu::checked_div_rem(input[id], input[0]);
    q[id] = quotient;
    r[id] = remainder;
}

#[crate::pcu(crate_path = crate, invocations = 3)]
fn grid(_unused: &[i64], r: &mut [i64], input: &[i64; 4], q: &mut [i64]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < 4 {
        let (quotient, remainder) = pcu::checked_div_rem(input[0], input[id]);
        r[id] = remainder;
        q[id] = quotient;
        id += stride;
    }
}

#[test]
#[cfg_attr(
    not(all(target_os = "macos", target_arch = "aarch64")),
    ignore = "requires native MLX actual-read division and joint prefix publication"
)]
fn one_resident_input_and_unread_foreign_owner_preserve_transactional_outputs() {
    let _guard = global::policy::TEST_LOCK.lock().unwrap();
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Mlx,
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
    let root = root();
    let foreign = owner(&super::root(), &[81_i64; 4]);
    let mut q = owner(&root, &[91_i64; 6]);
    let mut r = owner(&root, &[92_i64; 9]);
    for phase in 0..3_i64 {
        let input = [-17 - phase, -19 - phase, 21 + phase, 23 + phase];
        let input_owner = owner(&root, &input);
        repeated(&mut q, &input_owner, &mut r).unwrap();
        assert_eq!(values::<6>(&q), [1, 1, 1, 1, 91, 91]);
        assert_eq!(values::<9>(&r), [0, 0, 0, 0, 92, 92, 92, 92, 92]);

        // The unread owner has another actual session. It contributes neither
        // residency affinity nor a native input; this is a source fact, not a
        // permission to ignore a real foreign device read.
        unread(&foreign, &mut r, &input_owner, &mut q).unwrap();
        let expected_q = input.map(|value| value / input[0]);
        let expected_r = input.map(|value| value % input[0]);
        assert_eq!(&values::<6>(&q)[..4], &expected_q);
        assert_eq!(&values::<9>(&r)[..4], &expected_r);
        assert_eq!(&values::<6>(&q)[4..], &[91; 2]);
        assert_eq!(&values::<9>(&r)[4..], &[92; 5]);

        let mut host_q = [93_i64; 7];
        grid(&[], &mut r, &input_owner, &mut host_q).unwrap();
        assert_eq!(&host_q[..4], &input.map(|value| input[0] / value));
        assert_eq!(&host_q[4..], &[93; 3]);
        let mut host_r = [94_i64; 8];
        grid(&foreign, &mut host_r, &input_owner, &mut q).unwrap();
        assert_eq!(&host_r[..4], &input.map(|value| input[0] % value));
        assert_eq!(&host_r[4..], &[94; 4]);

        let before_q = values::<6>(&q);
        let before_r = values::<9>(&r);
        let zeros = owner(&root, &[17_i64, 19, 0, 0]);
        fault(
            &repeated(&mut q, &zeros, &mut r).unwrap_err(),
            PcuExecutionFaultKind::DivideByZero,
        );
        assert_eq!(values::<6>(&q), before_q);
        assert_eq!(values::<9>(&r), before_r);
        let overflow = owner(&root, &[i64::MIN, 1, -1, 0]);
        fault(
            &grid(&foreign, &mut r, &overflow, &mut q).unwrap_err(),
            PcuExecutionFaultKind::SignedDivisionOverflow,
        );
        assert_eq!(values::<6>(&q), before_q);
        assert_eq!(values::<9>(&r), before_r);
        let mut short = [95_i64; 3];
        assert!(grid(&[], &mut short, &input_owner, &mut q).is_err());
        assert_eq!(values::<6>(&q), before_q);
        assert_eq!(short, [95; 3]);
        repeated(&mut q, &input_owner, &mut r).unwrap();

        let mut host_q = [96_i64; 6];
        let mut host_r = [97_i64; 9];
        // A lone resident input with host destinations legitimately selects its
        // own session; only a conflict between actual resident owners rejects.
        repeated(&mut host_q, &foreign, &mut host_r).unwrap();
        assert_eq!(host_q, [1, 1, 1, 1, 96, 96]);
        assert_eq!(host_r, [0, 0, 0, 0, 97, 97, 97, 97, 97]);
        let before_q = values::<6>(&q);
        let before_r = values::<9>(&r);
        assert!(matches!(
            repeated(&mut q, &foreign, &mut r),
            Err(PcuExecutionError::Argument(
                global::PcuArgumentError::SessionMismatch
            ))
        ));
        assert_eq!(values::<6>(&q), before_q);
        assert_eq!(values::<9>(&r), before_r);
    }
    global::clear_thread_cache().unwrap();
    drop((foreign, q, r));
    global::use_defaults().unwrap();
}
