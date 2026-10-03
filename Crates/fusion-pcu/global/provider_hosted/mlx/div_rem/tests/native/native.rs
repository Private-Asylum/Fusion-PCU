//! Authentic session fixtures exercise normal Rust calls and joint publication.
use core::marker::PhantomData;
use std::rc::Rc;
#[rustfmt::skip]
use crate::{
    global,
    global::arguments::{
        MlxSourceRoot,
        TensorBacking,
    },
    PcuDeviceActivation,
    PcuExecutionError,
    PcuExecutionFaultKind,
    PcuTensor,
};
use fusion_pcu_mlx::MlxDiscovery;

#[path = "roles/roles.rs"]
mod roles;

#[crate::pcu(crate_path = crate, invocations = 4)]
fn direct(left: &[i64; 4], right: &[i64; 4], q: &mut [i64], r: &mut [i64]) {
    let id = pcu::context::global_invocation_id();
    let (quotient, remainder) = pcu::checked_div_rem(left[id], right[id]);
    q[id] = quotient;
    r[id] = remainder;
}

#[crate::pcu(crate_path = crate, invocations = 3)]
fn grid(left: &[i64; 4], right: &[i64; 4], q: &mut [i64], r: &mut [i64]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < 4 {
        let (quotient, remainder) = pcu::checked_div_rem(left[id], right[id]);
        q[id] = quotient;
        r[id] = remainder;
        id += stride;
    }
}

fn root() -> Rc<MlxSourceRoot> {
    let discovery = MlxDiscovery::discover_default().unwrap();
    let reference = discovery.device_reference(0).unwrap();
    let session = discovery.open_device(reference).unwrap();
    Rc::new(MlxSourceRoot { discovery, session })
}
fn owner(root: &Rc<MlxSourceRoot>, values: &[i64]) -> PcuTensor<i64> {
    PcuTensor {
        backing: TensorBacking::MlxEncoded {
            array: root.session.upload_encoded(values).unwrap(),
            shape: Rc::from([values.len()]),
            root: Rc::clone(root),
            marker: PhantomData,
        },
    }
}
fn values<const N: usize>(owner: &PcuTensor<i64>) -> [i64; N] {
    let mut result = [0; N];
    owner.read_into(&mut result).unwrap();
    result
}
fn fault(error: &PcuExecutionError, kind: PcuExecutionFaultKind) {
    assert!(matches!(error, PcuExecutionError::ArithmeticFault(fault)
        if fault.kind == kind && fault.invocation_id == 2 && !fault.recovered));
}

#[test]
#[cfg_attr(
    not(all(target_os = "macos", target_arch = "aarch64")),
    ignore = "requires native MLX joint source/session/prefix publication"
)]
fn unequal_resident_prefixes_and_host_siblings_publish_only_as_a_completed_pair() {
    let _guard = global::policy::TEST_LOCK.lock().unwrap();
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Mlx,
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
    let root = root();
    let right = owner(&root, &[3_i64; 4]);
    let mut q = owner(&root, &[91_i64; 6]);
    let mut r = owner(&root, &[92_i64; 9]);
    for phase in 0..3_i64 {
        let left = [-17 - phase, -19 - phase, 21 + phase, 23 + phase];
        let expected_q = left.map(|value| value / 3);
        let expected_r = left.map(|value| value % 3);
        direct(&left, &right, &mut q, &mut r).unwrap();
        assert_eq!(
            values::<6>(&q),
            [
                expected_q[0],
                expected_q[1],
                expected_q[2],
                expected_q[3],
                91,
                91
            ]
        );
        assert_eq!(
            values::<9>(&r),
            [
                expected_r[0],
                expected_r[1],
                expected_r[2],
                expected_r[3],
                92,
                92,
                92,
                92,
                92
            ]
        );
        let prior_q = values::<6>(&q);
        let prior_r = values::<9>(&r);
        fault(
            &direct(&left, &[3_i64, 3, 0, 0], &mut q, &mut r).unwrap_err(),
            PcuExecutionFaultKind::DivideByZero,
        );
        assert_eq!(values::<6>(&q), prior_q);
        assert_eq!(values::<9>(&r), prior_r);
        fault(
            &direct(
                &[17_i64, 19, i64::MIN, i64::MIN],
                &[3_i64, 3, -1, -1],
                &mut q,
                &mut r,
            )
            .unwrap_err(),
            PcuExecutionFaultKind::SignedDivisionOverflow,
        );
        assert_eq!(values::<6>(&q), prior_q);
        assert_eq!(values::<9>(&r), prior_r);
        let mut host_q = [93_i64; 7];
        let mut host_r = [94_i64; 8];
        grid(&left, &right, &mut host_q, &mut r).unwrap();
        assert_eq!(&host_q[..4], &expected_q);
        assert_eq!(&host_q[4..], &[93; 3]);
        grid(&left, &right, &mut q, &mut host_r).unwrap();
        assert_eq!(&host_r[..4], &expected_r);
        assert_eq!(&host_r[4..], &[94; 4]);
        let mut short = [95_i64; 3];
        assert!(grid(&left, &right, &mut q, &mut short).is_err());
        assert_eq!(values::<6>(&q), prior_q);
        assert_eq!(short, [95; 3]);
        grid(&left, &right, &mut host_q, &mut host_r).unwrap();
        assert_eq!(&host_q[..4], &expected_q);
        assert_eq!(&host_r[..4], &expected_r);
    }
    let q_before = values::<6>(&q);
    let r_before = values::<9>(&r);
    let foreign = owner(&self::root(), &[17_i64; 4]);
    assert!(matches!(
        direct(&foreign, &right, &mut q, &mut r),
        Err(PcuExecutionError::Argument(
            global::PcuArgumentError::SessionMismatch
        ))
    ));
    assert_eq!(values::<6>(&q), q_before);
    assert_eq!(values::<9>(&r), r_before);
    global::clear_thread_cache().unwrap();
    drop((foreign, right, q, r));
    global::use_defaults().unwrap();
}
