//! Full-width independent quotient/remainder bytes and transactional genuine source calls.
#[path = "oracle/oracle.rs"]
mod oracle;
#[path = "source/source.rs"]
mod source;
use oracle::Wide;
use fusion_pcu_cpu::PcuCpuHostBackend;
#[rustfmt::skip]
use pcu_facade::{
    global,
    PcuCheckedIntegerDivision,
    PcuExecutionFaultKind,
    PcuI256,
    PcuU256,
    PcuI512,
    PcuU512,
};
fn configure() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
}
fn golden<T: Wide + PcuCheckedIntegerDivision>() {
    configure();
    let mut prepared = source::direct_prepare::<T, 1, _>(&PcuCpuHostBackend::scalar()).unwrap();
    let sentinel = oracle::small::<T>(77);
    let vectors = oracle::goldens::<T>();
    assert!(vectors.len() >= 90);
    for (left, right, expected) in vectors {
        assert_eq!(left.pcu_div_rem_domain(right), expected.map(|_| ()));
        assert_eq!(oracle::domain(left, right), expected.map(|_| ()));
        assert_eq!(
            oracle::evaluate(left, right),
            expected,
            "independent byte oracle vs BigInt"
        );
        let mut q = [sentinel; 4];
        let mut r = q;
        let result = prepared(&[left], &[right], &mut q, &mut r);
        match expected {
            Ok((quotient, remainder)) => {
                result.unwrap();
                assert_eq!((q[0], r[0]), (quotient, remainder));
            }
            Err(kind) => {
                let fault = result.unwrap_err().fault().unwrap();
                assert_eq!(
                    (fault.kind, fault.invocation_id, fault.recovered),
                    (kind, 0, false)
                );
                assert_eq!((q, r), ([sentinel; 4], [sentinel; 4]));
            }
        }
        assert_eq!((&q[1..], &r[1..]), (&[sentinel; 3][..], &[sentinel; 3][..]));
    }
    let valid: Vec<_> = oracle::goldens::<T>()
        .into_iter()
        .filter(|row| row.2.is_ok())
        .take(65)
        .collect();
    assert_eq!(valid.len(), 65);
    let left: Vec<_> = valid.iter().map(|row| row.0).collect();
    let mut right: Vec<_> = valid.iter().map(|row| row.1).collect();
    let mut q = vec![sentinel; 68];
    let mut r = q.clone();
    let expected: Vec<_> = valid.iter().map(|row| row.2.unwrap()).collect();
    let mut grid = source::grid_prepare::<T, 65, _>(&PcuCpuHostBackend::scalar()).unwrap();
    for route in 0..3 {
        match route {
            0 => source::direct::<T, 65>(&left, &right, &mut q, &mut r).unwrap(),
            1 => source::grid::<T, 65>(&left, &right, &mut q, &mut r).unwrap(),
            _ => grid(&left, &right, &mut q, &mut r).unwrap(),
        }
        for (lane, pair) in expected.iter().enumerate() {
            assert_eq!((q[lane], r[lane]), *pair);
        }
        assert_eq!(
            (&q[65..], &r[65..]),
            (&[sentinel; 3][..], &[sentinel; 3][..])
        );
    }
    for lane in [0, 2, 64] {
        let old = right[lane];
        right[lane] = oracle::small(0);
        let before = (q.clone(), r.clone());
        let error = source::grid::<T, 65>(&left, &right, &mut q, &mut r).unwrap_err();
        let fault = error.arithmetic_fault().unwrap();
        assert_eq!(
            (fault.kind, fault.invocation_id, fault.recovered),
            (PcuExecutionFaultKind::DivideByZero, lane as u64, false)
        );
        assert_eq!((&q, &r), (&before.0, &before.1));
        right[lane] = old;
        source::grid::<T, 65>(&left, &right, &mut q, &mut r).unwrap();
    }
    let before = (q.clone(), r.clone());
    assert!(source::direct::<T, 65>(&left, &right, &mut q, &mut r[..64]).is_err());
    assert_eq!((&q, &r), (&before.0, &before.1));
    source::direct::<T, 65>(&left, &right, &mut q, &mut r).unwrap();
}
macro_rules! golden_test {
    ($name:ident, $ty:ty) => {
        #[test]
        fn $name() {
            golden::<$ty>();
        }
    };
}
golden_test!(i128_full_width, i128);
golden_test!(u128_full_width, u128);
golden_test!(i256_full_width, PcuI256);
golden_test!(u256_full_width, PcuU256);
golden_test!(i512_full_width, PcuI512);
golden_test!(u512_full_width, PcuU512);
fn endpoints<T: Wide + PcuCheckedIntegerDivision>() {
    let one = oracle::small(1);
    let zero = oracle::small(0);
    let mut minus_one = [255; 64];
    if !T::SIGNED {
        minus_one = [0; 64];
        minus_one[0] = 1;
    }
    let values = [
        zero,
        one,
        oracle::small(3),
        oracle::minimum::<T>(),
        oracle::maximum::<T>(),
        T::from_bytes(minus_one),
    ];
    for left in values {
        for right in values {
            let expected = oracle::evaluate(left, right);
            assert_eq!(left.pcu_checked_div_rem(right), expected);
            assert_eq!(left.pcu_div_rem_domain(right), expected.map(|_| ()));
            assert_eq!(oracle::domain(left, right), expected.map(|_| ()));
        }
    }
}
#[test]
fn fourteen_carrier_joint_core_endpoint_oracle() {
    endpoints::<i8>();
    endpoints::<u8>();
    endpoints::<i16>();
    endpoints::<u16>();
    endpoints::<i32>();
    endpoints::<u32>();
    endpoints::<i64>();
    endpoints::<u64>();
    endpoints::<i128>();
    endpoints::<u128>();
    endpoints::<PcuI256>();
    endpoints::<PcuU256>();
    endpoints::<PcuI512>();
    endpoints::<PcuU512>();
}

fn profile<T: Wide + PcuCheckedIntegerDivision>(id: u32) {
    use pcu_facade::PcuHostKernelBackend;
    let bindings = source::direct_bindings::<T>();
    let builder = source::direct_ir::<T, 65>(&bindings).unwrap();
    let mut kernel = builder.ir();
    let backend = fusion_pcu_cpu::PcuCpuCheckedDivRem::<T>::new();
    let prepared = backend.prepare_host_kernel(&kernel).unwrap();
    assert_eq!((prepared.local_id(), prepared.scalar_type()), (id, T::TYPE));
    kernel.numerical_requirements.range_policy = pcu_facade::PcuRangePolicy::Clamp;
    assert!(backend.prepare_host_kernel(&kernel).is_err());
    kernel.numerical_requirements.range_policy = pcu_facade::PcuRangePolicy::Reject;
    kernel
        .numerical_requirements
        .numerical_options
        .reproducibility = pcu_facade::PcuReproducibility::PortableV1;
    let portable = backend.prepare_host_kernel(&kernel).unwrap();
    assert_eq!(
        (portable.local_id(), portable.implementation_revision()),
        (12296 + id - 512, 1)
    );
}
#[test]
fn distinct_cold_wide_ids_and_unproved_policy_rejection() {
    profile::<i128>(512);
    profile::<u128>(513);
    profile::<PcuI256>(514);
    profile::<PcuU256>(515);
    profile::<PcuI512>(516);
    profile::<PcuU512>(517);
}

#[test]
fn exhaustive_eight_bit_domains_match_independent_sixteen_bit_pairs() {
    for a in 0..=u8::MAX {
        for b in 0..=u8::MAX {
            let unsigned = if b == 0 {
                Err(PcuExecutionFaultKind::DivideByZero)
            } else {
                Ok(())
            };
            assert_eq!(a.pcu_div_rem_domain(b), unsigned);
            assert_eq!(oracle::domain(a, b), unsigned);
            let left = i8::from_ne_bytes([a]);
            let right = i8::from_ne_bytes([b]);
            let signed = if right == 0 {
                Err(PcuExecutionFaultKind::DivideByZero)
            } else if i16::from(left) / i16::from(right) > i16::from(i8::MAX) {
                Err(PcuExecutionFaultKind::SignedDivisionOverflow)
            } else {
                Ok(())
            };
            assert_eq!(left.pcu_div_rem_domain(right), signed);
            assert_eq!(oracle::domain(left, right), signed);
        }
    }
}
