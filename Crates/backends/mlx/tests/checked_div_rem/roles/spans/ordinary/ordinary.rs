//! Actual ordinary source shares the retained full-capacity, terminal joint host publication law.
#[rustfmt::skip]
use pcu_facade::{
    global,
    pcu,
    PcuScalar,
    PcuCheckedIntegerDivision,
    PcuTensor,
    PcuExecutionError,
    PcuExecutionFaultKind,
    PcuI256,
    PcuI512,
    PcuU256,
    PcuU512,
};
#[rustfmt::skip]
use super::super::{
    same,
    Sample,
    source,
};
use super::heap;
use std::sync::atomic::{AtomicUsize, Ordering};
static SCORES: AtomicUsize = AtomicUsize::new(0);
fn score(_: &global::PcuInvocationCandidate<'_>) -> i128 {
    SCORES.fetch_add(1, Ordering::Relaxed);
    1
}
#[pcu(crate_path=::pcu_facade)]
fn retain<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
#[pcu(invocations=5,crate_path=::pcu_facade)]
fn scalar_rhs_slice<T: PcuCheckedIntegerDivision>(q: &mut [T], lhs: &[T], r: &mut [T], rhs: &[T]) {
    let id = pcu::context::global_invocation_id();
    let (quotient, remainder) = pcu::checked_div_rem(lhs[id], rhs[0]);
    q[id] = quotient;
    r[id] = remainder;
}
#[pcu(invocations=3,crate_path=::pcu_facade)]
fn scalar_self_slice<T: PcuCheckedIntegerDivision>(q: &mut [T], input: &[T], r: &mut [T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < 5 {
        let (quotient, remainder) = pcu::checked_div_rem(input[0], input[0]);
        q[id] = quotient;
        r[id] = remainder;
        id += stride;
    }
}
fn call<T: Sample>(
    profile: usize,
    left: &PcuTensor<T>,
    right: &PcuTensor<T>,
    q: &mut [T],
    r: &mut [T],
) -> Result<(), PcuExecutionError> {
    match profile {
        0 => source::repeated::<T, 5>(q, left, r),
        1 => source::unread::<T, 5>(&[], r, left, q),
        2 => source::reordered::<T, 5>(right, r, left, q),
        3 => source::grid::<T, 5>(&[], r, left, q),
        4 => scalar_rhs_slice::<T>(q, left, r, right),
        5 => scalar_self_slice::<T>(q, left, r),
        _ => unreachable!(),
    }
}
fn verify<T: Sample>(profile: usize) {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Mlx,
        score_invocation: Some(score),
        ..Default::default()
    })
    .unwrap();
    let banks: Vec<_> = (0..64)
        .map(|bank| {
            let mut a = [T::raw(1); 8];
            let mut b = [T::raw(3); 9];
            for lane in 0..5 {
                a[lane] = T::raw(u64::try_from((bank ^ lane) % 89 + 1).unwrap());
                b[lane] = T::raw(u64::try_from((bank + lane) % 7 + 1).unwrap());
            }
            a[2] = T::minimum();
            a[5..].fill(T::raw(0));
            b[5..].fill(T::raw(0));
            if profile == 5 {
                a[1] = T::raw(0);
            }
            if profile == 4 {
                b[1] = T::raw(0);
            }
            (retain::<T>(&a).unwrap(), retain::<T>(&b).unwrap(), a, b)
        })
        .collect();
    let mut q = [T::raw(91); 7];
    let mut r = [T::raw(91); 8];
    call(profile, &banks[0].0, &banks[0].1, &mut q, &mut r).unwrap();
    let scored = SCORES.load(Ordering::Relaxed);
    fusion_pcu_mlx::reset_div_rem_call_census();
    let mut total = heap::Census::default();
    for (left, right, a, b) in &banks {
        let ((), counts) = heap::measure(|| call(profile, left, right, &mut q, &mut r).unwrap());
        total.alloc_calls += counts.alloc_calls;
        total.realloc_calls += counts.realloc_calls;
        total.dealloc_calls += counts.dealloc_calls;
        total.requested_bytes += counts.requested_bytes;
        for lane in 0..5 {
            let lhs = a[if matches!(profile, 3 | 5) { 0 } else { lane }];
            let rhs = match profile {
                0 | 3 => a[lane],
                1 | 5 => a[0],
                2 => b[lane],
                4 => b[0],
                _ => unreachable!(),
            };
            same(&q[lane..=lane], &[lhs.pcu_checked_div(rhs).unwrap()]);
            same(&r[lane..=lane], &[lhs.pcu_checked_rem(rhs).unwrap()]);
        }
        same(&q[5..], &[T::raw(91); 2]);
        same(&r[5..], &[T::raw(91); 3]);
    }
    assert_eq!(SCORES.load(Ordering::Relaxed), scored);
    let counts = fusion_pcu_mlx::div_rem_call_census();
    assert_eq!(counts.exact_constructor_calls, 0);
    assert_eq!(counts.prefix_constructor_calls, 0);
    assert_eq!(counts.prime_calls, 0);
    assert_eq!(counts.table_symbol_attempts, 0);
    assert_eq!(counts.apply_calls, 64);
    eprintln!(
        "MLX ordinary full-capacity census/{:?}/profile{profile}: calls=64 alloc={} realloc={} dealloc={} requested_bytes={} score_callbacks=0 adapter={counts:?}; resident inputs -> both private terminal reads/checked releases -> joint host publication; SDK-internal heap/JIT unknown",
        T::TYPE,
        total.alloc_calls,
        total.realloc_calls,
        total.dealloc_calls,
        total.requested_bytes
    );
    let old = (q, r);
    let bad = retain::<T>(&[T::raw(0); 8]).unwrap();
    let error = call(profile, &bad, &bad, &mut q, &mut r).unwrap_err();
    match error {
        PcuExecutionError::ArithmeticFault(fault) => {
            assert_eq!(fault.kind, PcuExecutionFaultKind::DivideByZero);
            assert_eq!(fault.invocation_id, 0);
            assert!(!fault.recovered);
        }
        other => panic!("expected checked divide fault, got {other:?}"),
    }
    same(&q, &old.0);
    same(&r, &old.1);
    call(profile, &banks[1].0, &banks[1].1, &mut q, &mut r).unwrap();
    global::clear_thread_cache().unwrap();
}
#[test]
#[ignore = "required actual ordinary MLX full-capacity 64-changing-call caller/adapter proof"]
fn fourteen_width_ordinary_full_capacity_caller_and_adapter_census() {
    macro_rules! run {($($ty:ty),+)=>{$(for profile in 0..6 {verify::<$ty>(profile);})+};}
    run!(
        u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, PcuU256, PcuI256, PcuU512, PcuI512
    );
    global::configure(global::PcuExecutionPolicy::default()).unwrap();
}
