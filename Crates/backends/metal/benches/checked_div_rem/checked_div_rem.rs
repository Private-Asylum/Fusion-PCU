//! Genuine fourteen-width source/IR/native packed private quotient/remainder controls.
#[rustfmt::skip]
use criterion::{Criterion,criterion_group,criterion_main};
use fusion_pcu_metal::MetalSession;
use std::time::Duration;
#[path = "../support/activity/activity.rs"]
mod activity;
#[path = "../../tests/checked_div_rem/source/source.rs"]
mod source;
#[path = "support/support.rs"]
mod support;

#[cfg(feature = "allocation-census")]
static SCORES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(feature = "allocation-census")]
fn score(_: &pcu_facade::global::PcuInvocationCandidate<'_>) -> i128 {
    SCORES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    1
}

fn benchmark(criterion: &mut Criterion) {
    if !cfg!(target_os = "macos") {
        println!("SKIP: requires actual Metal GPU");
        return;
    }
    if !std::env::args().any(|arg| arg == "--test") {
        activity::guard();
    }
    pcu_facade::global::configure(pcu_facade::global::PcuExecutionPolicy {
        backend: pcu_facade::global::PcuBackendChoice::Metal,
        #[cfg(feature = "allocation-census")]
        score_invocation: Some(score),
        ..pcu_facade::global::PcuExecutionPolicy::default()
    })
    .unwrap();
    let session = MetalSession::open(0).unwrap();
    let backend = session.checked_div_rem_backend();
    macro_rules! extent {
        ($module:ident,$ty:ty,$n:literal) => {{
            support::compare::<$ty, $n>(
                criterion,
                &session,
                0,
                source::$module::direct_prepare::<$n, _>(&backend).unwrap(),
                source::$module::direct::<$n>,
            );
            support::compare::<$ty, $n>(
                criterion,
                &session,
                1,
                source::$module::strict_prepare::<$n, _>(&backend).unwrap(),
                source::$module::strict::<$n>,
            );
            support::compare::<$ty, $n>(
                criterion,
                &session,
                2,
                source::$module::grid_prepare::<$n, _>(&backend).unwrap(),
                source::$module::grid::<$n>,
            );
        }};
    }
    macro_rules! wide_extent {
        ($ty:ty,$n:literal) => {{
            support::compare::<$ty, $n>(
                criterion,
                &session,
                0,
                source::generic::direct_prepare::<$ty, $n, _>(&backend).unwrap(),
                source::generic::direct::<$ty, $n>,
            );
            support::compare::<$ty, $n>(
                criterion,
                &session,
                1,
                source::generic::strict_prepare::<$ty, $n, _>(&backend).unwrap(),
                source::generic::strict::<$ty, $n>,
            );
            support::compare::<$ty, $n>(
                criterion,
                &session,
                2,
                source::generic::grid_prepare::<$ty, $n, _>(&backend).unwrap(),
                source::generic::grid::<$ty, $n>,
            );
        }};
    }
    macro_rules! wide {($($ty:ty),+) => {$ (wide_extent!($ty,65);wide_extent!($ty,4096);)+};}
    wide!(
        u128,
        i128,
        pcu_facade::PcuU256,
        pcu_facade::PcuI256,
        pcu_facade::PcuU512,
        pcu_facade::PcuI512
    );
    macro_rules! all{($($module:ident:$ty:ty),+)=>{$(extent!($module,$ty,65);extent!($module,$ty,4096);)+};}
    all!(u8_source:u8,i8_source:i8,u16_source:u16,i16_source:i16,u32_source:u32,i32_source:i32,u64_source:u64,i64_source:i64);
}
criterion_group! {name=benches;config=Criterion::default().sample_size(30).confidence_level(0.95).warm_up_time(Duration::from_millis(250)).measurement_time(Duration::from_secs(1));targets=benchmark}
criterion_main!(benches);
