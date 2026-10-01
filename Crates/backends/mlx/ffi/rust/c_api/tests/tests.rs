//! Bounded failure fixtures use a separately instrumented native image, never production.

#[rustfmt::skip]
use std::{
    path::Path,
    rc::Rc,
};
use crate::MlxError;
#[rustfmt::skip]
use super::{
    abi::Opaque,
    api::Api,
    owner::Owner,
};

type SetFailpoint = unsafe extern "C" fn(i32);
type OwnerCount = unsafe extern "C" fn() -> usize;

#[test]
#[ignore = "requires a separate unknown-ABI sentinel image and one bounded subprocess"]
fn instrumented_unknown_abi_rejects_before_layout_dependent_foreign_calls() {
    let path = std::env::var_os("PCU_MLX_ABI_REJECT_LIBRARY")
        .expect("set the separate zero-argument unknown-ABI sentinel image");
    // The sentinel exports a stable zero-argument ABI query returning99. Its manifest
    // function exits91 if invoked; no GPU SDK is linked and no incompatible layout is called.
    let error = Api::load(Path::new(&path))
        .err()
        .expect("unknown family must reject");
    assert_eq!(
        error,
        MlxError::Abi("unsupported C safety ABI family".into())
    );
}

struct Fixture {
    api: Rc<Api>,
    set: SetFailpoint,
    owners: OwnerCount,
    frees: OwnerCount,
}

impl Fixture {
    fn load() -> Self {
        let path = std::env::var_os("PCU_MLX_FAULT_LIBRARY")
            .expect("set separately instrumented direct-C fault image");
        let api = Api::load(Path::new(&path)).unwrap();
        // SAFETY: only the separately built/recorded bounded fault image exposes these exact
        // test signatures. Production dylib exposes none. Api retains every pointer's image.
        let (set, owners, frees) = unsafe {
            (
                *api.library
                    .get::<SetFailpoint>(b"_pcu_mlx_test_failpoint_set\0")
                    .unwrap(),
                *api.library
                    .get::<OwnerCount>(b"_pcu_mlx_test_array_owners\0")
                    .unwrap(),
                *api.library
                    .get::<OwnerCount>(b"_pcu_mlx_test_free_calls\0")
                    .unwrap(),
            )
        };
        Self {
            api,
            set,
            owners,
            frees,
        }
    }
    fn counts(&self) -> (usize, usize) {
        // SAFETY: retained instrumented image, read-only counters; one test per subprocess.
        unsafe { ((self.owners)(), (self.frees)()) }
    }
    fn fail(&self, mode: i32) {
        // SAFETY: exact controlled fixture modes0/1/2 only, caller-thread instrumentation.
        unsafe { (self.set)(mode) };
    }
}

#[test]
#[ignore = "requires separate instrumented C image and one bounded subprocess"]
fn instrumented_read_failure_preserves_host_and_actual_quarantined_holder() {
    let fixture = Fixture::load();
    let session = fixture.api.open(0).unwrap();
    let array = session.upload([2, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let before = fixture.counts();
    let mut host = [-719.0_f32; 4];
    fixture.fail(1);
    assert!(matches!(
        array.read(&mut host),
        Err(MlxError::CompletionUnknown(_))
    ));
    fixture.fail(0);
    assert_eq!(host.map(f32::to_bits), [(-719.0_f32).to_bits(); 4]);
    assert!(matches!(
        session.ensure_ready(),
        Err(MlxError::CompletionUnknown(_))
    ));
    let after = fixture.counts();
    assert_eq!(
        after.0,
        before.0 + 1,
        "actual cloned C holder is quarantined"
    );
    assert_eq!(
        after.1, before.1,
        "failed materialization did not free accessed holder"
    );
    drop((array, session));
    assert!(
        fixture.counts().0 > 0,
        "quarantine retains native backing after safe owners drop"
    );
}

#[test]
#[ignore = "requires separate instrumented C image and one bounded subprocess"]
fn instrumented_eval_failure_retains_actual_pending_inputs_output_and_session() {
    let fixture = Fixture::load();
    let session = fixture.api.open(0).unwrap();
    let left = session.upload([2, 2], &[1.0; 4]).unwrap();
    let right = session.upload([2, 2], &[2.0; 4]).unwrap();
    let before = fixture.counts();
    fixture.fail(2);
    assert!(matches!(
        session.matmul(&left, &right),
        Err(MlxError::CompletionUnknown(_))
    ));
    fixture.fail(0);
    assert!(matches!(
        session.ensure_ready(),
        Err(MlxError::CompletionUnknown(_))
    ));
    let after = fixture.counts();
    assert_eq!(
        after.0,
        before.0 + 3,
        "two native input clones and output are retained"
    );
    assert_eq!(
        after.1, before.1,
        "unknown completion released no pending holder"
    );
    assert!(matches!(
        session.matmul(&left, &right),
        Err(MlxError::CompletionUnknown(_))
    ));
    drop((left, right, session));
    assert!(
        fixture.counts().0 >= 3,
        "actual pending backing survives user owner drops"
    );
}

#[test]
#[ignore = "requires exact production C library and an external GPU activity check"]
fn native_shape_error_is_contained_and_valid_submission_retries() {
    let api = Api::load_default().unwrap();
    native_fault_retry(&api);
}

fn native_fault_retry(api: &Rc<Api>) {
    let session = api.open(0).unwrap();
    let a = session.upload([2, 3], &[1.0; 6]).unwrap();
    let bad = session.upload([4, 2], &[2.0; 8]).unwrap();
    let mut output = Owner::empty(Rc::clone(api), api.array_free);
    // SAFETY: live same-session dense F32 arrays with incompatible inner dimensions.
    // MLX rejects shape construction before any kernel is queued; no OOB GPU work is run.
    let error = api
        .status(|| unsafe {
            (api.matmul)(
                &raw mut output.raw,
                a.owner.raw,
                bad.owner.raw,
                session.0.stream.raw,
            )
        })
        .unwrap_err();
    assert!(error.to_string().contains("matmul"));
    assert!(output.raw.is_empty());
    let b = session.upload([3, 2], &[2.0; 6]).unwrap();
    let c = session.matmul(&a, &b).unwrap();
    let mut host = [0.0; 4];
    c.read(&mut host).unwrap();
    assert_eq!(host.map(f32::to_bits), [6.0_f32.to_bits(); 4]);
}
