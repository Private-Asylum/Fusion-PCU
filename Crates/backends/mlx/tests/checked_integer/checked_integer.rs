//! Public native integer control: exact host/resident prefixes, observable Clamp and old owners.
#[path = "cold/cold.rs"]
mod cold;
#[path = "graph/graph.rs"]
mod graph;
#[path = "offers/offers.rs"]
mod offers;
#[path = "ordinary/ordinary.rs"]
mod ordinary;
#[path = "prefix/prefix.rs"]
mod prefix;
#[path = "prepared/prepared.rs"]
mod prepared;
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxRuntime,
    MlxSession,
    MlxError,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuDispatchIntegerBinaryOp as Op,
    PcuRangePolicy as Range,
    PcuI256,
    PcuU256,
    PcuI512,
    PcuU512,
};
#[path = "support/support.rs"]
mod support;
#[rustfmt::skip]
use support::{
    Sample,
    same,
    expected,
};
#[allow(clippy::too_many_lines)] // One public-control oracle retains type/extent/session preflight and immutable publication law.
fn qualify<T: Sample>(session: &MlxSession, foreign: &MlxSession) {
    prepared::qualify::<T>(session);
    prepared::mixed::<T>(session, foreign);
    let sentinel = T::small(7);
    let lhs = [T::zero(), T::small(1), T::max(), T::min(), T::max()];
    let rhs = [T::small(1), T::small(2), T::small(1), T::max(), T::small(2)];
    for broadcast in [[false, false], [true, false], [false, true]] {
        let left = if broadcast[0] { &lhs[..1] } else { &lhs[..] };
        let right = if broadcast[1] { &rhs[..1] } else { &rhs[..] };
        let a = session.upload_encoded(left).unwrap();
        let b = session.upload_encoded(right).unwrap();
        let other = foreign.upload_encoded(left).unwrap();
        for range in [Range::Reject, Range::Clamp] {
            for op in [Op::Add, Op::Sub, Op::Mul] {
                let mut control = session
                    .prepare_checked_integer_control(
                        T::TYPE,
                        op,
                        range,
                        5,
                        [left.len(), right.len()],
                        broadcast,
                    )
                    .unwrap();
                let (wanted, fault) = expected(left, right, op, range, broadcast);
                let result = control.execute_resident([&a, &b]);
                if fault.is_some_and(|fault| !fault.recovered) {
                    assert_eq!(result.err(), fault.map(MlxError::Arithmetic));
                } else {
                    let (owner, recovered) = result.unwrap().into_parts();
                    assert_eq!(recovered, fault);
                    let mut output = [sentinel; 7];
                    owner.read_into(&mut output).unwrap();
                    same(&output[..5], &wanted);
                    same(&output[5..], &[sentinel; 2]);
                    let next = control.execute_encoded([left, right]).unwrap();
                    drop(next);
                    owner.read_into(&mut output).unwrap();
                    same(&output[..5], &wanted);
                }
                let mut output = [sentinel; 7];
                let published = control.call([left, right], &mut output);
                assert_eq!(published.err(), fault.map(MlxError::Arithmetic));
                if fault.is_some_and(|fault| !fault.recovered) {
                    same(&output, &[sentinel; 7]);
                } else {
                    same(&output[..5], &wanted);
                    same(&output[5..], &[sentinel; 2]);
                }
                assert!(matches!(
                    control.execute_resident([&other, &b]),
                    Err(MlxError::ForeignSession)
                ));
                assert!(!control.last_call_may_have_written());
                assert!(
                    control
                        .execute_encoded::<f32>([&[0.0; 5], &[0.0; 5]])
                        .is_err()
                );
                assert!(!control.last_call_may_have_written());
                let mut short = [sentinel; 4];
                assert!(control.call([left, right], &mut short).is_err());
                same(&short, &[sentinel; 4]);
                assert!(!control.last_call_may_have_written());
                let zero = [T::zero(); 5];
                control
                    .call([&zero[..left.len()], &zero[..right.len()]], &mut output)
                    .unwrap();
                same(&output[..5], &zero);
                same(&output[5..], &[sentinel; 2]);
                let mut old_left = vec![sentinel; left.len()];
                a.read_into(&mut old_left).unwrap();
                same(&old_left, left);
                let mut old_right = vec![sentinel; right.len()];
                b.read_into(&mut old_right).unwrap();
                same(&old_right, right);
            }
        }
    }
}
#[test]
#[ignore = "Requires actual fourteen-width MLX native host/resident/private integer control qualification."]
fn fourteen_width_public_control_publication_and_preflight() {
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let foreign = runtime.open_gpu(0).unwrap();
    qualify::<u8>(&session, &foreign);
    qualify::<i8>(&session, &foreign);
    qualify::<u16>(&session, &foreign);
    qualify::<i16>(&session, &foreign);
    qualify::<u32>(&session, &foreign);
    qualify::<i32>(&session, &foreign);
    qualify::<u64>(&session, &foreign);
    qualify::<i64>(&session, &foreign);
    qualify::<u128>(&session, &foreign);
    qualify::<i128>(&session, &foreign);
    qualify::<PcuU256>(&session, &foreign);
    qualify::<PcuI256>(&session, &foreign);
    qualify::<PcuU512>(&session, &foreign);
    qualify::<PcuI512>(&session, &foreign);
}
