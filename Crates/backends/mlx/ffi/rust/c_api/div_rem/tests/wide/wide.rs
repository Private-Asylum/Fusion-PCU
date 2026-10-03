//! Private wide synthesis oracle; public dispatch remains primitive eight-width only.
#[rustfmt::skip]
use super::{
    CheckedDivRem,
    EncodedArray,
    Session,
    PcuScalar,
    PcuHostArgument,
    PcuBindingRef,
    PcuCheckedIntegerDivision,
    PcuExecutionFaultKind,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuI256,
    PcuU256,
    PcuI512,
    PcuU512,
};
trait Wide: PcuCheckedIntegerDivision {
    fn raw(bytes: [u8; 64]) -> Self;
}
macro_rules! samples {
    ($($ty:ty=>$bytes:literal;)+) => {$(impl Wide for $ty {
        fn raw(bytes:[u8;64])->Self {Self::decode_le(bytes[..$bytes].try_into().unwrap())}
    })+};
}
samples! {u128=>16;i128=>16;PcuU256=>32;PcuI256=>32;PcuU512=>64;PcuI512=>64;}
fn qualify<T: Wide>(session: &Session, left: &[T], right: &[T], broadcast: [bool; 2]) {
    let count = left.len().max(right.len());
    let left_arg = PcuHostArgument::read(PcuBindingRef::new(0, 0), left);
    let right_arg = PcuHostArgument::read(PcuBindingRef::new(0, 1), right);
    let a = EncodedArray::upload(session, T::TYPE, left.len(), left_arg.bytes()).unwrap();
    let b = EncodedArray::upload(session, T::TYPE, right.len(), right_arg.bytes()).unwrap();
    let prepared = CheckedDivRem::prepare_wide_proof(
        session,
        T::TYPE,
        count,
        [left.len(), right.len()],
        broadcast,
    )
    .unwrap();
    let ([q, r], records) = prepared.execute_recorded([&a, &b]).unwrap();
    let statuses = prepared.status_records(&records).unwrap();
    let mut qb = vec![0; count * T::ENCODED_SIZE];
    let mut rb = vec![0; count * T::ENCODED_SIZE];
    q.read(&mut qb).unwrap();
    r.read(&mut rb).unwrap();
    for index in 0..count {
        let lhs = left[if broadcast[0] { 0 } else { index }];
        let rhs = right[if broadcast[1] { 0 } else { index }];
        let (want_q, want_r, fault) = match (lhs.pcu_checked_div(rhs), lhs.pcu_checked_rem(rhs)) {
            (Ok(q), Ok(r)) => (q, r, 0),
            (Err(q), Err(r)) => {
                assert_eq!(q, r);
                (
                    T::raw([0; 64]),
                    T::raw([0; 64]),
                    match q {
                        PcuExecutionFaultKind::DivideByZero => 4,
                        PcuExecutionFaultKind::SignedDivisionOverflow => 1,
                        other => panic!("unexpected wide division fault {other:?}"),
                    },
                )
            }
            _ => panic!("independent wide quotient/remainder oracle disagreed"),
        };
        let span = index * T::ENCODED_SIZE..(index + 1) * T::ENCODED_SIZE;
        assert_eq!(
            &qb[span.clone()],
            want_q.encode_le().as_ref(),
            "wide quotient {:?} lane{index}",
            T::TYPE
        );
        assert_eq!(
            &rb[span],
            want_r.encode_le().as_ref(),
            "wide remainder {:?} lane{index}",
            T::TYPE
        );
        assert_eq!(
            statuses[index],
            fault,
            "wide status {:?} lane{index}",
            T::TYPE
        );
    }
    q.release().unwrap();
    r.release().unwrap();
    records.release().unwrap();
    a.release().unwrap();
    b.release().unwrap();
    eprintln!(
        "wide div/rem bit/status oracle {:?}: count={count} broadcast={broadcast:?}",
        T::TYPE
    );
}
fn raw_edge<T: Wide>(kind: usize) -> T {
    let width = T::ENCODED_SIZE;
    let mut bytes = [0; 64];
    match kind {
        0 => {}
        1 => bytes[0] = 1,
        2 => bytes[0] = 2,
        3 => bytes[0] = 3,
        4 => {
            bytes[..width].fill(255);
            bytes[width - 1] = 127;
        }
        5 => bytes[width - 1] = 128,
        6 => {
            bytes[width - 1] = 128;
            bytes[0] = 1;
        }
        7 => {
            bytes[..width].fill(255);
            bytes[0] = 254;
        }
        _ => bytes[..width].fill(255),
    }
    T::raw(bytes)
}
fn run<T: Wide>(session: &Session) {
    let mut left = Vec::with_capacity(4096);
    let mut right = Vec::with_capacity(4096);
    for a in 0..9 {
        for b in 0..9 {
            left.push(raw_edge::<T>(a));
            right.push(raw_edge::<T>(b));
        }
    }
    let mut seed = 0x243f_6a88_85a3_08d3_u64;
    while left.len() < 4096 {
        let mut a = [0; 64];
        let mut b = [0; 64];
        for (aw, bw) in a[..T::ENCODED_SIZE]
            .as_chunks_mut::<8>()
            .0
            .iter_mut()
            .zip(b[..T::ENCODED_SIZE].as_chunks_mut::<8>().0.iter_mut())
        {
            seed = seed.wrapping_mul(0x5851_f42d_4c95_7f2d).wrapping_add(1);
            aw.copy_from_slice(&seed.to_le_bytes());
            seed = seed.wrapping_mul(0x5851_f42d_4c95_7f2d).wrapping_add(1);
            bw.copy_from_slice(&seed.to_le_bytes());
        }
        // Vary divisor magnitude as well as every high input limb, exercising wide quotient words.
        let kept = left.len() % T::ENCODED_SIZE + 1;
        b[kept..T::ENCODED_SIZE].fill(0);
        left.push(T::raw(a));
        right.push(T::raw(b));
    }
    qualify(session, &left, &right, [false; 2]);
    for kind in [0, 1, 5, 8] {
        let scalar = raw_edge::<T>(kind);
        qualify(session, &[scalar], &right[..1024], [true, false]);
        qualify(session, &left[..1024], &[scalar], [false, true]);
    }
}
#[test]
#[ignore = "Requires actual private MLX wide U32-limb quotient/remainder/status synthesis."]
fn native_six_wide_div_rem_bit_and_fault_oracle() {
    let runtime = crate::MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let native = session.native_for_binary_proof();
    run::<u128>(native);
    run::<i128>(native);
    run::<PcuU256>(native);
    run::<PcuI256>(native);
    run::<PcuU512>(native);
    run::<PcuI512>(native);
}
