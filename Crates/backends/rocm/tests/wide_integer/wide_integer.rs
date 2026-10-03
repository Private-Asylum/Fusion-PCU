//! Independent arbitrary-precision byte/fault proof through genuine generic source.
extern crate pcu_facade as fusion_pcu;
#[path = "../../benches/wide_integer/oracle/oracle.rs"]
mod oracle;
#[path = "../../benches/wide_integer/source/source.rs"]
#[allow(dead_code)] // All source paths are shared with the canonical benchmark.
mod source;
#[rustfmt::skip]
use fusion_pcu::{global,PcuExecutionError,PcuExecutionFaultKind,PcuNumericalMode,PcuTensor};
use oracle::Format;
fn host<T: Format, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
    op: u32,
    grid: bool,
) -> Result<(), PcuExecutionError> {
    match (op, grid) {
        (0, false) => source::add::direct::<T, N>(left, right, output),
        (0, true) => source::add::grid::<T, N>(left, right, output),
        (1, false) => source::sub::direct::<T, N>(left, right, output),
        (1, true) => source::sub::grid::<T, N>(left, right, output),
        (2, false) => source::mul::direct::<T, N>(left, right, output),
        (2, true) => source::mul::grid::<T, N>(left, right, output),
        _ => unreachable!(),
    }
}
fn resident<T: Format, const N: usize>(
    left: &PcuTensor<T>,
    right: &PcuTensor<T>,
    output: &mut PcuTensor<T>,
    op: u32,
) -> Result<(), PcuExecutionError> {
    match op {
        0 => source::add::grid::<T, N>(left, right, output),
        1 => source::sub::grid::<T, N>(left, right, output),
        2 => source::mul::grid::<T, N>(left, right, output),
        _ => unreachable!(),
    }
}
fn fault(result: &Result<(), PcuExecutionError>, index: u64, code: u32) {
    let kind = if code == 3 {
        PcuExecutionFaultKind::ArithmeticOverflow
    } else {
        PcuExecutionFaultKind::ArithmeticUnderflow
    };
    assert!(
        matches!(result,Err(PcuExecutionError::ArithmeticFault(f)) if f.invocation_id==index&&f.kind==kind&&!f.recovered),
        "{result:?}"
    );
}
fn bits<T: Format>(actual: &[T], expected: &[T]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert_eq!(a.encode_le().as_ref(), b.encode_le().as_ref());
    }
}
fn reference<T: Format>() {
    for op in 0..3 {
        let rows = oracle::rows::<T>(op);
        for &(a, b, expected, code) in &rows {
            let reference = match op {
                0 => a.pcu_checked_add(b),
                1 => a.pcu_checked_sub(b),
                2 => a.pcu_checked_mul(b),
                _ => unreachable!(),
            };
            if code == 0 {
                assert_eq!(
                    reference.unwrap().encode_le().as_ref(),
                    expected.encode_le().as_ref()
                );
            } else {
                assert_eq!(
                    reference.unwrap_err(),
                    if code == 3 {
                        PcuExecutionFaultKind::ArithmeticOverflow
                    } else {
                        PcuExecutionFaultKind::ArithmeticUnderflow
                    }
                );
            }
        }
    }
}
#[test]
fn independent_arbitrary_precision_goldens_agree_with_core() {
    reference::<i128>();
    reference::<u128>();
    reference::<fusion_pcu::PcuI256>();
    reference::<fusion_pcu::PcuU256>();
    reference::<fusion_pcu::PcuI512>();
    reference::<fusion_pcu::PcuU512>();
}
#[allow(clippy::too_many_lines)] // One format proves exact bits, fatal publication, and retry together.
fn format<T: Format>() {
    const N: usize = 65;
    for op in 0..3 {
        let rows = oracle::rows::<T>(op);
        assert_eq!(rows.iter().filter(|row| row.3 == 0).count(), 65);
        for phase in [0, 17, 41] {
            let (a, b, want) = oracle::inputs::<T>(N, phase, op);
            let mut out = vec![T::sentinel(); N + 2];
            host::<T, N>(&a, &b, &mut out, op, false).unwrap();
            oracle::verify(&want, &out);
            host::<T, N>(&a, &b, &mut out, op, true).unwrap();
            oracle::verify(&want, &out);
            let neutral = T::small(u8::from(op == 2));
            let mut broadcast = vec![T::sentinel(); N + 2];
            match op {
                0 => source::add::broadcast::<T, N>(&a, &neutral, &mut broadcast),
                1 => source::sub::broadcast::<T, N>(&a, &neutral, &mut broadcast),
                2 => source::mul::broadcast::<T, N>(&a, &neutral, &mut broadcast),
                _ => unreachable!(),
            }
            .unwrap();
            oracle::verify(&a, &broadcast);
        }
        let (mut a, mut b, want) = oracle::inputs::<T>(N, 7, op);
        let mut out = vec![T::sentinel(); N + 2];
        host::<T, N>(&a, &b, &mut out, op, true).unwrap();
        let previous = out.clone();
        let mut ready = source::identity(out.as_slice()).unwrap();
        for &(bad_left, bad_right, _, code) in rows.iter().filter(|row| row.3 != 0) {
            a[5] = bad_left;
            b[5] = bad_right;
            a[41] = bad_left;
            b[41] = bad_right;
            fault(&host::<T, N>(&a, &b, &mut out, op, true), 5, code);
            bits(&out, &previous);
            let ra = source::identity(a.as_slice()).unwrap();
            let rb = source::identity(b.as_slice()).unwrap();
            let mut rq = source::identity(previous.as_slice()).unwrap();
            fault(&resident::<T, N>(&ra, &rb, &mut rq, op), 5, code);
            assert!(matches!(
                rq.read_into(&mut out),
                Err(PcuExecutionError::Argument(
                    global::PcuArgumentError::ResidentValueDiscarded
                ))
            ));
            assert!(source::identity(&rq).is_err());
            assert!(resident::<T, N>(&ra, &rb, &mut rq, op).is_err());
            let (good_a, good_b, _) = oracle::inputs::<T>(N, 7, op);
            let ga = source::identity(good_a.as_slice()).unwrap();
            let gb = source::identity(good_b.as_slice()).unwrap();
            let mut fresh = source::identity(previous.as_slice()).unwrap();
            resident::<T, N>(&ga, &gb, &mut fresh, op).unwrap();
            fresh.read_into(&mut out).unwrap();
            oracle::verify(&want, &out);
        }
        let short = &a[..N - 1];
        assert!(host::<T, N>(short, &b, &mut out, op, false).is_err());
        bits(&out, &previous);
        assert!(
            match op {
                0 => source::add::direct::<T, N>(short, &b, &mut ready),
                1 => source::sub::direct::<T, N>(short, &b, &mut ready),
                2 => source::mul::direct::<T, N>(short, &b, &mut ready),
                _ => unreachable!(),
            }
            .is_err()
        );
        let mut restored = vec![T::sentinel(); N + 2];
        ready.read_into(&mut restored).unwrap();
        bits(&restored, &previous);
    }
}
fn configure(mode: PcuNumericalMode) {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Rocm,
        device: Some(0),
        block_size: 256,
        numerical_mode: mode,
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
}
macro_rules! fixtures {($($name:ident,$ty:ty;)+)=>{$(
 #[test] #[ignore="requires authorized actual GPU, correctness only, run serially"] fn $name(){for mode in [PcuNumericalMode::Boundary,PcuNumericalMode::Strict]{configure(mode);format::<$ty>();}}
)+};}
fixtures!(signed128,i128;unsigned128,u128;signed256,fusion_pcu::PcuI256;unsigned256,fusion_pcu::PcuU256;signed512,fusion_pcu::PcuI512;unsigned512,fusion_pcu::PcuU512;);
