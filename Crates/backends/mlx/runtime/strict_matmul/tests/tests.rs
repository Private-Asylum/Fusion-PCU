//! Independent ordered scalar reference; actual MLX immutable publication and fault ordinals.
use super::*;
use crate::MlxRuntime;
#[rustfmt::skip]
use fusion_pcu::{
    PcuCheckedFloat,
    PcuExecutionFault,
    PcuExecutionFaultKind,
};
use fusion_pcu::dialect::tensor::TensorStrictFaultDomain;
#[test]
fn strict_matmul_event_domain_rejects_impossible_status_and_wrong_extent() {
    let domain = TensorStrictFaultDomain::matmul(
        PcuScalarType::F64,
        2,
        3,
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
    )
    .unwrap();
    assert!(crate::ffi::StrictMatMul::validate_domain(&[0; 4], domain).is_ok());
    assert!(crate::ffi::StrictMatMul::validate_domain(&[0; 3], domain).is_err());
    for records in [
        [1, 0, 0, 0],
        [0, 2, 0, 0],
        [1, 1, 0, 0],
        [3, 4, 0, 0],
        [0, 0x101, 0, 0],
        [6, 1, 0, 0],
        [0, 0, 5, 1],
        [0, 0, 12, 1],
        [0, 1, 6, u32::MAX],
    ] {
        assert!(crate::ffi::StrictMatMul::validate_domain(&records, domain).is_err());
    }
}
fn bytes<T: PcuCheckedFloat>(values: &[T]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.encode_le().as_ref().to_vec())
        .collect()
}
fn oracle<T: PcuCheckedFloat>(
    left: &[T; 9],
    right: &[T; 6],
    zero: T,
    policy: PcuFloatUnderflowPolicy,
) -> Result<Vec<T>, PcuExecutionFault> {
    let mut output = Vec::new();
    for cell in 0..6 {
        let mut accumulator = zero;
        for reduction in 0..3 {
            let base = u64::try_from(cell * 6 + reduction * 2).unwrap();
            let fault = |kind: PcuExecutionFaultKind, invocation_id| PcuExecutionFault {
                kind,
                invocation_id,
                recovered: false,
            };
            let product = left[cell / 2 * 3 + reduction]
                .pcu_checked_mul_with_policy(right[reduction * 2 + cell % 2], policy)
                .map_err(|kind| fault(kind, base))?;
            accumulator = accumulator
                .pcu_checked_add_with_policy(product, policy)
                .map_err(|kind| fault(kind, base + 1))?;
        }
        output.push(accumulator);
    }
    Ok(output)
}
fn verify<T: PcuCheckedFloat>(
    session: &MlxSession,
    map: &MlxPreparedStrictMatMul,
    left: &[T; 9],
    right: &[T; 6],
    zero: T,
    policy: PcuFloatUnderflowPolicy,
) {
    let a = session.upload_encoded(left).unwrap();
    let b = session.upload_encoded(right).unwrap();
    let actual = map.execute_resident(&a, &b);
    match oracle(left, right, zero, policy) {
        Ok(expected) => {
            let result = actual.unwrap();
            assert!(result.recovered_fault().is_none());
            let mut output = [zero; 8];
            result.output().read_into(&mut output).unwrap();
            assert_eq!(bytes(&output[..6]), bytes(&expected));
            assert_eq!(bytes(&output[6..]), bytes(&[zero; 2]));
        }
        Err(expected) => {
            assert!(matches!(actual,Err(MlxError::Arithmetic(fault)) if fault==expected));
        }
    }
    let mut old = [zero; 9];
    a.read_into(&mut old).unwrap();
    assert_eq!(bytes(&old), bytes(left));
    let mut old = [zero; 6];
    b.read_into(&mut old).unwrap();
    assert_eq!(bytes(&old), bytes(right));
}
fn native<T: PcuCheckedFloat>(
    session: &MlxSession,
    foreign: &MlxSession,
    zero: T,
    edges: &[T],
    from_bits: impl Fn(u64) -> T,
) {
    for policy in [
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
    ] {
        let map = session
            .prepare_strict_matmul(T::TYPE, policy, [3, 3, 2])
            .unwrap();
        for offset in 0..edges.len() {
            let left = core::array::from_fn(|index| edges[(offset + index) % edges.len()]);
            let right = core::array::from_fn(|index| edges[(offset + index * 3) % edges.len()]);
            verify(session, &map, &left, &right, zero, policy);
        }
        let mut seed = 0x739a_93b1_e518_2525u64;
        for _ in 0..128 {
            let mut next = || {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                from_bits(seed)
            };
            let left = core::array::from_fn(|_| next());
            let right = core::array::from_fn(|_| next());
            verify(session, &map, &left, &right, zero, policy);
        }
        let good = session.upload_encoded(&[zero; 9]).unwrap();
        let wrong = foreign.upload_encoded(&[zero; 6]).unwrap();
        assert!(matches!(
            map.execute_resident(&good, &wrong),
            Err(MlxError::ForeignSession)
        ));
        let short = session.upload_encoded(&[zero; 1]).unwrap();
        assert!(matches!(
            map.execute_resident(&short, &good),
            Err(MlxError::InvalidExtent)
        ));
    }
}
#[test]
#[ignore = "Requires actual pinned MLX GPU ordered checked MatMul."]
fn strict_matmul_native_f32_f64_ordered_steps_payload_status_and_owners() {
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let foreign = runtime.open_gpu(0).unwrap();
    let bits32 = [
        0,
        0x8000_0000,
        0x3f80_0000,
        0xbf80_0000,
        1,
        0x0080_0000,
        0x3f00_0000,
        0x7f7f_ffff,
        0x7f80_0000,
        0x7fc0_1234,
        0x4040_0000,
    ];
    native(
        &session,
        &foreign,
        0f32,
        &bits32.map(f32::from_bits),
        |bits| f32::from_bits(u32::try_from(bits & 0xffff_ffff).unwrap()),
    );
    let bits64 = [
        0,
        0x8000_0000_0000_0000,
        0x3ff0_0000_0000_0000,
        0xbff0_0000_0000_0000,
        1,
        0x0010_0000_0000_0000,
        0x3fe0_0000_0000_0000,
        0x7fef_ffff_ffff_ffff,
        0x7ff0_0000_0000_0000,
        0x7ff8_0000_0000_1234,
        0x4008_0000_0000_0000,
    ];
    native(
        &session,
        &foreign,
        0f64,
        &bits64.map(f64::from_bits),
        f64::from_bits,
    );
    for shape in [
        [0, 3, 2],
        [3, 0, 2],
        [3, 3, 0],
        [3, 65536, 2],
        [usize::MAX, 3, 2],
    ] {
        assert!(
            session
                .prepare_strict_matmul(
                    PcuScalarType::F64,
                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    shape
                )
                .is_err()
        );
    }
}

#[path = "compact/compact.rs"]
mod compact;
