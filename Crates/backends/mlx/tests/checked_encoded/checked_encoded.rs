//! Exact logical carrier ownership, private completed outputs and retained session borrowing.
#[rustfmt::skip]
use fusion_pcu_mlx::{MlxRuntime,MlxError};
#[rustfmt::skip]
use pcu_facade::{PcuScalarType,PcuF16Bits,PcuBf16Bits,PcuF8E4M3FnBits,PcuF8E5M2Bits,PcuDispatchFloatUnaryOp as Op,PcuFloatUnderflowPolicy as Policy,PcuRangePolicy as Range};
#[path = "../checked_unary/graph/graph.rs"]
mod graph;
#[path = "../checked_unary/oracle/oracle.rs"]
mod oracle;
use oracle::Low;
fn transport<T: Low>() {
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let maximum = u32::from(T::SIGN) * 2 - 1;
    let input: Vec<_> = (0..=maximum)
        .map(|bits| T::from_bits(u16::try_from(bits).unwrap()))
        .collect();
    let owner = session.upload_encoded(&input).unwrap();
    assert_eq!(owner.scalar_type(), T::TYPE);
    assert_eq!(owner.element_count(), input.len());
    assert_eq!(owner.byte_len(), input.len() * T::HOST_SIZE);
    assert!(owner.same_session(&session));
    let clone = owner.clone();
    drop(owner);
    drop(runtime);
    let sentinel = T::from_bits(T::SIGN | 1);
    let mut output = vec![sentinel; input.len() + 3];
    clone.read_into(&mut output).unwrap();
    assert_eq!(&output[..input.len()], &input);
    assert_eq!(&output[input.len()..], &[sentinel; 3]);
    let mut short = [sentinel; 1];
    assert_eq!(clone.read_into(&mut short), Err(MlxError::InvalidExtent));
    assert_eq!(short, [sentinel]);
    let mut wrong = [0x1234_u32; 3];
    assert_eq!(
        clone.read_into(&mut wrong),
        Err(MlxError::UnsupportedScalar(PcuScalarType::U32))
    );
    assert_eq!(wrong, [0x1234; 3]);
    assert!(matches!(
        session.upload_encoded::<T>(&[]),
        Err(MlxError::InvalidExtent)
    ));
    drop(session);
    clone.read_into(&mut output).unwrap();
    assert_eq!(&output[..input.len()], &input);
}
// The native control is a matched benchmark peer, so its public owner seam gets
// independent preflight, terminal recovery and escaped-sibling checks.
fn control<T: Low>() {
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let other = runtime.open_gpu(0).unwrap();
    let input = [T::from_bits(1), T::from_bits(T::SIGN), T::from_bits(0)];
    let resident = session.upload_encoded(&input).unwrap();
    let foreign = other.upload_encoded(&input).unwrap();
    let mut native = session
        .prepare_checked_unary_control(
            T::TYPE,
            Op::Neg,
            Policy::RejectSubnormalResult,
            Range::Clamp,
            3,
            false,
        )
        .unwrap();
    assert!(matches!(
        native.execute_resident(&foreign),
        Err(MlxError::ForeignSession)
    ));
    assert!(matches!(
        native.execute_encoded(&input[..1]),
        Err(MlxError::InvalidExtent)
    ));
    assert!(matches!(
        native.execute_encoded(&[1_u32; 3]),
        Err(MlxError::UnsupportedScalar(PcuScalarType::U32))
    ));
    let result = native.execute_resident(&resident).unwrap();
    assert!(result.recovered_fault().unwrap().recovered);
    assert_eq!(result.recovered_fault().unwrap().invocation_id, 0);
    let (old, _) = result.into_parts();
    let mut expected = [T::from_bits(T::MAX); 6];
    oracle::native::<T, 3>(
        &input,
        &mut expected,
        Op::Neg,
        Policy::RejectSubnormalResult,
        Range::Clamp,
    )
    .unwrap_err();
    let changed = [T::from_bits(1 << T::FRACTION); 3];
    let next = native.execute_encoded(&changed).unwrap();
    assert!(next.recovered_fault().is_none());
    let fatal = [T::from_bits(1), T::from_bits(T::SIGN - 1), T::from_bits(0)];
    assert!(matches!(native.execute_encoded(&fatal),
        Err(MlxError::Arithmetic(fault)) if !fault.recovered && fault.invocation_id == 1));
    drop(next);
    drop(native);
    drop(session);
    drop(runtime);
    let mut actual = [T::from_bits(T::MAX); 6];
    old.read_into(&mut actual).unwrap();
    assert_eq!(actual, expected);
    let mut unchanged = [T::from_bits(0); 3];
    resident.read_into(&mut unchanged).unwrap();
    assert_eq!(unchanged, input);
}
#[allow(clippy::too_many_lines, clippy::cognitive_complexity)] // Paired immutable owner lifecycle covers all admitted tuples and every preflight/fatal/recovery class.
fn lifecycle<T: Low>() {
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let other = runtime.open_gpu(0).unwrap();
    let input = [T::from_bits(1), T::from_bits(T::SIGN | 1), T::from_bits(0)];
    let resident = session.upload_encoded(&input).unwrap();
    let foreign = other.upload_encoded(&input).unwrap();
    let sentinel = T::from_bits(T::MAX);
    let mut latest = [sentinel; 5];
    let mut default =
        graph::fixture::<T, _>(3, Op::Neg, Policy::IeeeAfterRounding, false, |kernel| {
            session.prepare_unary_host_kernel(kernel)
        })
        .unwrap();
    assert!(default.execute_resident(&resident).is_ok());
    for op in [Op::Neg, Op::Relu] {
        for policy in [
            Policy::IeeeAfterRounding,
            Policy::RejectSubnormalResult,
            Policy::AllowGradualUnderflow,
        ] {
            for range in [Range::Reject, Range::Clamp] {
                for grid in [false, true] {
                    let mut prepared = graph::fixture_profile::<T, _>(
                        3,
                        op,
                        policy,
                        range,
                        grid,
                        false,
                        |kernel| session.prepare_unary_host_kernel(kernel),
                    )
                    .unwrap();
                    assert!(matches!(
                        prepared.execute_resident(&foreign),
                        Err(MlxError::ForeignSession)
                    ));
                    assert!(!prepared.last_call_may_have_written());
                    let short = session.upload_encoded(&input[..1]).unwrap();
                    assert!(matches!(
                        prepared.execute_resident(&short),
                        Err(MlxError::InvalidExtent)
                    ));
                    assert!(!prepared.last_call_may_have_written());
                    let mut expected = [sentinel; 5];
                    let expected_fault =
                        oracle::native::<T, 3>(&input, &mut expected, op, policy, range).err();
                    let result = prepared.execute_resident(&resident);
                    if expected_fault.is_some_and(|fault| !fault.recovered) {
                        assert!(
                            matches!(result,Err(MlxError::Arithmetic(fault)) if Some(fault)==expected_fault)
                        );
                    } else {
                        let completed = result.unwrap();
                        assert_eq!(completed.recovered_fault(), expected_fault);
                        assert!(completed.output().same_session(&session));
                        completed.output().read_into(&mut latest).unwrap();
                        assert_eq!(latest, expected);
                        let (old_output, old_fault) = completed.into_parts();
                        assert_eq!(old_fault, expected_fault);
                        let changed = [
                            T::from_bits(T::SIGN),
                            T::from_bits(1 << T::FRACTION),
                            T::from_bits(T::SIGN),
                        ];
                        let changed_owner = session.upload_encoded(&changed).unwrap();
                        let next = prepared.execute_resident(&changed_owner).unwrap();
                        assert!(next.recovered_fault().is_none());
                        old_output.read_into(&mut latest).unwrap();
                        assert_eq!(latest, expected);
                        let host = prepared.execute_encoded(&input).unwrap();
                        assert_eq!(host.recovered_fault(), expected_fault);
                        host.output().read_into(&mut latest).unwrap();
                        assert_eq!(latest, expected);
                    }
                    assert!(!prepared.last_call_may_have_written_existing_encoded_owner());
                    let mut original = [sentinel; 3];
                    resident.read_into(&mut original).unwrap();
                    assert_eq!(original, input);
                }
            }
        }
    }
    let mut clamp = graph::fixture_profile::<T, _>(
        3,
        Op::Neg,
        Policy::RejectSubnormalResult,
        Range::Clamp,
        false,
        false,
        |kernel| session.prepare_unary_host_kernel(kernel),
    )
    .unwrap();
    let nan = T::from_bits(T::SIGN - 1);
    let bad = [T::from_bits(1), nan, T::from_bits(0)];
    let old = session.upload_encoded(&bad).unwrap();
    assert!(
        matches!(clamp.execute_resident(&old),Err(MlxError::Arithmetic(fault)) if !fault.recovered&&fault.invocation_id==1)
    );
    assert!(clamp.last_call_may_have_written());
    assert!(!clamp.last_call_may_have_written_existing_encoded_owner());
    let mut unchanged = [sentinel; 3];
    old.read_into(&mut unchanged).unwrap();
    assert_eq!(unchanged, bad);
    let scalar = session.upload_encoded(&input[..1]).unwrap();
    let mut broadcast = graph::fixture_profile::<T, _>(
        257,
        Op::Neg,
        Policy::RejectSubnormalResult,
        Range::Clamp,
        true,
        true,
        |kernel| session.prepare_unary_host_kernel(kernel),
    )
    .unwrap();
    let completed = broadcast.execute_resident(&scalar).unwrap();
    assert_eq!(completed.recovered_fault().unwrap().invocation_id, 0);
    assert!(completed.recovered_fault().unwrap().recovered);
    let mut values = vec![sentinel; 260];
    completed.output().read_into(&mut values).unwrap();
    assert_eq!(&values[..257], &vec![T::from_bits(T::SIGN | 1); 257]);
    assert_eq!(&values[257..], &[sentinel; 3]);
    assert!(broadcast.execute_resident(&resident).is_err());
    assert!(!broadcast.last_call_may_have_written());
    let wrong = if T::TYPE == PcuScalarType::F16 {
        session
            .upload_encoded(&[PcuBf16Bits::from_bits(1); 3])
            .unwrap()
    } else {
        session
            .upload_encoded(&[PcuF16Bits::from_bits(1); 3])
            .unwrap()
    };
    assert!(matches!(
        clamp.execute_resident(&wrong),
        Err(MlxError::UnsupportedScalar(_))
    ));
    assert!(!clamp.last_call_may_have_written());
    // Returned immutable owners retain the real array/runtime/stream after preparation drops.
    let completed = clamp.execute_resident(&resident).unwrap();
    let (output, _) = completed.into_parts();
    drop(clamp);
    drop(session);
    drop(runtime);
    output.read_into(&mut unchanged).unwrap();
    assert_eq!(
        unchanged,
        [
            T::from_bits(T::SIGN | 1),
            T::from_bits(1),
            T::from_bits(T::SIGN)
        ]
    );
}
macro_rules! formats {
    ($name:ident,$ty:ty) => {
        #[test]
        #[ignore = "Requires actual MLX encoded-owner full transport and resident lifecycle proof."]
        fn $name() {
            transport::<$ty>();
            lifecycle::<$ty>();
            control::<$ty>();
        }
    };
}
formats!(half_encoded, PcuF16Bits);
formats!(bfloat_encoded, PcuBf16Bits);
formats!(e4_encoded, PcuF8E4M3FnBits);
formats!(e5_encoded, PcuF8E5M2Bits);
