//! Twenty-two authentic encoded Identity owners, exact bytes, tails and native session lifetime.
#[path = "source/source.rs"]
mod source;
#[rustfmt::skip]
use std::sync::Arc;
#[rustfmt::skip]
use pcu_facade::{PcuF16Bits,PcuBf16Bits,PcuF8E4M3FnBits,PcuF8E5M2Bits,
    PcuF128Bits,PcuF256Bits,PcuU256,PcuI256,PcuU512,PcuI512,PcuImplementationRequirements,
    PcuNumericalMode,PcuCompoundArithmeticPolicy,PcuPrecisionPolicy,PcuFloatUnderflowPolicy};
#[rustfmt::skip]
use fusion_pcu_mlx::{MlxRuntime,MlxError,MlxCheckedProgramInput};
#[path = "support/support.rs"]
mod support;
use support::{Sample, compare};
#[allow(clippy::too_many_lines)] // One retained-owner, request tuple and byte oracle preserves a complete native lifetime.
fn qualify<T: Sample>() {
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let foreign = runtime.open_gpu(0).unwrap();
    let sentinel = T::sample(91);
    let mut last: Option<(fusion_pcu_mlx::MlxEncodedArray, [T; 7])> = None;
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for compound in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for precision in [
                PcuPrecisionPolicy::Preserve,
                PcuPrecisionPolicy::BackendOptimized,
            ] {
                for underflow in [
                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    PcuFloatUnderflowPolicy::RejectSubnormalResult,
                    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                ] {
                    let requirements = PcuImplementationRequirements {
                        numerical_mode: mode,
                        float_underflow: underflow,
                        numerical_options: pcu_facade::PcuNumericalOptions {
                            compound_arithmetic: compound,
                            precision,
                            reproducibility: pcu_facade::PcuReproducibility::Unspecified,
                        },
                        ..PcuImplementationRequirements::default()
                    };
                    let capture = pcu_facade::global::__pcu_capture_tensor_program(
                        [pcu_facade::global::PcuSourceShape::Slice { length: 7 }],
                        underflow,
                        mode,
                        requirements.numerical_options,
                        source::retain::__pcu_capture_entry::<T>,
                    )
                    .unwrap();
                    let prepared = session
                        .prepare_checked_program(Arc::clone(capture.program()), requirements)
                        .unwrap();
                    assert_eq!(prepared.plan().scalar_type(), T::TYPE);
                    assert_eq!(prepared.plan().shape(), &[7]);
                    for phase in [0_u8, 23, 71] {
                        let mut input = [0_u8, 1, 2, 3, 5, 7, 11]
                            .map(|seed| T::sample(seed.wrapping_add(phase)));
                        let expected = input;
                        let owner = prepared.execute_host(&input).unwrap();
                        assert_eq!(owner.scalar_type(), T::TYPE);
                        assert_eq!(owner.element_count(), 7);
                        assert_eq!(owner.byte_len(), 7 * T::HOST_SIZE);
                        input.fill(sentinel);
                        let mut actual = [sentinel; 9];
                        owner.read_into(&mut actual).unwrap();
                        compare(&actual[..7], &expected);
                        compare(&actual[7..], &[sentinel; 2]);
                        let mut short = [sentinel; 6];
                        assert_eq!(owner.read_into(&mut short), Err(MlxError::InvalidExtent));
                        compare(&short, &[sentinel; 6]);
                        let sibling = owner.clone();
                        let borrowed = prepared
                            .execute_mixed(&[(
                                prepared.plan().input(),
                                MlxCheckedProgramInput::Resident(&owner),
                            )])
                            .unwrap();
                        drop(owner);
                        borrowed.read_into(&mut actual).unwrap();
                        compare(&actual[..7], &expected);
                        let prefix = session.upload_encoded(&expected[..1]).unwrap();
                        let background = session.upload_encoded(&[sentinel; 7]).unwrap();
                        let merge = session.prepare_encoded_prefix(T::TYPE, 1, 7).unwrap();
                        let merged = merge.execute(&prefix, &background).unwrap();
                        merged.read_into(&mut actual).unwrap();
                        compare(&actual[..1], &expected[..1]);
                        compare(&actual[1..7], &[sentinel; 6]);
                        background.read_into(&mut actual).unwrap();
                        compare(&actual[..7], &[sentinel; 7]);
                        let other = foreign.upload_encoded(&expected).unwrap();
                        assert!(matches!(
                            prepared.execute_mixed(&[(
                                prepared.plan().input(),
                                MlxCheckedProgramInput::Resident(&other)
                            )]),
                            Err(MlxError::ForeignSession)
                        ));
                        if let Some((previous, previous_expected)) =
                            last.replace((sibling, expected))
                        {
                            previous.read_into(&mut actual).unwrap();
                            compare(&actual[..7], &previous_expected);
                        }
                    }
                }
            }
        }
    }
    drop(session);
    drop(foreign);
    drop(runtime);
    let (owner, expected) = last.unwrap();
    let mut actual = [sentinel; 9];
    owner.read_into(&mut actual).unwrap();
    compare(&actual[..7], &expected);
}
#[test]
#[ignore = "Requires real MLX integer carriers, authentic owned source capture and terminal readback."]
fn twenty_two_carrier_source_identity_owners() {
    macro_rules! all {($($ty:ty),+) => {$(qualify::<$ty>();)+};}
    all!(
        u8,
        i8,
        u16,
        i16,
        u32,
        i32,
        u64,
        i64,
        u128,
        i128,
        PcuU256,
        PcuI256,
        PcuU512,
        PcuI512,
        PcuF16Bits,
        PcuBf16Bits,
        PcuF8E4M3FnBits,
        PcuF8E5M2Bits,
        f32,
        f64,
        PcuF128Bits,
        PcuF256Bits
    );
}
