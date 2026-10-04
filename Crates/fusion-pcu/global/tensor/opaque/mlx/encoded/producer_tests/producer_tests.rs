//! Transport metadata must not fabricate arithmetic tininess requirements.
#[rustfmt::skip]
use crate::{
    PcuExecutionError,
    PcuF128Bits,
    PcuFloatUnderflowPolicy,
    PcuTensor,
    global::{
        PcuBackendChoice,
        PcuExecutionPolicy,
        tensor::capture,
    },
};

#[crate::pcu(crate_path = crate)]
fn raw_literal() -> Result<PcuTensor<PcuF128Bits>, PcuExecutionError> {
    pcu::constant(
        const {
            [
                PcuF128Bits::from_limbs_le([0x42, 0x7fff_0000_0000_0000]),
                PcuF128Bits::from_limbs_le([0, 0x8000_0000_0000_0000]),
                PcuF128Bits::from_limbs_le([1, 0]),
            ]
        },
    )
}

#[crate::pcu(crate_path = crate)]
fn raw_uniform() -> Result<PcuTensor<PcuF128Bits>, PcuExecutionError> {
    let anchor = pcu::constant(const { [PcuF128Bits::from_limbs_le([1, 0]); 3] })?;
    pcu::uniform_like(
        &anchor,
        const { PcuF128Bits::from_limbs_le([0, 0x8000_0000_0000_0000]) },
    )
}

#[test]
fn immutable_raw_producers_keep_storage_metadata_without_arithmetic_policy() {
    for underflow in [
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
    ] {
        let policy = PcuExecutionPolicy {
            backend: PcuBackendChoice::Mlx,
            float_underflow: underflow,
            ..Default::default()
        };
        for uniform in [false, true] {
            let captured = if uniform {
                capture::build(
                    [],
                    underflow,
                    policy.numerical_mode,
                    policy.numerical_options,
                    raw_uniform::__pcu_capture_entry,
                )
            } else {
                capture::build(
                    [],
                    underflow,
                    policy.numerical_mode,
                    policy.numerical_options,
                    raw_literal::__pcu_capture_entry,
                )
            }
            .unwrap();
            assert!(captured.input_indices.is_empty());
            let requirements = super::requirements(&captured, policy).unwrap();
            assert_eq!(requirements.float_underflow, underflow);
            assert_eq!(requirements.numerical_mode, policy.numerical_mode);
            assert!(super::assess(&captured, policy).is_ok());
            for &value in captured.program.node_order() {
                let node = captured.program.graph().node(value).unwrap();
                assert!(node.numerical_mode.is_none());
                assert!(node.float_underflow_policy.is_none());
            }
        }
    }
}
