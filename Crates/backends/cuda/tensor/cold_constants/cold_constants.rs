//! Independent cold literal expectations under every rounding/flush state.
#[path = "environment/environment.rs"]
mod environment;
#[rustfmt::skip]
use fusion_pcu::{
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuScalarType,
};
use fusion_pcu::dialect::tensor::Graph;
use std::hint::black_box;

fn loss_constants() {
    for (count, f32_bits, f64_bits) in [
        (3, 0x4040_0000_u64, 0x4008_0000_0000_0000_u64),
        (16_777_217, 0x4b80_0000, 0x4170_0000_1000_0000),
        (16_777_219, 0x4b80_0002, 0x4170_0000_3000_0000),
        (2_147_483_647, 0x4f00_0000, 0x41df_ffff_ffc0_0000),
    ] {
        for (scalar, expected) in [
            (PcuScalarType::F32, f32_bits),
            (PcuScalarType::F64, f64_bits),
        ] {
            let mut graph = Graph::default();
            graph.set_numerical_mode(PcuNumericalMode::Strict);
            let left = graph.input([black_box(count)], scalar).unwrap();
            let right = graph.input([count], scalar).unwrap();
            let loss = graph.mean_squared_error(left, right).unwrap();
            let source = super::strict_mse::Profile::from_node(&graph, graph.node(loss).unwrap())
                .unwrap()
                .source();
            assert!(
                source.contains(&format!("({expected}ull)")),
                "count={count} scalar={scalar:?}"
            );
        }
    }
}

fn rate_constants() {
    // Binary64 values are independent encoding goldens, never host floating casts.
    for (bits, expected) in [
        (0x0000_0001, 0x36a0_0000_0000_0000_u64),
        (0x007f_ffff, 0x380f_ffff_c000_0000),
        (0x8000_0001, 0xb6a0_0000_0000_0000),
        (0x8000_0000, 0x8000_0000_0000_0000),
        (0x3dcc_cccd, 0x3fb9_9999_a000_0000),
    ] {
        for policy in [
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        ] {
            let mut graph = Graph::default();
            graph.set_numerical_mode(PcuNumericalMode::Strict);
            let left = graph.input([17], PcuScalarType::F64).unwrap();
            let right = graph.input([17], PcuScalarType::F64).unwrap();
            let rate = f32::from_bits(black_box(bits));
            let update = graph.sgd_update(left, right, rate).unwrap();
            graph
                .set_value_float_underflow_policy(update, policy)
                .unwrap();
            let source = super::strict_sgd::assess(&graph, graph.node(update).unwrap())
                .unwrap()
                .source();
            assert!(
                source.contains(&format!("({expected}ull)")),
                "rate={bits:#010x}"
            );
        }
    }
}

fn native_reciprocals() {
    assert_eq!(
        super::mse_scale(black_box(3)).unwrap().to_bits(),
        0x3eaa_aaab
    );
    assert_eq!(
        super::mse_scale_f64(black_box(3)).unwrap().to_bits(),
        0x3fd5_5555_5555_5555
    );
    assert_eq!(
        super::mse_scale(black_box(16_777_217)).unwrap().to_bits(),
        0x3380_0000
    );
    for count in [0, i32::MAX as usize + 1] {
        assert!(matches!(
            super::mse_scale(black_box(count)),
            Err(super::CudaTensorExecutionError::SizeOverflow)
        ));
        assert!(matches!(
            super::mse_scale_f64(black_box(count)),
            Err(super::CudaTensorExecutionError::SizeOverflow)
        ));
    }
    assert!(matches!(
        super::mse_scale_f64(0),
        Err(super::CudaTensorExecutionError::SizeOverflow)
    ));
}

#[test]
fn cold_constants_ignore_directed_rounding_and_flush_inputs() {
    let original = environment::state();
    for rounding in 0..4 {
        for flush in [false, true] {
            let _guard = environment::Guard::enter(rounding, flush);
            eprintln!("cold constants rounding={rounding} flush={flush}");
            loss_constants();
            rate_constants();
            native_reciprocals();
        }
    }
    assert_eq!(environment::state(), original);
}
