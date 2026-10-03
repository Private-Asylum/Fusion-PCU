//! Permissions preserve separately rounded Strict arithmetic and private scratch publication.
#[rustfmt::skip]
use fusion_pcu_core::{
    PcuCompoundArithmeticPolicy,
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuPrecisionPolicy,
    PcuScalar,
    dialect::tensor::{Graph,TensorArithmeticStep,TensorError},
};
use fusion_pcu_cpu::PcuCpuPreparedTensorGraph;

macro_rules! verify {
    ($name:ident, $ty:ty) => {
        #[test]
        #[allow(clippy::too_many_lines)] // One policy matrix keeps zero-allocation publication, intermediate faults and retry paired for each tuple.
        fn $name() {
            for compound_arithmetic in [
                PcuCompoundArithmeticPolicy::Checked,
                PcuCompoundArithmeticPolicy::BackendDefined,
            ] {
                for precision in [
                    PcuPrecisionPolicy::Preserve,
                    PcuPrecisionPolicy::BackendOptimized,
                ] {
                    for policy in [
                        PcuFloatUnderflowPolicy::IeeeAfterRounding,
                        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                        PcuFloatUnderflowPolicy::RejectSubnormalResult,
                    ] {
                        for operation in 0..3 {
                            let mut graph = Graph::default();
                            graph.set_numerical_mode(PcuNumericalMode::Strict);
                            graph.set_numerical_options(PcuNumericalOptions {
                                compound_arithmetic,
                                precision,
                                ..Default::default()
                            });
                            let left = graph.input([2, 2], <$ty>::TYPE).unwrap();
                            let right = graph.input([2, 2], <$ty>::TYPE).unwrap();
                            let output = match operation {
                                0 => graph.matmul(left, right),
                                1 => graph.mean_squared_error(left, right),
                                _ => graph.sgd_update(left, right, 0.5),
                            }
                            .unwrap();
                            graph
                                .set_value_float_underflow_policy(output, policy)
                                .unwrap();
                            let mut plan =
                                PcuCpuPreparedTensorGraph::<$ty>::prepare(&graph, &[output])
                                    .unwrap();
                            let banks: [[$ty; 4]; 2] = [[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]];
                            let rhs: [$ty; 4] = if operation == 0 {
                                [1.0, 0.0, 0.0, 1.0]
                            } else {
                                [0.0; 4]
                            };
                            let mut observed = [17.0 as $ty; 5];
                            for call in 0..64 {
                                let bank = call % 2;
                                let input = &banks[bank];
                                let rhs = if operation == 2 { input } else { &rhs };
                                assert_eq!(
                                    super::allocation::count(|| plan
                                        .call(&[input, rhs], &mut [&mut observed])
                                        .unwrap()),
                                    0
                                );
                                let expected: &[$ty] = match (operation, bank) {
                                    (0, 0) => &[1.0, 2.0, 3.0, 4.0],
                                    (0, _) => &[2.0, 4.0, 6.0, 8.0],
                                    (1, 0) => &[7.5],
                                    (1, _) => &[30.0],
                                    (_, 0) => &[0.5, 1.0, 1.5, 2.0],
                                    (_, _) => &[1.0, 2.0, 3.0, 4.0],
                                };
                                for (a, b) in observed.iter().zip(expected) {
                                    assert_eq!(a.to_bits(), b.to_bits());
                                }
                                for value in &observed[expected.len()..] {
                                    assert_eq!(value.to_bits(), (17.0 as $ty).to_bits());
                                }
                            }
                        }
                        let mut graph = Graph::default();
                        graph.set_numerical_mode(PcuNumericalMode::Strict);
                        graph.set_numerical_options(PcuNumericalOptions {
                            compound_arithmetic,
                            precision,
                            ..Default::default()
                        });
                        let left = graph.input([1, 2], <$ty>::TYPE).unwrap();
                        let right = graph.input([2, 1], <$ty>::TYPE).unwrap();
                        let output = graph.matmul(left, right).unwrap();
                        graph
                            .set_value_float_underflow_policy(output, policy)
                            .unwrap();
                        let mut plan =
                            PcuCpuPreparedTensorGraph::<$ty>::prepare(&graph, &[output]).unwrap();
                        let mut observed = [17.0 as $ty; 2];
                        let result = plan.call(
                            &[&[<$ty>::from_bits(1), 1.0], &[0.5, 1.0]],
                            &mut [&mut observed],
                        );
                        if policy == PcuFloatUnderflowPolicy::AllowGradualUnderflow {
                            result.unwrap();
                            assert_eq!(observed[0].to_bits(), (1.0 as $ty).to_bits());
                        } else {
                            assert!(matches!(
                                result,
                                Err(TensorError::CompoundArithmeticFault {
                                    kind: PcuExecutionFaultKind::ArithmeticUnderflow,
                                    step: TensorArithmeticStep::Multiply,
                                    ..
                                })
                            ));
                            assert_eq!(
                                observed.map(<$ty>::to_bits),
                                [17.0 as $ty; 2].map(<$ty>::to_bits)
                            );
                            assert!(plan.output(0).is_none());
                        }
                        plan.call(&[&[1.0, 2.0], &[0.5, 1.0]], &mut [&mut observed])
                            .unwrap();
                        assert_eq!(
                            observed.map(<$ty>::to_bits),
                            [2.5 as $ty, 17.0].map(<$ty>::to_bits)
                        );
                    }
                }
            }
        }
    };
}
verify!(f32_strict_permissions_keep_checks, f32);
verify!(f64_strict_permissions_keep_checks, f64);
