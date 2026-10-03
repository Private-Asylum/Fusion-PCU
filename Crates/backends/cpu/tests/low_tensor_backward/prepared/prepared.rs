//! Independent bit-oracle proof of detached plans, whole-call publication and workspace reuse.
#[path = "../../prepared_tensor/allocation/allocation.rs"]
mod allocation;
#[rustfmt::skip]
use pcu_facade::{PcuScalar,PcuFloatUnderflowPolicy,PcuNumericalMode,PcuF16Bits,PcuBf16Bits,PcuF8E4M3FnBits,PcuF8E5M2Bits};
#[rustfmt::skip]
use pcu_facade::dialect::tensor::{Graph,TensorElement,TensorError};
use fusion_pcu_cpu::PcuCpuPreparedTensorGraph;
use super::oracle::{self, Format};
#[allow(clippy::too_many_lines)] // One cold plan is reused across the complete representation and transactional proof.
fn proof<T: Format + TensorElement>(low: bool) {
    let mut lanes = 0_usize;
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for policy in [
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        ] {
            let mut graph = Graph::default();
            graph.set_numerical_mode(mode);
            let input = graph.input([1], T::TYPE).unwrap();
            let upstream = graph.input([1], T::TYPE).unwrap();
            let derivative = graph.relu_backward(input, upstream).unwrap();
            graph
                .set_value_float_underflow_policy(derivative, policy)
                .unwrap();
            let mut plan =
                PcuCpuPreparedTensorGraph::<T>::prepare(&graph, &[input, derivative]).unwrap();
            drop(graph);
            let partners = [
                0,
                T::SIGN,
                1,
                T::SIGN | 1,
                T::ONE,
                T::SIGN | T::MAX,
                T::MAX + 1,
                T::SIGN | (T::MAX + 1),
            ];
            let representations = if low {
                (0..T::SIGN * 2).collect::<Vec<_>>()
            } else {
                vec![
                    0,
                    T::SIGN,
                    1,
                    T::SIGN | 1,
                    T::MIN_NORMAL - 1,
                    T::MIN_NORMAL,
                    T::ONE,
                    T::MAX,
                    T::SIGN | T::MAX,
                    T::MAX + 1,
                    T::SIGN | (T::MAX + 1),
                    T::SIGN - 1,
                ]
            };
            let mut copied = [T::sentinel(); 3];
            let mut output = [T::sentinel(); 3];
            for bits in representations {
                for partner in partners {
                    for swap in [false, true] {
                        let (x, dy) = if swap {
                            (T::from(partner), T::from(bits))
                        } else {
                            (T::from(bits), T::from(partner))
                        };
                        copied.fill(T::sentinel());
                        output.fill(T::sentinel());
                        let actual = plan.call(&[&[x], &[dy]], &mut [&mut copied, &mut output]);
                        match oracle::expected(x, dy, policy) {
                            Ok(want) => {
                                actual.unwrap();
                                assert_eq!(copied[0].bits(), x.bits());
                                assert_eq!(output[0].bits(), want.bits());
                            }
                            Err(kind) => {
                                assert!(
                                    matches!(actual,Err(TensorError::ArithmeticFault{value,element_index:0,kind:actual})if value==derivative&&actual==kind)
                                );
                                assert!(plan.output(0).is_none());
                                assert!(plan.output(1).is_none());
                                assert_eq!(copied[0].bits(), T::sentinel().bits());
                                assert_eq!(output[0].bits(), T::sentinel().bits());
                            }
                        }
                        assert!(
                            copied[1..]
                                .iter()
                                .chain(&output[1..])
                                .all(|v| v.bits() == T::sentinel().bits())
                        );
                        lanes += 1;
                    }
                }
            }
            let one = [T::one()];
            assert_eq!(
                allocation::count(|| {
                    for _ in 0..64 {
                        plan.call(&[&one, &one], &mut [&mut copied, &mut output])
                            .unwrap();
                    }
                }),
                0
            );
            let saved = (copied, output);
            assert!(
                plan.call(&[&[], &one], &mut [&mut copied, &mut output])
                    .is_err()
            );
            assert_eq!((copied, output), saved);
            assert!(
                plan.call(&[&one, &one], &mut [&mut copied, &mut []])
                    .is_err()
            );
            assert_eq!((copied, output), saved);
            plan.call(&[&one, &one], &mut [&mut copied, &mut output])
                .unwrap();
            assert_eq!(output[0].bits(), T::ONE);
        }
    }
    println!("{} backward independent bit-oracle lanes={lanes}", T::LABEL);
}
#[test]
fn f16() {
    proof::<PcuF16Bits>(true);
}
#[test]
fn bf16() {
    proof::<PcuBf16Bits>(true);
}
#[test]
fn e4m3fn() {
    proof::<PcuF8E4M3FnBits>(true);
}
#[test]
fn e5m2() {
    proof::<PcuF8E5M2Bits>(true);
}
#[test]
fn f32() {
    proof::<f32>(false);
}
#[test]
fn f64() {
    proof::<f64>(false);
}

#[test]
fn selected_constant_uniform_and_local_underflow_are_frozen() {
    let mut graph = Graph::default();
    let input = graph
        .uniform_typed([3], PcuF16Bits::from_bits(0x3c00))
        .unwrap();
    let upstream = graph.constant_typed(
        pcu_facade::dialect::tensor::Tensor::new(
            [3],
            vec![
                PcuF16Bits::from_bits(0x8000),
                PcuF16Bits::from_bits(1),
                PcuF16Bits::from_bits(0x3c00),
            ],
        )
        .unwrap(),
    );
    let derivative = graph
        .relu_backward(input.erase(), upstream.erase())
        .unwrap();
    graph
        .set_value_float_underflow_policy(
            derivative,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        )
        .unwrap();
    let mut plan = PcuCpuPreparedTensorGraph::<PcuF16Bits>::prepare(&graph, &[derivative]).unwrap();
    graph
        .set_value_float_underflow_policy(
            derivative,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
        )
        .unwrap();
    drop(graph);
    let mut output = [PcuF16Bits::from_bits(0x4000); 5];
    assert_eq!(
        allocation::count(|| plan.call(&[], &mut [&mut output]).unwrap()),
        0
    );
    assert_eq!(
        output.map(PcuF16Bits::to_bits),
        [0x8000, 1, 0x3c00, 0x4000, 0x4000]
    );
}

#[test]
fn first_node_lane_fault_prevents_all_output_publication_and_retry() {
    let mut graph = Graph::default();
    let input = graph.input([9], PcuF8E5M2Bits::TYPE).unwrap();
    let upstream = graph.input([9], PcuF8E5M2Bits::TYPE).unwrap();
    let derivative = graph.relu_backward(input, upstream).unwrap();
    graph
        .set_value_float_underflow_policy(
            derivative,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
        )
        .unwrap();
    let mut plan =
        PcuCpuPreparedTensorGraph::<PcuF8E5M2Bits>::prepare(&graph, &[input, derivative]).unwrap();
    let one = PcuF8E5M2Bits::one();
    let sentinel = PcuF8E5M2Bits::sentinel();
    let input = [one; 9];
    let mut upstream = [one; 9];
    let mut copied = [sentinel; 12];
    let mut output = [sentinel; 12];
    for (first, later, kind) in [
        (
            1,
            0x7c,
            pcu_facade::PcuExecutionFaultKind::ArithmeticUnderflow,
        ),
        (
            0x7c,
            1,
            pcu_facade::PcuExecutionFaultKind::InvalidFloatingOperand,
        ),
    ] {
        upstream[2] = PcuF8E5M2Bits::from_bits(first);
        upstream[6] = PcuF8E5M2Bits::from_bits(later);
        assert!(
            matches!(plan.call(&[&input,&upstream],&mut[&mut copied,&mut output]),Err(TensorError::ArithmeticFault{value,element_index:2,kind:actual})if value==derivative&&actual==kind)
        );
        assert_eq!(copied, [sentinel; 12]);
        assert_eq!(output, [sentinel; 12]);
        assert!(plan.output(0).is_none());
        assert!(plan.output(1).is_none());
        upstream.fill(one);
        plan.call(&[&input, &upstream], &mut [&mut copied, &mut output])
            .unwrap();
        assert_eq!(output[..9], [one; 9]);
        assert_eq!(output[9..], [sentinel; 3]);
        copied.fill(sentinel);
        output.fill(sentinel);
    }
}
