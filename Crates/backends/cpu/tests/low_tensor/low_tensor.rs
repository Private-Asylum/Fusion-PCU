//! Exact low-format cold plans and reusable execution, independent midpoint oracle.
extern crate pcu_facade as fusion_pcu;
#[path = "../prepared_tensor/allocation/allocation.rs"]
mod allocation;
#[path = "../../../rocm/benches/low_tensor/oracle/oracle.rs"]
mod oracle;
#[path = "source/source.rs"]
#[allow(dead_code)]
mod source;
#[rustfmt::skip]
use fusion_pcu::{PcuScalar,PcuFloatUnderflowPolicy,PcuNumericalOptions,PcuNumericalMode,PcuCompoundArithmeticPolicy,PcuPrecisionPolicy,PcuReproducibility,PcuF16Bits,PcuBf16Bits,PcuF8E4M3FnBits,PcuF8E5M2Bits};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{Graph,Tensor,TensorError,TensorElement,TensorOperationAssessor,TensorOperationSupport};
use fusion_pcu_cpu::{PcuCpuPreparedTensorGraph, PcuCpuTensorAssessor};
use oracle::Format;
fn check<T: Format + TensorElement>() {
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for policy in [
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        ] {
            for op in 0..5 {
                let mut graph = Graph::default();
                graph.set_numerical_mode(mode);
                let left = graph.input([257], T::TYPE).unwrap();
                let right = graph.input([257], T::TYPE).unwrap();
                let output = match op {
                    0 => graph.add(left, right),
                    1 => graph.sub(left, right),
                    2 => graph.mul(left, right),
                    3 => graph.div(left, right),
                    _ => graph.relu(left),
                }
                .unwrap();
                graph
                    .set_value_float_underflow_policy(output, policy)
                    .unwrap();
                let mut plan = PcuCpuPreparedTensorGraph::<T>::prepare(&graph, &[output]).unwrap();
                let mut observed = vec![T::sentinel(); 260];
                for phase in [1, 17, 51] {
                    let (a, b, want) = oracle::inputs::<T>(257, phase, op, policy);
                    let inputs = if op == 4 {
                        vec![a.as_slice()]
                    } else {
                        vec![a.as_slice(), b.as_slice()]
                    };
                    plan.call(&inputs, &mut [&mut observed]).unwrap();
                    assert_eq!(&observed[..257], &want);
                    assert_eq!(&observed[257..], &[T::sentinel(); 3]);
                    assert_eq!(
                        allocation::count(|| {
                            for _ in 0..64 {
                                plan.call(&inputs, &mut [&mut observed]).unwrap();
                            }
                        }),
                        0
                    );
                }
                let before = observed.clone();
                let short = [T::one(); 1];
                let inputs = if op == 4 {
                    vec![short.as_slice()]
                } else {
                    vec![short.as_slice(), short.as_slice()]
                };
                assert!(plan.call(&inputs, &mut [&mut observed]).is_err());
                assert_eq!(observed, before);
                assert!(plan.output(0).is_none());
            }
        }
    }
}
#[test]
fn f16_prepared_oracle() {
    check::<PcuF16Bits>();
}
#[test]
fn bf16_prepared_oracle() {
    check::<PcuBf16Bits>();
}
#[test]
fn e4_prepared_oracle() {
    check::<PcuF8E4M3FnBits>();
}
#[test]
fn e5_prepared_oracle() {
    check::<PcuF8E5M2Bits>();
}
#[test]
fn exact_cold_permissions_detached_constants_and_multioutput_transaction() {
    for compound in [
        PcuCompoundArithmeticPolicy::Checked,
        PcuCompoundArithmeticPolicy::BackendDefined,
    ] {
        for precision in [
            PcuPrecisionPolicy::Preserve,
            PcuPrecisionPolicy::BackendOptimized,
        ] {
            let mut graph = Graph::default();
            graph.set_numerical_options(PcuNumericalOptions {
                compound_arithmetic: compound,
                precision,
                reproducibility: PcuReproducibility::Unspecified,
            });
            let input = graph.input([3], PcuF16Bits::TYPE).unwrap();
            let values = [PcuF16Bits::from_bits(0x3c00); 3];
            let constant = graph.constant_typed(Tensor::new([3], values.to_vec()).unwrap());
            let uniform = graph
                .uniform_typed([3], PcuF16Bits::from_bits(0x4000))
                .unwrap();
            let added = graph.add(input, constant.erase()).unwrap();
            let multiplied = graph.mul(added, uniform.erase()).unwrap();
            let activation = graph.relu(multiplied).unwrap();
            assert!(matches!(
                PcuCpuTensorAssessor.assess_node(&graph, graph.node(activation).unwrap()),
                TensorOperationSupport::Supported { .. }
            ));
            let mut plan =
                PcuCpuPreparedTensorGraph::<PcuF16Bits>::prepare(&graph, &[added, activation])
                    .unwrap();
            drop(graph);
            let sentinel = PcuF16Bits::from_bits(0x3555);
            let mut first = [sentinel; 5];
            let mut second = [sentinel; 5];
            plan.call(&[&values], &mut [&mut first, &mut second])
                .unwrap();
            assert_eq!(first[..3], [PcuF16Bits::from_bits(0x4000); 3]);
            assert_eq!(second[..3], [PcuF16Bits::from_bits(0x4400); 3]);
            let before = (first, second);
            let bad = [values[0], PcuF16Bits::from_bits(0x7c00), values[0]];
            assert!(matches!(
                plan.call(&[&bad], &mut [&mut first, &mut second]),
                Err(TensorError::ArithmeticFault {
                    element_index: 1,
                    ..
                })
            ));
            assert_eq!((first, second), before);
            assert!(plan.output(0).is_none());
            assert!(plan.output(1).is_none());
            plan.call(&[&values], &mut [&mut first, &mut second])
                .unwrap();
            assert_eq!((first, second), before);
        }
    }
    let mut graph = Graph::default();
    graph.set_numerical_options(PcuNumericalOptions {
        reproducibility: PcuReproducibility::PortableV1,
        ..Default::default()
    });
    let input = graph.input([3], PcuF16Bits::TYPE).unwrap();
    assert!(matches!(
        PcuCpuPreparedTensorGraph::<PcuF16Bits>::prepare(&graph, &[input]),
        Err(TensorError::UnsupportedNumericalOptions { .. })
    ));
}

#[rustfmt::skip]
use fusion_pcu::{global,PcuTensor,PcuExecutionError,PcuExecutionFaultKind};
fn call<T: Format>(
    a: &PcuTensor<T>,
    b: &PcuTensor<T>,
    op: u32,
) -> Result<PcuTensor<T>, PcuExecutionError> {
    match op {
        0 => source::add(a, b),
        1 => source::sub(a, b),
        2 => source::mul(a, b),
        3 => source::div(a, b),
        4 => source::relu(a),
        _ => unreachable!(),
    }
}
fn read<T: Format>(owner: &PcuTensor<T>, expected: &[T]) {
    let mut output = vec![T::sentinel(); expected.len() + 2];
    owner.read_into(&mut output).unwrap();
    assert_eq!(&output[..expected.len()], expected);
    assert_eq!(&output[expected.len()..], &[T::sentinel(); 2]);
}
fn format<T: Format>(policy: PcuFloatUnderflowPolicy) {
    for op in 0..5 {
        global::clear_thread_cache().unwrap();
        let mut previous: Option<(PcuTensor<T>, Vec<T>)> = None;
        for phase in [1, 17, 51] {
            let (a, b, want) = oracle::inputs::<T>(257, phase, op, policy);
            let ra = source::identity(a.as_slice()).unwrap();
            let rb = source::identity(b.as_slice()).unwrap();
            let output = call(&ra, &rb, op).unwrap();
            read(&output, &want);
            let host = match op {
                0 => source::add(a.as_slice(), b.as_slice()),
                1 => source::sub(a.as_slice(), b.as_slice()),
                2 => source::mul(a.as_slice(), b.as_slice()),
                3 => source::div(a.as_slice(), b.as_slice()),
                4 => source::relu(a.as_slice()),
                _ => unreachable!(),
            }
            .unwrap();
            read(&host, &want);
            let mixed = match op {
                0 => source::add(&ra, b.as_slice()),
                1 => source::sub(&ra, b.as_slice()),
                2 => source::mul(&ra, b.as_slice()),
                3 => source::div(&ra, b.as_slice()),
                4 => source::relu(&ra),
                _ => unreachable!(),
            }
            .unwrap();
            read(&mixed, &want);
            read(&source::consumed_identity(host).unwrap(), &want);
            if let Some((old, values)) = &previous {
                read(old, values);
            }
            previous = Some((output, want));
            read(&ra, &a);
            read(&rb, &b);
        }
        let mut a = vec![T::one(); 65];
        let mut b = vec![T::one(); 65];
        let kind = if op == 3 {
            b[2] = T::zero();
            b[6] = T::zero();
            PcuExecutionFaultKind::DivideByZero
        } else {
            a[2] = T::from(T::MAX + 1);
            a[6] = T::from(T::MAX + 1);
            PcuExecutionFaultKind::InvalidFloatingOperand
        };
        let ra = source::identity(a.as_slice()).unwrap();
        let rb = source::identity(b.as_slice()).unwrap();
        let result = call(&ra, &rb, op);
        assert!(
            matches!(result.as_ref().err().map(fault),Some((2,actual,false)) if actual==kind),
            "{result:?}"
        );
        read(&ra, &a);
        read(&rb, &b);
        let (a, b, want) = oracle::inputs::<T>(65, 51, op, policy);
        let ra = source::identity(a.as_slice()).unwrap();
        let rb = source::identity(b.as_slice()).unwrap();
        read(&call(&ra, &rb, op).unwrap(), &want);
    }
}
#[test]
fn four_low_formats_owned_pointwise() {
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for policy in [
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        ] {
            global::configure(global::PcuExecutionPolicy {
                backend: global::PcuBackendChoice::Cpu,
                device: None,
                numerical_mode: mode,
                float_underflow: policy,
                ..Default::default()
            })
            .unwrap();
            format::<fusion_pcu::PcuF16Bits>(policy);
            format::<fusion_pcu::PcuBf16Bits>(policy);
            format::<fusion_pcu::PcuF8E4M3FnBits>(policy);
            format::<fusion_pcu::PcuF8E5M2Bits>(policy);
        }
    }
}

fn fault(error: &PcuExecutionError) -> (u64, PcuExecutionFaultKind, bool) {
    match error {
        PcuExecutionError::ArithmeticFault(fault) => {
            (fault.invocation_id, fault.kind, fault.recovered)
        }
        PcuExecutionError::TensorBuild(TensorError::ArithmeticFault {
            element_index,
            kind,
            ..
        }) => (u64::try_from(*element_index).unwrap(), *kind, false),
        error => panic!("unexpected terminal error:{error:?}"),
    }
}
