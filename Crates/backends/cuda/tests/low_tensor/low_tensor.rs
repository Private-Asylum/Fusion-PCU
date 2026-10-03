//! Exact low-format owned source results, private failures, and captured numerical requirements.
extern crate pcu_facade as fusion_pcu;
#[path = "../../benches/low_tensor/oracle/oracle.rs"]
mod oracle;
#[path = "../../benches/low_tensor/source/source.rs"]
#[allow(dead_code)] // Remaining source functions serve the benchmark and canonical contract.
mod source;
#[rustfmt::skip]
use fusion_pcu::{global,PcuTensor,PcuExecutionError,PcuExecutionFaultKind,PcuFloatUnderflowPolicy,PcuNumericalMode};
use oracle::Format;
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
            matches!(result,Err(PcuExecutionError::ArithmeticFault(f)) if f.invocation_id==2&&f.kind==kind&&!f.recovered),
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
#[ignore = "actual GPU source bit oracle and owned publication law"]
fn four_low_formats_owned_pointwise() {
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for policy in [
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        ] {
            global::configure(global::PcuExecutionPolicy {
                backend: global::PcuBackendChoice::Cuda,
                device: Some(0),
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
#[test]
fn native_inspection_rejects_unproved_profiles() {
    use fusion_pcu::dialect::tensor::{Graph, TensorScalarValue};
    use fusion_pcu::{PcuScalarType, PcuNumericalOptions, PcuReproducibility};
    use fusion_pcu_cuda::lower_checked_float_tensor_to_cuda_source as lower;
    for dtype in [
        PcuScalarType::F16,
        PcuScalarType::BF16,
        PcuScalarType::F8E4M3FN,
        PcuScalarType::F8E5M2,
    ] {
        for policy in [
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        ] {
            let mut graph = Graph::default();
            let a = graph.input([65], dtype).unwrap();
            let b = graph.input([65], dtype).unwrap();
            assert!(lower(&graph, a).is_err());
            for value in [
                graph.add(a, b).unwrap(),
                graph.sub(a, b).unwrap(),
                graph.mul(a, b).unwrap(),
                graph.div(a, b).unwrap(),
                graph.relu(a).unwrap(),
            ] {
                graph
                    .set_value_float_underflow_policy(value, policy)
                    .unwrap();
                let source = lower(&graph, value).unwrap();
                assert!(source.contains("void fusion_kernel("));
                assert!(source.contains("fusion_fault"));
                assert!(!source.contains("#include <cuda"));
                graph
                    .set_value_numerical_options(
                        value,
                        PcuNumericalOptions {
                            reproducibility: PcuReproducibility::PortableV1,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                assert!(lower(&graph, value).is_err());
            }
        }
    }
    let mut graph = Graph::default();
    let a = graph.input([65], PcuScalarType::F16).unwrap();
    let uniform = graph
        .uniform_value(
            [65],
            TensorScalarValue::F16(fusion_pcu::PcuF16Bits::from_bits(0x3c00)),
        )
        .unwrap();
    let sum = graph.add(a, uniform).unwrap();
    assert!(lower(&graph, sum).is_err());
}
