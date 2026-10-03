//! Independent wide bytes and fatal effect witnesses through whole owned source.
extern crate pcu_facade as fusion_pcu;
#[path = "../../benches/wide_integer/oracle/oracle.rs"]
mod oracle;
#[path = "../../benches/wide_tensor/source/source.rs"]
#[allow(dead_code)] // Canonical native benchmark shares the source functions.
mod source;
#[rustfmt::skip]
use fusion_pcu::{
    global,
    PcuTensor,
    PcuExecutionError,
    PcuExecutionFaultKind,
    PcuNumericalMode,
};
use oracle::Format;
fn host<T: Format>(a: &[T], b: &[T], op: u32) -> Result<PcuTensor<T>, PcuExecutionError> {
    match op {
        0 => source::add(a, b),
        1 => source::sub(a, b),
        2 => source::mul(a, b),
        _ => unreachable!(),
    }
}
fn resident<T: Format>(
    a: &PcuTensor<T>,
    b: &PcuTensor<T>,
    op: u32,
) -> Result<PcuTensor<T>, PcuExecutionError> {
    match op {
        0 => source::add(a, b),
        1 => source::sub(a, b),
        2 => source::mul(a, b),
        _ => unreachable!(),
    }
}
fn read<T: Format>(owner: &PcuTensor<T>, expected: &[T]) {
    let mut out = vec![T::sentinel(); expected.len() + 2];
    owner.read_into(&mut out).unwrap();
    oracle::verify(expected, &out);
}
fn format<T: Format>() {
    for op in 0..3 {
        let mut previous: Option<(PcuTensor<T>, Vec<T>)> = None;
        for phase in [1, 17, 51] {
            let (a, b, want) = oracle::inputs::<T>(65, phase, op);
            let output = host(&a, &b, op).unwrap();
            read(&output, &want);
            let ra = source::identity(a.as_slice()).unwrap();
            let rb = source::identity(b.as_slice()).unwrap();
            let borrowed = resident(&ra, &rb, op).unwrap();
            read(&borrowed, &want);
            let mixed = match op {
                0 => source::add(&ra, b.as_slice()),
                1 => source::sub(&ra, b.as_slice()),
                2 => source::mul(&ra, b.as_slice()),
                _ => unreachable!(),
            }
            .unwrap();
            read(&mixed, &want);
            let consumed = source::consumed_identity(output).unwrap();
            read(&consumed, &want);
            if let Some((owner, old)) = &previous {
                read(owner, old);
            }
            previous = Some((borrowed, want));
        }
        let (mut a, mut b, want) = oracle::inputs::<T>(65, 1, op);
        let rows = oracle::rows::<T>(op);
        let bad = rows.iter().find(|row| row.3 != 0).unwrap();
        a[2] = bad.0;
        b[2] = bad.1;
        let ra = source::identity(a.as_slice()).unwrap();
        let rb = source::identity(b.as_slice()).unwrap();
        let result = resident(&ra, &rb, op);
        let kind = if bad.3 == 3 {
            PcuExecutionFaultKind::ArithmeticOverflow
        } else {
            PcuExecutionFaultKind::ArithmeticUnderflow
        };
        assert!(
            matches!(result,Err(PcuExecutionError::ArithmeticFault(f)) if f.invocation_id==2&&f.kind==kind&&!f.recovered),
            "{result:?}"
        );
        read(&ra, &a);
        read(&rb, &b);
        let (a, b, _) = oracle::inputs::<T>(65, 1, op);
        read(&host(&a, &b, op).unwrap(), &want);
        if op == 0 {
            let bad = rows.iter().find(|row| row.3 != 0).unwrap();
            let mut a = a;
            let mut b = b;
            a[2] = bad.0;
            b[2] = bad.1;
            assert!(source::discarded_add(&a, &b).is_err());
        }
    }
    let (mut a, mut b, want) = oracle::inputs::<T>(65, 17, 2);
    read(&source::permitted_mul(&a, &b).unwrap(), &want);
    let rows = oracle::rows::<T>(2);
    let bad = rows.iter().find(|row| row.3 != 0).unwrap();
    a[2] = bad.0;
    b[2] = bad.1;
    assert!(
        matches!(source::permitted_mul(&a,&b),Err(PcuExecutionError::ArithmeticFault(f)) if f.invocation_id==2&&!f.recovered)
    );
}
#[test]
#[ignore = "authorized actual GPU, dense tensor Reject exact bytes and lifetime law"]
fn six_wide_tensor_formats_checked_and_owned() {
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        global::configure(global::PcuExecutionPolicy {
            backend: global::PcuBackendChoice::Cuda,
            device: Some(0),
            numerical_mode: mode,
            ..Default::default()
        })
        .unwrap();
        global::clear_thread_cache().unwrap();
        format::<i128>();
        format::<u128>();
        format::<fusion_pcu::PcuI256>();
        format::<fusion_pcu::PcuU256>();
        format::<fusion_pcu::PcuI512>();
        format::<fusion_pcu::PcuU512>();
    }
}

#[test]
fn native_inspection_rejects_unproved_profiles_and_preserves_exact_widths() {
    use fusion_pcu::dialect::tensor::{Graph, TensorScalarValue};
    use fusion_pcu::{PcuScalarType, PcuNumericalOptions, PcuReproducibility};
    use fusion_pcu_cuda::lower_checked_integer_tensor_to_cuda_source as lower;
    for dtype in [
        PcuScalarType::I8,
        PcuScalarType::U8,
        PcuScalarType::I16,
        PcuScalarType::U16,
        PcuScalarType::I32,
        PcuScalarType::U32,
        PcuScalarType::I64,
        PcuScalarType::U64,
        PcuScalarType::I128,
        PcuScalarType::U128,
        PcuScalarType::I256,
        PcuScalarType::U256,
        PcuScalarType::I512,
        PcuScalarType::U512,
    ] {
        for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
            let mut graph = Graph::default();
            graph.set_numerical_mode(mode);
            let a = graph.input([65], dtype).unwrap();
            let b = graph.input([65], dtype).unwrap();
            assert!(lower(&graph, a).is_err());
            for value in [
                graph.add(a, b).unwrap(),
                graph.sub(a, b).unwrap(),
                graph.mul(a, b).unwrap(),
            ] {
                let source = lower(&graph, value).unwrap();
                assert!(source.contains("void fusion_kernel("));
                assert!(source.contains("fusion_fault"));
                if dtype.bit_width() > 64 {
                    assert!(source.contains(&format!("FusionBits{}", dtype.bit_width())));
                }
            }
        }
    }
    let mut graph = Graph::default();
    let a = graph.input([65], PcuScalarType::F32).unwrap();
    let b = graph.input([65], PcuScalarType::F32).unwrap();
    let value = graph.add(a, b).unwrap();
    assert!(lower(&graph, value).is_err());
    let mut graph = Graph::default();
    let a = graph.input([65], PcuScalarType::I128).unwrap();
    let b = graph
        .uniform_value([65], TensorScalarValue::I128(1))
        .unwrap();
    let value = graph.add(a, b).unwrap();
    assert!(lower(&graph, value).is_err());
    let mut graph = Graph::default();
    graph.set_numerical_options(PcuNumericalOptions {
        reproducibility: PcuReproducibility::PortableV1,
        ..Default::default()
    });
    let a = graph.input([65], PcuScalarType::I128).unwrap();
    let b = graph.input([65], PcuScalarType::I128).unwrap();
    let value = graph.mul(a, b).unwrap();
    assert!(lower(&graph, value).is_err());
}
