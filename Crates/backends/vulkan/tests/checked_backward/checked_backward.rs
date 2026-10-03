//! Actual six-format exact selection, low4 exhaustive finite encodings, and owned failure/publication.
extern crate pcu_facade as fusion_pcu;
#[path = "../../../rocm/benches/checked_relu_backward/oracle/oracle.rs"]
mod oracle;
#[path = "../../../rocm/benches/checked_relu_backward/source/source.rs"]
mod source;
#[rustfmt::skip]
use fusion_pcu::{
    global,
    PcuTensor,
    PcuExecutionError,
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuCompoundArithmeticPolicy,
    PcuPrecisionPolicy,
};
use oracle::Format;
fn read<T: Format>(owner: &PcuTensor<T>, expected: &[T]) {
    let mut actual = vec![T::sentinel(); expected.len() + 2];
    owner.read_into(&mut actual).unwrap();
    for (a, b) in actual[..expected.len()].iter().zip(expected) {
        assert_eq!(a.bits(), b.bits());
    }
    for tail in &actual[expected.len()..] {
        assert_eq!(tail.bits(), T::sentinel().bits());
    }
}
fn fault<T: Format>(
    result: Result<PcuTensor<T>, PcuExecutionError>,
    index: u64,
    kind: PcuExecutionFaultKind,
) {
    let error = result.unwrap_err();
    let fault = error
        .arithmetic_fault()
        .unwrap_or_else(|| panic!("expected {kind:?} lane{index}, got {error:?}"));
    assert_eq!(fault.invocation_id, index);
    assert_eq!(fault.kind, kind);
    assert!(!fault.recovered);
}
#[allow(clippy::too_many_lines)] // Each profile keeps source, owner lifetimes and failure/retry proof in one scope.
fn execute<T: Format>(policy: PcuFloatUnderflowPolicy, low: bool) {
    eprintln!(
        "checked backward {} {policy:?} exhaustive_low={low}",
        T::LABEL
    );
    global::clear_thread_cache().unwrap();
    let (a, b, want) = oracle::inputs::<T>(257, 17, 0, policy);
    let ra = source::identity(a.as_slice()).unwrap();
    let rb = source::identity(b.as_slice()).unwrap();
    let previous = source::checked(&ra, &rb).unwrap();
    read(&previous, &want);
    read(&source::checked(a.as_slice(), b.as_slice()).unwrap(), &want);
    read(&source::checked(&ra, b.as_slice()).unwrap(), &want);
    read(
        &source::consume(source::checked(&ra, &rb).unwrap()).unwrap(),
        &want,
    );
    let (a2, b2, want2) = oracle::inputs::<T>(257, 51, 0, policy);
    read(
        &source::checked(a2.as_slice(), b2.as_slice()).unwrap(),
        &want2,
    );
    read(&previous, &want);
    let one = T::one();
    let negative = T::from(T::SIGN | T::ONE);
    let invalid = T::from(T::MAX + 1);
    let tiny = T::from(1);
    let negativezero = T::from(T::SIGN);
    for x in [one, negative, T::zero(), negativezero] {
        let mut input = vec![x; 17];
        let mut upstream = vec![one; 17];
        upstream[2] = invalid;
        upstream[6] = invalid;
        let ri = source::identity(input.as_slice()).unwrap();
        let ru = source::identity(upstream.as_slice()).unwrap();
        fault(
            source::checked(&ri, &ru),
            2,
            PcuExecutionFaultKind::InvalidFloatingOperand,
        );
        fault(
            source::discarded(&ri, &ru),
            2,
            PcuExecutionFaultKind::InvalidFloatingOperand,
        );
        read(&ri, &input);
        read(&ru, &upstream);
        read(&previous, &want);
        input[2] = invalid;
        upstream[2] = one;
        fault(
            source::checked(input.as_slice(), upstream.as_slice()),
            2,
            PcuExecutionFaultKind::InvalidFloatingOperand,
        );
        input[2] = one;
        upstream[6] = one;
        read(
            &source::checked(input.as_slice(), upstream.as_slice()).unwrap(),
            &input
                .iter()
                .zip(&upstream)
                .map(|(&x, &dy)| oracle::expected(x, dy, policy).unwrap())
                .collect::<Vec<_>>(),
        );
    }
    // Both signed-zero inputs select +0; selected upstream -0 is copied exactly.
    read(
        &source::checked(&[T::zero(), negativezero, one][..], &[negativezero; 3][..]).unwrap(),
        &[T::zero(), T::zero(), negativezero],
    );
    read(
        &source::tight(&[negative][..], &[tiny][..]).unwrap(),
        &[T::zero()],
    );
    read(&source::tight(&[tiny][..], &[one][..]).unwrap(), &[one]);
    fault(
        source::tight(&[one][..], &[tiny][..]),
        0,
        PcuExecutionFaultKind::ArithmeticUnderflow,
    );
    read(&source::allow(&[one][..], &[tiny][..]).unwrap(), &[tiny]);
    // First logical fault wins across kinds: invalid lane6 cannot replace underflow lane2.
    let input = [one; 17];
    let mut upstream = [one; 17];
    upstream[2] = tiny;
    upstream[6] = invalid;
    if policy == PcuFloatUnderflowPolicy::RejectSubnormalResult {
        fault(
            source::checked(&input[..], &upstream[..]),
            2,
            PcuExecutionFaultKind::ArithmeticUnderflow,
        );
    } else {
        fault(
            source::checked(&input[..], &upstream[..]),
            6,
            PcuExecutionFaultKind::InvalidFloatingOperand,
        );
    }
    if low {
        // Every finite encoding is used as input and upstream, with an independent bit oracle.
        let finite = (0..T::SIGN * 2)
            .filter(|bits| bits & (T::SIGN - 1) <= T::MAX)
            .map(T::from)
            .collect::<Vec<_>>();
        let dy = finite
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let upstream = finite[(i * 37 + 17) % finite.len()];
                if oracle::expected(x, upstream, policy).is_ok() {
                    upstream
                } else {
                    one
                }
            })
            .collect::<Vec<_>>();
        let exact = finite
            .iter()
            .zip(&dy)
            .map(|(&x, &upstream)| oracle::expected(x, upstream, policy).unwrap())
            .collect::<Vec<_>>();
        read(
            &source::checked(finite.as_slice(), dy.as_slice()).unwrap(),
            &exact,
        );
        // NaN-only E4M3FN and every infinity/NaN encoding in the IEEE formats are invalid,
        // including inactive branches. Vectorize all invalid payloads without relying on scan order.
        let bad = (0..T::SIGN * 2)
            .filter(|bits| bits & (T::SIGN - 1) > T::MAX)
            .map(T::from)
            .collect::<Vec<_>>();
        for invalid in bad {
            fault(
                source::checked(&[invalid][..], &[one][..]),
                0,
                PcuExecutionFaultKind::InvalidFloatingOperand,
            );
            fault(
                source::checked(&[negative][..], &[invalid][..]),
                0,
                PcuExecutionFaultKind::InvalidFloatingOperand,
            );
        }
        read(&source::permitted(&ra, &rb).unwrap(), &want);
        assert!(source::portable(&[one][..], &[one][..]).is_err());
    }
    global::clear_thread_cache().unwrap();
    read(&previous, &want);
    read(&ra, &a);
    read(&rb, &b);
}
#[test]
#[ignore = "actual GPU checked derivative raw-bit and owned lifetime proof"]
fn six_formats_checked_backward_and_low_permissions() {
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for policy in [
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
        ] {
            for compound in [
                PcuCompoundArithmeticPolicy::Checked,
                PcuCompoundArithmeticPolicy::BackendDefined,
            ] {
                for precision in [
                    PcuPrecisionPolicy::Preserve,
                    PcuPrecisionPolicy::BackendOptimized,
                ] {
                    global::configure(global::PcuExecutionPolicy {
                        backend: global::PcuBackendChoice::Vulkan,
                        device: Some(0),
                        numerical_mode: mode,
                        float_underflow: policy,
                        numerical_options: PcuNumericalOptions {
                            compound_arithmetic: compound,
                            precision,
                            ..Default::default()
                        },
                        ..Default::default()
                    })
                    .unwrap();
                    macro_rules! low {
                        ($ty:ty) => {
                            execute::<$ty>(policy, true);
                        };
                    }
                    low!(fusion_pcu::PcuF16Bits);
                    low!(fusion_pcu::PcuBf16Bits);
                    low!(fusion_pcu::PcuF8E4M3FnBits);
                    low!(fusion_pcu::PcuF8E5M2Bits);
                    execute::<f32>(policy, false);
                    execute::<f64>(policy, false);
                }
            }
        }
    }
}

#[path = "../scalar_transport/device/device.rs"]
mod device;
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    Graph,
    TensorElement,
    TensorError,
};
#[rustfmt::skip]
use fusion_pcu_vulkan::{
    PcuVulkanBackend,
    PcuVulkanPreparedTensorGraph,
    PcuVulkanTensorInput,
    PcuVulkanTensorError,
    PcuVulkanError,
};

fn native<T: Format + TensorElement>(backend: &PcuVulkanBackend, policy: PcuFloatUnderflowPolicy) {
    let (a, b, want) = oracle::inputs::<T>(17, 17, 0, policy);
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    graph.set_numerical_options(PcuNumericalOptions {
        compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
        precision: PcuPrecisionPolicy::BackendOptimized,
        ..Default::default()
    });
    let left = graph.input([17], T::TYPE).unwrap();
    let right = graph.input([17], T::TYPE).unwrap();
    let output = graph.relu_backward(left, right).unwrap();
    graph
        .set_value_float_underflow_policy(output, policy)
        .unwrap();
    let mut plan = PcuVulkanPreparedTensorGraph::<T>::prepare(backend, &graph, &[output]).unwrap();
    drop(graph);
    let left = backend.upload_owned(&a).unwrap();
    let right = backend.upload_owned(&b).unwrap();
    let escaped = plan
        .execute_owned(&[
            PcuVulkanTensorInput::Owned(&left),
            PcuVulkanTensorInput::Owned(&right),
        ])
        .unwrap();
    let mut actual = vec![T::sentinel(); 19];
    escaped.read_into(&mut actual).unwrap();
    for (&x, &y) in actual[..17].iter().zip(&want) {
        assert_eq!(x.bits(), y.bits());
    }
    for tail in &actual[17..] {
        assert_eq!(tail.bits(), T::sentinel().bits());
    }
    for args in [
        [
            PcuVulkanTensorInput::Host(&a),
            PcuVulkanTensorInput::Host(&b),
        ],
        [
            PcuVulkanTensorInput::Owned(&left),
            PcuVulkanTensorInput::Host(&b),
        ],
    ] {
        plan.execute_owned(&args)
            .unwrap()
            .read_into(&mut actual)
            .unwrap();
        for (&x, &y) in actual[..17].iter().zip(&want) {
            assert_eq!(x.bits(), y.bits());
        }
    }
    let mut bad = b.clone();
    bad[2] = T::from(T::MAX + 1);
    assert!(
        matches!(plan.execute_owned(&[PcuVulkanTensorInput::Owned(&left),
        PcuVulkanTensorInput::Host(&bad)]),Err(PcuVulkanTensorError::Graph(
            TensorError::ArithmeticFault {value,element_index:2,
                kind:PcuExecutionFaultKind::InvalidFloatingOperand})) if value==output)
    );
    assert!(
        plan.execute_owned(&[
            PcuVulkanTensorInput::Host(&a),
            PcuVulkanTensorInput::Host(&b[..16])
        ])
        .is_err()
    );
    let (other, _) = device::selected();
    let foreign = other.upload_owned(&b).unwrap();
    assert!(matches!(
        plan.execute_owned(&[
            PcuVulkanTensorInput::Host(&a),
            PcuVulkanTensorInput::Owned(&foreign)
        ]),
        Err(PcuVulkanTensorError::Native(
            PcuVulkanError::InvalidArguments
        ))
    ));
    for (owner, want) in [
        (&left, a.as_slice()),
        (&right, b.as_slice()),
        (&escaped, want.as_slice()),
    ] {
        owner.read_into(&mut actual).unwrap();
        for (&x, &y) in actual[..17].iter().zip(want) {
            assert_eq!(x.bits(), y.bits());
        }
    }
    plan.execute_owned(&[
        PcuVulkanTensorInput::Owned(&left),
        PcuVulkanTensorInput::Owned(&right),
    ])
    .unwrap()
    .read_into(&mut actual)
    .unwrap();
    for (&x, &y) in actual[..17].iter().zip(&want) {
        assert_eq!(x.bits(), y.bits());
    }
}
#[test]
#[ignore = "requires actual Vulkan GPU; exact same-session derivative and private fault retry"]
fn six_formats_native_same_session_and_private_fault_retry() {
    let (backend, _) = device::selected();
    for policy in [
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
    ] {
        native::<fusion_pcu::PcuF16Bits>(&backend, policy);
        native::<fusion_pcu::PcuBf16Bits>(&backend, policy);
        native::<fusion_pcu::PcuF8E4M3FnBits>(&backend, policy);
        native::<fusion_pcu::PcuF8E5M2Bits>(&backend, policy);
        native::<f32>(&backend, policy);
        native::<f64>(&backend, policy);
    }
}
