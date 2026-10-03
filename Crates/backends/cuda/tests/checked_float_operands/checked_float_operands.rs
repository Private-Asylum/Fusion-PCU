//! Genuine six-format repeated-input source, publication and unread-owner boundary.
extern crate pcu_facade as fusion_pcu;
#[path = "../../benches/checked_relu_backward/oracle/oracle.rs"]
#[allow(dead_code)] // Derivative-only helpers are not part of this representation fixture.
mod oracle;
#[path = "../../benches/checked_float_operands/source/source.rs"]
#[allow(dead_code)] // The paired benchmark uses the remaining authored schema functions.
mod source;
#[rustfmt::skip]
use fusion_pcu::{global,PcuTensor,PcuExecutionFaultKind,PcuNumericalMode,
    PcuNumericalOptions,PcuCompoundArithmeticPolicy,PcuPrecisionPolicy,
    PcuFloatUnderflowPolicy,PcuRangePolicy,PcuExecutionError};
use oracle::Format;
fn bits<T: Format>(a: &[T], b: &[T]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert_eq!(a.bits(), b.bits());
    }
}
fn fault_at(index: u64, result: Result<(), PcuExecutionError>, kind: PcuExecutionFaultKind) {
    let error = result.unwrap_err();
    let fault = error
        .arithmetic_fault()
        .unwrap_or_else(|| panic!("{error:?}"));
    assert_eq!(fault.invocation_id, index);
    assert_eq!(fault.kind, kind);
    assert!(!fault.recovered);
}
fn read<T: Format>(tensor: &PcuTensor<T>, want: &[T]) {
    let mut actual = vec![T::sentinel(); want.len() + 2];
    tensor.read_into(&mut actual).unwrap();
    bits(&actual[..want.len()], want);
    bits(&actual[want.len()..], &[T::sentinel(); 2]);
}
#[allow(clippy::too_many_lines)] // Keep each real source/owner/rollback/retry scope together.
fn execute<T: Format>(policy: global::PcuExecutionPolicy) {
    macro_rules! trace {
        ($stage:literal) => {
            if std::env::var_os("PCU_OPERANDS_TRACE").is_some() {
                eprintln!("operand-stage/{}/{:?}: {}", T::LABEL, policy, $stage);
            }
        };
    }
    trace!("begin");
    let range = policy.range_policy;
    global::clear_thread_cache().unwrap();
    let one = T::one();
    let negative = T::from(T::SIGN | T::ONE);
    let bank = [one, negative, one, negative, one, negative, one];
    let want = [one; 7];
    let mut host = [T::sentinel(); 9];
    let empty: &[T] = &[];
    trace!("host-unused");
    source::mul::unused::<T, 7>(empty, &mut host, &bank).unwrap();
    bits(&host[..7], &want);
    bits(&host[7..], &[T::sentinel(); 2]);
    trace!("host-single");
    source::mul::single::<T, 7>(&mut host, &bank).unwrap();
    bits(&host[..7], &want);
    trace!("host-grid");
    source::mul::grid::<T, 7>(empty, &mut host, &bank).unwrap();
    bits(&host[..7], &want);
    trace!("host-seed");
    source::mul::seed::<T, 7>(empty, &mut host, &negative).unwrap();
    bits(&host[..7], &want);
    let mut bad = bank;
    bad[2] = T::zero();
    bad[6] = T::zero();
    // All active resident owners share one preparation session. Representation identity
    // is prepared under Reject once, then the original scalar range tuple is restored.
    if range == PcuRangePolicy::Clamp {
        global::configure(global::PcuExecutionPolicy {
            range_policy: PcuRangePolicy::Reject,
            ..policy
        })
        .unwrap();
    }
    trace!("resident-identity-left");
    let left = source::identity(&bank).unwrap();
    trace!("resident-identity-output");
    let mut output = source::identity(&host).unwrap();
    trace!("resident-identity-bad");
    let bad_owner = source::identity(&bad).unwrap();
    trace!("resident-identity-fresh");
    let mut fresh = source::identity(&[T::sentinel(); 9]).unwrap();
    if range == PcuRangePolicy::Clamp {
        global::configure(policy).unwrap();
    }

    trace!("resident-unused");
    source::mul::unused::<T, 7>(empty, &mut output, &left).unwrap();
    read(&output, &host);
    // A failed mutable output is unreadable, but an unused readonly declaration must
    // neither probe that owner nor acquire an affinity/access lease from it.

    fault_at(
        2,
        source::div::unused::<T, 7>(empty, &mut output, &bad_owner),
        PcuExecutionFaultKind::DivideByZero,
    );
    trace!("resident-fault-returned");
    let before_read = host;
    assert!(output.read_into(&mut host).is_err());
    bits(&host, &before_read);

    trace!("resident-discarded-unread");
    source::mul::unused::<T, 7>(&output, &mut fresh, &left).unwrap();
    read(
        &fresh,
        &[
            one,
            one,
            one,
            one,
            one,
            one,
            one,
            T::sentinel(),
            T::sentinel(),
        ],
    );
    // The exact original declaration/type/access remains mandatory in raw prepared ABI;
    // generic source authoring normalizes unread owners before carrier conversion.
    let prior = host;
    fault_at(
        2,
        source::div::grid::<T, 7>(empty, &mut host, &bad),
        PcuExecutionFaultKind::DivideByZero,
    );
    bits(&host, &prior);
    source::div::unused::<T, 7>(empty, &mut host, &bank).unwrap();
    bits(&host[..7], &want);
    let mut invalid = bank;
    invalid[2] = T::from(T::MAX + 1);
    invalid[6] = T::from(T::MAX + 1);
    let prior = host;
    fault_at(
        2,
        source::mul::unused::<T, 7>(empty, &mut host, &invalid),
        PcuExecutionFaultKind::InvalidFloatingOperand,
    );
    bits(&host, &prior);
    source::mul::unused::<T, 7>(&invalid, &mut host, &bank).unwrap();
    bits(&host[..7], &want);
    if range == PcuRangePolicy::Clamp {
        let mut wide = bank;
        wide[2] = T::from(T::MAX);
        let error = source::add::unused::<T, 7>(empty, &mut host, &wide).unwrap_err();
        let f = error.arithmetic_fault().unwrap();
        assert_eq!(f.invocation_id, 2);
        assert_eq!(f.kind, PcuExecutionFaultKind::ArithmeticOverflow);
        assert!(f.recovered);
        assert_eq!(host[2].bits(), T::MAX);
        wide[6] = T::from(T::MAX + 1);
        let prior = host;
        fault_at(
            6,
            source::add::unused::<T, 7>(empty, &mut host, &wide),
            PcuExecutionFaultKind::InvalidFloatingOperand,
        );
        bits(&host, &prior);
    }
    // Actual foreign CPU-owned carrier is unused: active input/output stay on this GPU.
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        device: None,
        range_policy: PcuRangePolicy::Reject,
        ..policy
    })
    .unwrap();
    trace!("foreign-cpu-create");
    let foreign = source::identity(&bank).unwrap();
    global::configure(policy).unwrap();
    trace!("foreign-cpu-unused");
    source::mul::unused::<T, 7>(&foreign, &mut fresh, &left).unwrap();
    read(
        &fresh,
        &[
            one,
            one,
            one,
            one,
            one,
            one,
            one,
            T::sentinel(),
            T::sentinel(),
        ],
    );
    trace!("foreign-cpu-read");
    read(&foreign, &bank);
    trace!("gpu-left-read");
    read(&left, &bank);
}
#[test]
#[ignore = "actual GPU source/readiness/publication proof; serial correctness window"]
fn six_formats_repeated_source_and_unread_discarded_owner() {
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for compound in [
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
                    for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                        let configuration = global::PcuExecutionPolicy {
                            backend: global::PcuBackendChoice::Cuda,
                            device: Some(0),
                            numerical_mode: mode,
                            numerical_options: PcuNumericalOptions {
                                compound_arithmetic: compound,
                                precision,
                                ..Default::default()
                            },
                            float_underflow: policy,
                            range_policy: range,
                            ..Default::default()
                        };
                        global::configure(configuration).unwrap();
                        execute::<f32>(configuration);
                        execute::<f64>(configuration);
                        execute::<fusion_pcu::PcuF16Bits>(configuration);
                        execute::<fusion_pcu::PcuBf16Bits>(configuration);
                        execute::<fusion_pcu::PcuF8E4M3FnBits>(configuration);
                        execute::<fusion_pcu::PcuF8E5M2Bits>(configuration);
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "actual GPU narrow unread-owner smoke; serial correctness only"]
fn empty_unused_and_discarded_foreign_owner_smoke() {
    let configuration = global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cuda,
        device: Some(0),
        ..Default::default()
    };
    global::configure(configuration).unwrap();
    execute::<fusion_pcu::PcuF16Bits>(configuration);
}

#[test]
#[ignore = "actual bounded six-format diagnostic; test-only stage breadcrumbs"]
fn repeated_operand_diagnostic() {
    let configuration = global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cuda,
        device: Some(0),
        ..Default::default()
    };
    global::configure(configuration).unwrap();
    match std::env::var("PCU_OPERANDS_FORMAT")
        .unwrap_or_else(|_| "f32".to_owned())
        .as_str()
    {
        "f32" => execute::<f32>(configuration),
        "f64" => execute::<f64>(configuration),
        "f16" => execute::<fusion_pcu::PcuF16Bits>(configuration),
        "bf16" => execute::<fusion_pcu::PcuBf16Bits>(configuration),
        "e4m3fn" => execute::<fusion_pcu::PcuF8E4M3FnBits>(configuration),
        "e5m2" => execute::<fusion_pcu::PcuF8E5M2Bits>(configuration),
        value => panic!("unknown diagnostic format {value}"),
    }
}
