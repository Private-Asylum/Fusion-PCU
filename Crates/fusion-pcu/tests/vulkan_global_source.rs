//! The ordinary source call uses statically selected Vulkan without a backend object.
#![cfg(feature = "vulkan")]

use core::sync::atomic::AtomicUsize;
use core::sync::atomic::Ordering;
#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuExecutionFault,
    PcuExecutionFaultKind,
    PcuNumericalMode,
};

static SCORES: AtomicUsize = AtomicUsize::new(0);

fn score(candidate: &global::PcuInvocationCandidate<'_>) -> i128 {
    assert_eq!(candidate.device.reference.provider.0, 0x564b_4c31);
    assert_eq!(candidate.kernel.entry.logical_shape, [17, 1, 1]);
    // Capacity has not been exposed by this provider; unknown is not a guessed zero.
    assert_eq!(candidate.total_memory_bytes, None);
    SCORES.fetch_add(1, Ordering::Relaxed);
    i128::from(candidate.device.reference.id)
}

#[pcu(invocations = N)]
fn negate<const N: usize>(input: &[f32], output: &mut [f32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}

#[pcu(invocations = N)]
fn transport<const N: usize>(input: &[f32], output: &mut [f32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id];
}

#[pcu(invocations = N, flag(reject_subnormal_result))]
fn tight_negate<const N: usize>(input: &[f32], output: &mut [f32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}

#[pcu(invocations = N)]
fn unsupported_add<const N: usize>(input: &[f32], output: &mut [f32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] + 1.0;
}

fn configure() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        numerical_mode: PcuNumericalMode::Strict,
        score_invocation: Some(score),
        ..Default::default()
    })
    .unwrap();
}

#[test]
#[ignore = "requires an idle Vulkan GPU; run serially"]
fn ordinary_calls_reuse_cold_selection_and_report_transactional_faults() {
    configure();
    global::clear_thread_cache().unwrap();
    SCORES.store(0, Ordering::Relaxed);
    let input = [
        0,
        0x8000_0000,
        1,
        0x8000_0001,
        0x3f80_0000,
        0xbf80_0000,
        0x7f7f_ffff,
        0xff7f_ffff,
        17,
        31,
        257,
        1025,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0001,
        0x7f80_0001,
        0xff80_0001,
    ]
    .map(f32::from_bits);
    let mut output = [19.0_f32; 20];
    transport::<17>(&input, &mut output).unwrap();
    assert_eq!(
        output[..17].iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        input.map(f32::to_bits)
    );
    assert_eq!(
        output[17..].iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        [19.0_f32; 3].map(f32::to_bits)
    );
    negate::<17>(&[2.0; 17], &mut output).unwrap();
    let cold_scores = SCORES.load(Ordering::Relaxed);
    assert!(cold_scores > 0);
    for phase in 1..4_u16 {
        let current = core::array::from_fn::<_, 17, _>(|index| {
            f32::from(u16::try_from(index).unwrap() + phase)
        });
        transport::<17>(&current, &mut output).unwrap();
        assert_eq!(SCORES.load(Ordering::Relaxed), cold_scores);
        negate::<17>(&current, &mut output).unwrap();
        for (value, actual) in current.iter().zip(&output) {
            assert_eq!(actual.to_bits(), value.to_bits() ^ 0x8000_0000);
        }
    }
    checked_faults(&mut output);
    // A new configuration generation forces cold selection for the same specialization.
    let before = SCORES.load(Ordering::Relaxed);
    configure();
    transport::<17>(&input, &mut output).unwrap();
    assert!(SCORES.load(Ordering::Relaxed) > before);
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

fn checked_faults(output: &mut [f32; 20]) {
    let mut input = [1.0_f32; 17];
    input[3] = f32::INFINITY;
    input[16] = f32::NAN;
    let previous = output.map(f32::to_bits);
    assert!(matches!(
        negate::<17>(&input, output),
        Err(global::PcuExecutionError::ArithmeticFault(
            PcuExecutionFault {
                recovered: false,
                kind: PcuExecutionFaultKind::InvalidFloatingOperand,
                invocation_id: 3,
            }
        ))
    ));
    assert_eq!(output.map(f32::to_bits), previous);
    input.fill(f32::from_bits(1));
    negate::<17>(&input, output).unwrap(); // IEEE exact subnormals are accepted.
    let previous = output.map(f32::to_bits);
    assert!(matches!(
        tight_negate::<17>(&input, output),
        Err(global::PcuExecutionError::ArithmeticFault(
            PcuExecutionFault {
                recovered: false,
                kind: PcuExecutionFaultKind::ArithmeticUnderflow,
                invocation_id: 0,
            }
        ))
    ));
    assert_eq!(output.map(f32::to_bits), previous);
    let failure = unsupported_add::<17>(&input, output).unwrap_err();
    let global::PcuExecutionError::NoCompatibleInvocationDevice { rejected, .. } = failure else {
        panic!("cold source admission error")
    };
    assert!(!rejected.is_empty());
    assert!(
        rejected
            .iter()
            .all(|(device, error)| device.provider.0 == 0x564b_4c31
                && matches!(error, global::PcuExecutionError::VulkanExecution(_)))
    );
    assert_eq!(output.map(f32::to_bits), previous);
    input.fill(2.0);
    negate::<17>(&input, output).unwrap();
    assert_eq!(
        output[..17].iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        [-2.0_f32; 17].map(f32::to_bits)
    );
}
