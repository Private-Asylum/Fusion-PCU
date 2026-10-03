//! Full raw-bit payload checks and ordinary-source policy specialization on Apple silicon.
use core::sync::atomic::AtomicUsize;
use core::sync::atomic::Ordering;
#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuBf16Bits,
    PcuCheckedFloat,
    PcuExecutionFaultKind,
    PcuF16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
};

#[pcu(invocations = 3)]
fn grid<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = -input[id];
        id += stride;
    }
}

#[pcu(invocations = N)]
fn broadcast<T: PcuCheckedFloat, const N: usize>(input: &T, output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(*input);
}

#[pcu(invocations = N, flag(strict), flag(clamp_range), flag(reject_subnormal_result))]
fn clamp<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}

static SCORES: AtomicUsize = AtomicUsize::new(0);
fn score(_: &global::PcuInvocationCandidate<'_>) -> i128 {
    SCORES.fetch_add(1, Ordering::Relaxed);
    1
}

macro_rules! format {
    ($ty:ty, $sign:expr, $normal:expr, $invalid:expr) => {{
        for phase in 0..3 {
            let raw = [0, $sign, 1 + phase, $sign | (1 + phase), $normal + phase];
            let input = raw.map(<$ty>::from_bits);
            let sentinel = <$ty>::from_bits($normal);
            let mut output = [sentinel; 7];
            for execute in [super::neg::<$ty, 5>, grid::<$ty, 5>] {
                execute(&input, &mut output).unwrap();
                assert_eq!(output[..5].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    raw.map(|v| v ^ $sign));
                assert_eq!(output[5..], [sentinel; 2]);
            }
            broadcast::<$ty, 5>(&input[4], &mut output).unwrap();
            assert_eq!(output[..5], [input[4]; 5]);
            broadcast::<$ty, 5>(&input[3], &mut output).unwrap();
            assert_eq!(output[..5], [<$ty>::from_bits(0); 5]);
            let error = clamp::<$ty, 5>(&input, &mut output).unwrap_err();
            let recovery = error.recovered_range_fault().unwrap();
            assert_eq!(recovery.kind, PcuExecutionFaultKind::ArithmeticUnderflow);
            assert_eq!(recovery.invocation_id, 2);
            assert_eq!(output[..5].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                raw.map(|v| v ^ $sign));
            assert_eq!(output[5..], [sentinel; 2]);
            let mut invalid = input;
            invalid[4] = <$ty>::from_bits($invalid);
            output.fill(sentinel);
            let error = clamp::<$ty, 5>(&invalid, &mut output).unwrap_err();
            assert!(matches!(error, global::PcuExecutionError::ArithmeticFault(fault)
                if fault.kind == PcuExecutionFaultKind::InvalidFloatingOperand
                    && fault.invocation_id == 4 && !fault.recovered));
            assert_eq!(output, [sentinel; 7]);
            super::neg::<$ty, 5>(&input, &mut output).unwrap();
        }
    }};
}

#[allow(clippy::cognitive_complexity)] // Four macro-expanded format fixtures exercise the same native contract.
fn formats() {
    format!(PcuF16Bits, 0x8000, 0x0400, 0x7c00);
    format!(PcuBf16Bits, 0x8000, 0x0080, 0x7f80);
    format!(PcuF8E4M3FnBits, 0x80, 0x08, 0x7f);
    format!(PcuF8E5M2Bits, 0x80, 0x04, 0x7c);
}

pub fn verify() {
    let _guard = super::POLICY_LOCK.lock().unwrap();
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Mlx,
        score_invocation: Some(score),
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
    formats();
    let cold = SCORES.load(Ordering::Relaxed);
    assert!(
        cold > 0,
        "the genuine ordinary route must enumerate an MLX candidate"
    );
    formats();
    assert_eq!(SCORES.load(Ordering::Relaxed), cold);
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
