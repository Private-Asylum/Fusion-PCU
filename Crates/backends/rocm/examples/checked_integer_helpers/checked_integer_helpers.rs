//! Genuine integer helper capture, explicitly selecting rocm and preserving failure outputs.
extern crate pcu_facade as fusion_pcu;
use fusion_pcu::pcu;

#[pcu]
#[allow(clippy::missing_const_for_fn)] // Scalar companion participates in checked source lowering.
fn polynomial(mut value: u64, seed: u64) -> u64 {
    let original: u64 = value;
    value += seed;
    value *= original;
    value - original
}

#[pcu(invocations = N)]
fn direct<const N: usize>(input: &[u64], seed: &u64, output: &mut [u64]) {
    let id = pcu::context::global_invocation_id();
    output[id] = polynomial(input[id], *seed);
}

#[pcu(invocations = 3)]
fn grid<const N: usize>(input: &[u64], seed: &u64, output: &mut [u64]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = polynomial(input[id], *seed);
        id += stride;
    }
}

fn main() {
    const N: usize = 65;
    fusion_pcu::global::configure(fusion_pcu::global::PcuExecutionPolicy {
        backend: fusion_pcu::global::PcuBackendChoice::Rocm,
        ..Default::default()
    })
    .unwrap();
    let mut output = [17_u64; N + 2];
    for phase in [0_u64, 17, 83] {
        let mut input: [u64; N] =
            core::array::from_fn(|index| 1 + (u64::try_from(index).unwrap() + phase) % 97);
        direct::<N>(&input, &1, &mut output).unwrap();
        assert_eq!(output[..N], input.map(|value| value * value));
        assert_eq!(output[N..], [17; 2]);
        grid::<N>(&input, &1, &mut output).unwrap();
        assert_eq!(output[..N], input.map(|value| value * value));
        let before = output;
        input[7] = u64::MAX;
        assert!(matches!(direct::<N>(&input, &1, &mut output),
            Err(fusion_pcu::PcuExecutionError::ArithmeticFault(fault))
                if fault.invocation_id == 7
                    && fault.kind == fusion_pcu::PcuExecutionFaultKind::ArithmeticOverflow
                    && !fault.recovered));
        assert_eq!(output, before);
        input[7] = 2;
        direct::<N>(&input, &1, &mut output).unwrap();
        assert_eq!(output[..N], input.map(|value| value * value));
        assert_eq!(output[N..], [17; 2]);
    }
    fusion_pcu::global::clear_thread_cache().unwrap();
    println!(
        "rocm integer helper direct/grid, changing inputs, overflow rollback and retry passed"
    );
}
