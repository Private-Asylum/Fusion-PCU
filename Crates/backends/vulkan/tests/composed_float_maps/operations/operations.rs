//! Independent sign/zero oracle and qualified exact low-format two-step packing.
use super::*;
const N: usize = 4096;

fn mixed<T: Format>(backend: &PcuVulkanBackend) -> usize {
    let mut execute = source::mixed_prepare::<T, N, _>(backend).unwrap();
    let sentinel = T::from(T::ONE + 1);
    let mut output = vec![sentinel; N + 3];
    let mut input = vec![T::from(0); N];
    let mut count = 0;
    let mut raw = 0;
    let mut state = 0x243f_6a88_85a3_08d3u64;
    let low = T::SIGN <= 0x8000;
    let edges = [
        0,
        T::SIGN,
        1,
        T::SIGN | 1,
        T::MIN_NORMAL - 1,
        T::MIN_NORMAL,
        T::MAX,
        T::SIGN | T::MAX,
    ];
    while if low {
        raw < T::SIGN * 2
    } else {
        count < 4 * N
    } {
        let mut filled = 0;
        while filled < N && (!low || raw < T::SIGN * 2) {
            let bits = if low {
                let bits = raw;
                raw += 1;
                bits
            } else {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let random = if T::SIGN == (1 << 31) {
                    state & u64::from(u32::MAX)
                } else {
                    state
                };
                if filled < edges.len() {
                    edges[filled]
                } else {
                    random
                }
            };
            if bits & (T::SIGN - 1) <= T::MAX {
                input[filled] = T::from(bits);
                filled += 1;
            }
        }
        input[filled..].fill(T::from(0));
        execute(&input, &mut output).unwrap();
        for (actual, value) in output[..filled].iter().zip(&input[..filled]) {
            let expected = if value.bits() & T::SIGN != 0 {
                value.bits() & (T::SIGN - 1)
            } else {
                0
            };
            assert_eq!(actual.bits(), expected, "{:?}/{:x}", T::TYPE, value.bits());
        }
        assert!(
            output[N..]
                .iter()
                .all(|value| value.bits() == sentinel.bits())
        );
        count += filled;
    }
    count
}

fn low_clamp<T: Format>(backend: &PcuVulkanBackend) -> usize {
    let sentinel = T::from(T::ONE + 1);
    let mut count = 0;
    for uf in UF {
        let mut source = source_plan::<T, N>(backend, uf, Range::Clamp);
        let mut graph = graph_plan::<T, N>(backend, uf, Range::Clamp);
        let mut input = vec![T::from(0); N];
        let mut output = vec![sentinel; N + 3];
        let mut graph_output = output.clone();
        let mut raw = 0;
        while raw < T::SIGN * 2 {
            let mut filled = 0;
            while filled < N && raw < T::SIGN * 2 {
                let bits = raw;
                raw += 1;
                if bits & (T::SIGN - 1) <= T::MAX {
                    input[filled] = T::from(bits);
                    filled += 1;
                }
            }
            input[filled..].fill(T::from(0));
            let mut notice = None;
            let expected: Vec<T> = input
                .iter()
                .enumerate()
                .map(|(lane, value)| {
                    let (fault, result) = oracle::low_expected(*value, uf, Range::Clamp);
                    if let Some(mut fault) = fault {
                        fault.invocation_id = u64::try_from(lane).unwrap();
                        notice.get_or_insert(fault);
                    }
                    result.unwrap()
                })
                .collect();
            let wanted = notice.map_or(Ok(()), Err);
            assert_eq!(observed(source(&mut output, &input)), wanted);
            assert_eq!(explicit(&mut graph, &input, &mut graph_output), wanted);
            for ((actual, graph), expected) in
                output[..N].iter().zip(&graph_output[..N]).zip(expected)
            {
                assert_eq!(actual.bits(), expected.bits());
                assert_eq!(graph.bits(), expected.bits());
            }
            assert!(
                output[N..]
                    .iter()
                    .all(|value| value.bits() == sentinel.bits())
            );
            assert!(
                graph_output[N..]
                    .iter()
                    .all(|value| value.bits() == sentinel.bits())
            );
            count += filled * 2;
        }
    }
    count
}

#[test]
#[ignore = "requires Vulkan hardware; complete finite low encodings and independent bit selection"]
fn finite_encodings_static_operations_and_independent_clamp_payloads() {
    global::use_defaults().unwrap();
    let backend = PcuVulkanBackend::new().unwrap();
    let mut selections = 0;
    let mut clamps = 0;
    macro_rules! low {
        ($ty:ty) => {
            selections += mixed::<$ty>(&backend);
            clamps += low_clamp::<$ty>(&backend);
        };
    }
    low!(pcu_facade::PcuF16Bits);
    low!(pcu_facade::PcuBf16Bits);
    low!(pcu_facade::PcuF8E4M3FnBits);
    low!(pcu_facade::PcuF8E5M2Bits);
    selections += mixed::<f32>(&backend);
    selections += mixed::<f64>(&backend);
    let input32 = [1.0_f32, -1.0, 2.0, 0.0, -0.0, 0.5];
    let input64 = [1.0_f64, -1.0, 2.0, 0.0, -0.0, 0.5];
    let mut out32 = [17.0_f32; 9];
    let mut out64 = [17.0_f64; 9];
    source::constant_f32_prepare::<6, _>(&backend).unwrap()(&input32, &mut out32).unwrap();
    source::constant_f64_prepare::<6, _>(&backend).unwrap()(&input64, &mut out64).unwrap();
    assert_eq!(
        out32.map(f32::to_bits),
        [0.5, 1.5, 0.0, 1.0, 1.0, 0.75, 17.0, 17.0, 17.0].map(f32::to_bits)
    );
    assert_eq!(
        out64.map(f64::to_bits),
        [0.5, 1.5, 0.0, 1.0, 1.0, 0.75, 17.0, 17.0, 17.0].map(f64::to_bits)
    );
    println!(
        "Static operations: {selections} representation-only mixed outputs +{clamps} source/IR low finite Clamp outputs; all3UF; no general F32/F64 compound oracle or latency claim"
    );
}
