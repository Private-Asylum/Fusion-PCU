use super::*;
use source::{four_inputs, grid, ordered, retain};

pub(super) fn verify<T: PcuScalar>() {
    let sentinel = sample::<T>(0xA5);
    let mut ghost = [];
    for phase in [0x00_u8, 0x55, 0xFF] {
        let input: [T; N + 3] =
            core::array::from_fn(|index| sample(phase.wrapping_add(u8::try_from(index).unwrap())));
        let old_stage: [T; N + 2] = core::array::from_fn(|index| {
            sample(
                phase
                    .wrapping_add(0x40)
                    .wrapping_add(u8::try_from(index).unwrap()),
            )
        });
        let seed = sample::<T>(phase.wrapping_add(0x80));
        let mut stage = old_stage;
        let mut output = [sentinel; N + 4];
        ordered::<T, N>(&input, &seed, &mut ghost, &mut stage, &mut output).unwrap();
        bits(&stage[..N], &[seed; N]);
        bits(&stage[N..], &old_stage[N..]);
        bits(&output[..N], &old_stage[..N]);
        bits(&output[N..], &[sentinel; 4]);
        let resident_input = retain(&input).unwrap();
        for grid_route in [false, true] {
            let mut stage = old_stage;
            let mut output = [sentinel; N + 4];
            if grid_route {
                grid::<T, N>(&resident_input, &seed, &mut ghost, &mut stage, &mut output).unwrap();
            } else {
                ordered::<T, N>(&resident_input, &seed, &mut ghost, &mut stage, &mut output)
                    .unwrap();
            }
            bits(&stage[..N], &[seed; N]);
            bits(&stage[N..], &old_stage[N..]);
            bits(&output[..N], &old_stage[..N]);
            bits(&output[N..], &[sentinel; 4]);
        }
        resident_outputs(&input, &old_stage, seed, &resident_input);
        read(&resident_input, &input);
    }
}

fn resident_outputs<T: PcuScalar>(
    input: &[T; N + 3],
    old_stage: &[T; N + 2],
    seed: T,
    resident_input: &PcuTensor<T>,
) {
    let sentinel = sample::<T>(0xA5);
    let mut ghost = [];
    for grid_route in [false, true] {
        let mut stage = retain(old_stage).unwrap();
        let mut output = retain(&[sentinel; N + 4]).unwrap();
        let sibling = retain(old_stage).unwrap();
        if grid_route {
            grid::<T, N>(resident_input, &seed, &mut ghost, &mut stage, &mut output).unwrap();
        } else {
            ordered::<T, N>(input, &seed, &mut ghost, &mut stage, &mut output).unwrap();
        }
        let mut expected_stage = *old_stage;
        expected_stage[..N].fill(seed);
        let mut expected_output = [sentinel; N + 4];
        expected_output[..N].copy_from_slice(&old_stage[..N]);
        read(&stage, &expected_stage);
        read(&output, &expected_output);
        read(&sibling, old_stage);
        let mut short = retain(&[sentinel; N - 1]).unwrap();
        assert!(
            ordered::<T, N>(resident_input, &seed, &mut ghost, &mut stage, &mut short).is_err()
        );
        read(&stage, &expected_stage);
        read(&short, &[sentinel; N - 1]);
        four_inputs::<T, N>(resident_input, &seed, &mut stage, &mut output).unwrap();
        let old_output = expected_output;
        expected_output[..N].copy_from_slice(&expected_stage[..N]);
        expected_stage[..N].copy_from_slice(&old_output[..N]);
        read(&stage, &expected_stage);
        read(&output, &expected_output);
        // Escaped owners retain their actual native roots after cache eviction.
        global::clear_thread_cache().unwrap();
        read(&stage, &expected_stage);
        read(&output, &expected_output);
        let mut host = [sentinel; N + 4];
        ordered::<T, N>(resident_input, &seed, &mut ghost, &mut stage, &mut host).unwrap();
        bits(&host[..N], &expected_stage[..N]);
        bits(&host[N..], &[sentinel; 4]);
    }
}
