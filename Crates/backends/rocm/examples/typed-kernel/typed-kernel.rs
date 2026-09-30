//! Reuse a typed PCU kernel through ordinary Rust slices on an explicitly selected `ROCm` device.

extern crate pcu_facade as fusion_pcu;

use std::process::ExitCode;

use fusion_pcu_macros::pcu;
use fusion_pcu_rocm::RocmDiscovery;

#[path = "../support/selection/selection.rs"]
mod selection;

const COUNT: usize = 2048;
#[pcu(invocations = 250)]
fn transform<const N: usize>(input: &[f32], output: &mut [f32]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = input[id] * 2.0 + 1.0;
        id += stride;
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("ERROR fusion-rocm-typed-kernel: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let discovery = RocmDiscovery::new();
    let candidates = selection::ranked_devices(&discovery, selection::preferred_device()?, false)?;
    let (session, selected) = selection::open_ranked(&discovery, candidates, 256)?;
    let mut transform = transform_prepare::<COUNT, _>(&session)?;

    println!("Prepared typed transform on {}", selected.name);
    for call in 0..3 {
        // Each invocation sees a fresh input and mutable output through ordinary slice borrows.
        let phase = f32::from(u8::try_from(call).expect("small call index fits")) * 0.25;
        let input: Vec<f32> = (0..COUNT)
            .map(|index| f32::from(u16::try_from(index).expect("example index fits")) + phase)
            .collect();
        // The suffix has no corresponding invocation and must be preserved by the backend.
        let mut output = vec![-7.0_f32; COUNT + 5];
        transform(&input, &mut output)?;

        for index in 0..COUNT {
            let scaled = input[index] * 2.0;
            let expected = scaled + 1.0;
            if output[index].to_bits() != expected.to_bits() {
                return Err(format!(
                    "call {call}, element {index}: expected {expected}, got {}",
                    output[index]
                )
                .into());
            }
        }
        if output[COUNT..]
            .iter()
            .any(|value| value.to_bits() != (-7.0_f32).to_bits())
        {
            return Err(format!("call {call}: output tail was not preserved").into());
        }
    }

    println!("Three fresh typed calls passed; output tail remained unchanged.");
    Ok(())
}
