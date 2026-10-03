//! Ordinary saved raw values survive an ordered mutable-bank overwrite.

#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuScalar,
    PcuU512,
};

#[pcu(invocations=N,crate_path=::pcu_facade)]
fn saved<T: PcuScalar, const N: usize>(input: &[T], seed: &T, stage: &mut [T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    stage[id] = input[id];
    let retained = stage[id];
    stage[id] = *seed;
    output[id] = retained;
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    pcu_facade::global::configure(pcu_facade::global::PcuExecutionPolicy {
        backend: pcu_facade::global::PcuBackendChoice::Mlx,
        ..pcu_facade::global::PcuExecutionPolicy::default()
    })?;
    let input = [f32::from_bits(0x7fc1_2345), f32::from_bits(0x8000_0000)];
    let seed = f32::from_bits(0x7f80_0055);
    let mut stage = [0.0_f32; 3];
    let mut output = [1.0_f32; 4];
    saved::<f32, 2>(&input, &seed, &mut stage, &mut output)?;
    assert_eq!(
        output.map(f32::to_bits),
        [0x7fc1_2345, 0x8000_0000, 0x3f80_0000, 0x3f80_0000]
    );
    assert_eq!(stage.map(f32::to_bits), [0x7f80_0055, 0x7f80_0055, 0]);
    let mut high = [0_u8; 64];
    high[0] = 7;
    high[63] = 0xf3;
    let input = [PcuU512::decode_le(high)];
    let seed = PcuU512::decode_le([0xa5; 64]);
    let mut stage = [PcuU512::decode_le([0x5e; 64]); 2];
    let mut output = [PcuU512::decode_le([0x79; 64]); 3];
    saved::<PcuU512, 1>(&input, &seed, &mut stage, &mut output)?;
    assert_eq!(output[0].encode_le(), high);
    assert_eq!(stage[0].encode_le(), [0xa5; 64]);
    assert_eq!(output[1].encode_le(), [0x79; 64]);
    assert_eq!(stage[1].encode_le(), [0x5e; 64]);
    pcu_facade::global::clear_thread_cache()?;
    println!(
        "MLX saved transport preserved NaN payloads, signed zero, high limbs and untouched tails."
    );
    Ok(())
}
