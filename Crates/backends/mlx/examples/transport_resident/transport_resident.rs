//! Ordinary retained raw inputs and two private prefix publications survive cache clearing.
#[rustfmt::skip]
use pcu_facade::{
    global,
    pcu,
    PcuBindingRef,
    PcuExecutionError,
    PcuHostArgument,
    PcuScalar,
    PcuTensor,
    PcuU512,
};

#[pcu(crate_path=::pcu_facade)]
fn retain<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}

#[pcu(invocations=2,crate_path=::pcu_facade)]
fn saved<T: PcuScalar>(input: &[T], seed: &[T], stage: &mut [T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    stage[id] = input[id];
    let retained = stage[id];
    stage[id] = seed[0];
    output[id] = retained;
}

fn bytes<T: PcuScalar>(input: &[T]) -> Vec<u8> {
    PcuHostArgument::read(PcuBindingRef::new(0, 0), input)
        .bytes()
        .to_vec()
}

fn run<T: PcuScalar + Copy>(input: &[T; 4], seed: &[T; 3], sentinel: T) {
    let input_owner = retain(input).unwrap();
    let seed_owner = retain(seed).unwrap();
    let mut stage = retain(&[sentinel; 5]).unwrap();
    let mut output = retain(&[sentinel; 7]).unwrap();
    saved(&input_owner, &seed_owner, &mut stage, &mut output).unwrap();
    drop(input_owner);
    drop(seed_owner);
    global::clear_thread_cache().unwrap();
    let mut actual_stage = [sentinel; 8];
    let mut actual_output = [sentinel; 9];
    stage.read_into(&mut actual_stage).unwrap();
    output.read_into(&mut actual_output).unwrap();
    assert_eq!(bytes(&actual_stage[..2]), bytes(&[seed[0]; 2]));
    assert_eq!(bytes(&actual_output[..2]), bytes(&input[..2]));
    assert_eq!(bytes(&actual_stage[2..]), bytes(&[sentinel; 6]));
    assert_eq!(bytes(&actual_output[2..]), bytes(&[sentinel; 7]));
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Mlx,
        ..Default::default()
    })?;
    run(
        &[
            f32::from_bits(0x7fc1_2345),
            f32::from_bits(0x8000_0000),
            f32::INFINITY,
            f32::from_bits(0x7f80_0055),
        ],
        &[f32::from_bits(0x7f80_0077), 0.0, f32::NEG_INFINITY],
        1.0,
    );
    let mut high = [0_u8; 64];
    high[0] = 7;
    high[63] = 0xf3;
    run(
        &[
            PcuU512::decode_le(high),
            PcuU512::decode_le([0xff; 64]),
            PcuU512::decode_le([0x91; 64]),
            PcuU512::decode_le([0x82; 64]),
        ],
        &[PcuU512::decode_le([0xa5; 64]); 3],
        PcuU512::decode_le([0x79; 64]),
    );
    println!(
        "MLX resident saved transport preserved raw payloads, two output tails and owners after input drop/cache clearing."
    );
    Ok(())
}
