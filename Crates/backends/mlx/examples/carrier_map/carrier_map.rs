//! Genuine generic source using a retained MLX integer-limb primitive.
#[path = "source/source.rs"]
mod source;
#[rustfmt::skip]
use pcu_facade::{PcuScalar,PcuU512};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = fusion_pcu_mlx::MlxRuntime::load_default()?;
    let session = runtime.open_gpu(0)?;
    let mut copy = source::copy_prepare::<PcuU512, 3, _>(&session)
        .map_err(|error| std::io::Error::other(format!("{error:?}")))?;
    let mut broadcast = source::broadcast_prepare::<PcuU512, 3, _>(&session)
        .map_err(|error| std::io::Error::other(format!("{error:?}")))?;
    let input = [
        PcuU512::decode_le([0xff; 64]),
        PcuU512::decode_le([0; 64]),
        PcuU512::decode_le([0x80; 64]),
    ];
    let sentinel = PcuU512::decode_le([0x37; 64]);
    let mut output = [sentinel; 5];
    copy(&input, &mut output).map_err(|error| std::io::Error::other(format!("{error:?}")))?;
    assert_eq!(&output[..3], &input);
    assert_eq!(&output[3..], &[sentinel; 2]);
    // The native prefix control freezes the full five-element resident shape cold;
    // execution copies only three values and never constructs a warm input view.
    let resident = session.upload_encoded(&[input[0], input[1], input[2], sentinel, sentinel])?;
    let mut prefix =
        session.prepare_carrier_control_with_input_extent(PcuU512::TYPE, 3, false, 5)?;
    let completed = prefix.execute_resident(&resident)?.into_parts().0;
    completed.read_into(&mut output)?;
    assert_eq!(&output[..3], &input);
    assert_eq!(&output[3..], &[sentinel; 2]);
    broadcast(&input[2], &mut output)
        .map_err(|error| std::io::Error::other(format!("{error:?}")))?;
    assert_eq!(&output[..3], &[input[2]; 3]);
    assert_eq!(&output[3..], &[sentinel; 2]);
    println!(
        "MLX retained source copied/broadcast exact U512 limbs; native resident prefix preserved both tails."
    );
    Ok(())
}
