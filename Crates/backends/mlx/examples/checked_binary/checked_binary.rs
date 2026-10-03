//! Checked integer-synthesized arithmetic owned and scheduled by MLX; no native half or double ALU claim.
#[path = "source/source.rs"]
mod source;
use fusion_pcu_mlx::MlxRuntime;
use pcu_facade::PcuF16Bits;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = MlxRuntime::load_default()?;
    let session = runtime.open_gpu(0)?;
    let backend = session.checked_binary_backend();
    let left = [
        PcuF16Bits::from_bits(0x3c00),
        PcuF16Bits::from_bits(0x4000),
        PcuF16Bits::from_bits(0x4200),
    ];
    let right = [PcuF16Bits::from_bits(0x4400); 3];
    let mut output = [PcuF16Bits::from_bits(0); 5];
    let mut add = source::add_prepare::<PcuF16Bits, _>(&backend)
        .map_err(|error| std::io::Error::other(format!("{error:?}")))?;
    add(&left, &right, &mut output).map_err(|error| std::io::Error::other(format!("{error:?}")))?;
    assert_eq!(
        output.map(PcuF16Bits::to_bits),
        [0x4500, 0x4600, 0x4700, 0, 0]
    );
    println!(
        "MLX checked F16 source output bits: {:?}",
        output.map(PcuF16Bits::to_bits)
    );
    let mut double = source::add_prepare::<f64, _>(&backend)
        .map_err(|error| std::io::Error::other(format!("{error:?}")))?;
    let mut result = [91.0_f64; 5];
    double(&[1.0, 2.0, 3.0], &[4.0; 3], &mut result)
        .map_err(|error| std::io::Error::other(format!("{error:?}")))?;
    assert_eq!(
        result.map(f64::to_bits),
        [5.0_f64, 6.0, 7.0, 91.0, 91.0].map(f64::to_bits)
    );
    println!("MLX checked F64 source output: {result:?}");
    Ok(())
}
