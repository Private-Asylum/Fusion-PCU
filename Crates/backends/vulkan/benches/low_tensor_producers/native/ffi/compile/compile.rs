//! Cold compilation of the independently handwritten packed producer expression.
use super::{NativeResult, PcuScalarType};
pub fn shader(
    policy: u32,
    range: u32,
    scalar: PcuScalarType,
    extent: u32,
    one_effect: bool,
) -> NativeResult<Vec<u32>> {
    if policy > 2 || range != 0 || extent == 0 || one_effect {
        return Err("independent low control profile".into());
    }
    let (sign, maximum, fraction, bias) = match scalar {
        PcuScalarType::F16 => (0x8000, 0x7bff, 10, 15),
        PcuScalarType::BF16 => (0x8000, 0x7f7f, 7, 127),
        PcuScalarType::F8E4M3FN => (0x80, 0x7e, 3, 7),
        PcuScalarType::F8E5M2 => (0x80, 0x7b, 2, 15),
        _ => return Err("independent low control type".into()),
    };
    let source = format!(
        "#version 450\nconst uint EXTENT={extent}u;const uint POLICY={policy}u;const uint BITS={}u;const uint SIGN={sign}u;const uint MAXIMUM={maximum}u;const uint FRACTION={fraction}u;const int BIAS={bias};\n{}",
        scalar.bit_width(),
        include_str!("../../arithmetic.comp")
    );
    let base = std::env::temp_dir().join(format!(
        "pcu-independent-low-producer-{}-{scalar:?}-{policy}-{extent}",
        std::process::id()
    ));
    let input = base.with_extension("comp");
    let output = base.with_extension("spv");
    std::fs::write(&input, source)?;
    let compiler = std::process::Command::new("glslangValidator")
        .args(["-V", "--target-env", "vulkan1.0", "-Os", "-S", "comp"])
        .arg(&input)
        .arg("-o")
        .arg(&output)
        .output()?;
    std::fs::remove_file(input)?;
    if !compiler.status.success() {
        return Err(format!(
            "independent GLSL: {} {}",
            String::from_utf8_lossy(&compiler.stdout),
            String::from_utf8_lossy(&compiler.stderr)
        )
        .into());
    }
    let bytes = std::fs::read(&output)?;
    std::fs::remove_file(output)?;
    if !bytes.len().is_multiple_of(4) {
        return Err("independent SPIR-V words".into());
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect())
}
