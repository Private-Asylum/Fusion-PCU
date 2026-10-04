//! Direct GLSL source compilation for the fixed Add-then-Mul expression.
use std::fmt::Write as _;
#[rustfmt::skip]
use super::{
    NativeResult,
    PcuScalarType,
};

const fn is_integer(scalar: PcuScalarType) -> bool {
    matches!(
        scalar,
        PcuScalarType::U8
            | PcuScalarType::I8
            | PcuScalarType::U16
            | PcuScalarType::I16
            | PcuScalarType::U32
            | PcuScalarType::I32
            | PcuScalarType::U64
            | PcuScalarType::I64
            | PcuScalarType::U128
            | PcuScalarType::I128
            | PcuScalarType::U256
            | PcuScalarType::I256
            | PcuScalarType::U512
            | PcuScalarType::I512
    )
}

fn source(
    policy: u32,
    range: u32,
    scalar: PcuScalarType,
    extent: u32,
    one_effect: bool,
) -> NativeResult<String> {
    if policy > 2 || range > 1 || extent == 0 {
        return Err("native unsupported policy/range/extent".into());
    }
    if is_integer(scalar) {
        return integer_source(policy, range, scalar, extent, one_effect);
    }
    let (template, format, wide) = match scalar {
        PcuScalarType::F32 => (
            include_str!("../../../../../spirv/composed/shader/f32.comp"),
            0,
            false,
        ),
        PcuScalarType::F64 => (
            include_str!("../../../../../spirv/composed/shader/f64.comp"),
            0,
            true,
        ),
        PcuScalarType::F16 => (
            include_str!("../../../../../spirv/composed/shader/low.comp"),
            0,
            false,
        ),
        PcuScalarType::BF16 => (
            include_str!("../../../../../spirv/composed/shader/low.comp"),
            1,
            false,
        ),
        PcuScalarType::F8E4M3FN => (
            include_str!("../../../../../spirv/composed/shader/low.comp"),
            2,
            false,
        ),
        PcuScalarType::F8E5M2 => (
            include_str!("../../../../../spirv/composed/shader/low.comp"),
            3,
            false,
        ),
        _ => return Err("native unsupported scalar".into()),
    };
    let header = template
        .split_once("void main(){")
        .ok_or("native template main absent")?
        .0;
    // Preserve the disclosed floating-bit arithmetic helpers, but construct the fixed
    // source program directly. No PCU IR/profile or SPIR-V rewriting is used here.
    let mut source =
        format!("#version 450\nconst uint EXTENT={extent};\nconst uint FORMAT={format};\n");
    for line in header
        .lines()
        .skip(1)
        .filter(|line| !line.starts_with("layout(constant_id="))
    {
        source.push_str(line);
        source.push('\n');
    }
    let seed = if wide {
        "uvec2(read_word(1,word*2u),read_word(1,word*2u+1u))"
    } else {
        "uvec2(read_word(1,word),0)"
    };
    let store = if wide {
        "write_word(0,word*2u,private_words[0].x);write_word(0,word*2u+1u,private_words[0].y);"
    } else {
        "write_word(0,word,private_words[0].x);"
    };
    let second = if one_effect {
        format!("native_load(2,1,0,{policy},{range},0,0,4);")
    } else {
        format!("native_mul(2,1,0,{policy},{range},0,0,4);")
    };
    write!(
        source,
        r"
void main() {{
 uint word=gl_GlobalInvocationID.x;
 uint word_extent=EXTENT/LANES+uint(EXTENT%LANES!=0);
 if(word>=word_extent)return;
 private_words[0]=uvec2(0);
 private_words[1]={seed};
 for(packed_lane=0;packed_lane<LANES;packed_lane++) {{
  logical_index=word*LANES+packed_lane;
  if(logical_index>=EXTENT)break;
  notice=0;notice_step=0;fatal=false;
  native_load(0,1,0,{policy},{range},0,0,0);
  native_add(1,0,0,{policy},{range},0,0,2);
  {second}
  native_store(0,0,2,{policy},{range},0,0,5);
  status_data.words[2*logical_index]=notice;
  status_data.words[2*logical_index+1]=notice_step;
 }}
 {store}
}}
"
    )?;
    Ok(source)
}

fn integer_source(
    _policy: u32,
    range: u32,
    scalar: PcuScalarType,
    extent: u32,
    one_effect: bool,
) -> NativeResult<String> {
    if !is_integer(scalar) {
        return Err("integer composed scalar".into());
    }
    let bits = scalar.bit_width();
    let signed = matches!(
        scalar,
        PcuScalarType::I8
            | PcuScalarType::I16
            | PcuScalarType::I32
            | PcuScalarType::I64
            | PcuScalarType::I128
            | PcuScalarType::I256
            | PcuScalarType::I512
    );
    Ok(format!(
        "#version 450\nconst uint BITS={bits}u;\nconst int L={};\nconst bool SIGNED={signed};\nconst uint EXTENT={extent}u;\nconst bool CLAMP={};\nconst bool ONE={one_effect};\n{}",
        bits.div_ceil(32),
        range == 1,
        include_str!("../../../composed_integer_maps/native/arithmetic.comp")
    ))
}

pub fn shader(
    policy: u32,
    range: u32,
    scalar: PcuScalarType,
    extent: u32,
    one_effect: bool,
) -> NativeResult<Vec<u32>> {
    let source = source(policy, range, scalar, extent, one_effect)?;
    let base = std::env::temp_dir().join(format!(
        "pcu-native-composed-{}-{scalar:?}-{policy}-{range}-{extent}-{one_effect}",
        std::process::id()
    ));
    let input = base.with_extension("comp");
    let output = base.with_extension("spv");
    std::fs::write(&input, source)?;
    let result = std::process::Command::new("glslangValidator")
        .args(["-V", "--target-env", "vulkan1.0", "-Os", "-S", "comp"])
        .arg(&input)
        .arg("-o")
        .arg(&output)
        .output()?;
    std::fs::remove_file(input)?;
    if !result.status.success() {
        return Err(format!(
            "native GLSL compiler: {} {}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        )
        .into());
    }
    let bytes = std::fs::read(&output)?;
    std::fs::remove_file(output)?;
    if !bytes.len().is_multiple_of(4) {
        return Err("native malformed SPIR-V word extent".into());
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect())
}
