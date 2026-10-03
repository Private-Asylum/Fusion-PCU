//! Independent direct native bit-copy program; no provider template or IR specialization.
use super::NativeResult;
pub fn shader(extent: u32, element_bytes: usize) -> NativeResult<Vec<u32>> {
    let source = format!(
        r"#version 450
layout(local_size_x=64) in;
layout(set=0,binding=0,std430) readonly buffer Input {{uint words[];}} input_data;
layout(set=0,binding=1,std430) readonly buffer Seed {{uint words[];}} seed_data;
layout(set=0,binding=2,std430) buffer Stage {{uint words[];}} stage_data;
layout(set=0,binding=3,std430) writeonly buffer Output {{uint words[];}} output_data;
layout(set=0,binding=4,std430) writeonly buffer Status {{uint words[];}} status_data;
const uint SIZE={element_bytes}u;
const uint N={extent}u;
void main(){{
 uint word=gl_GlobalInvocationID.x;
 uint bytes=SIZE*N;
 uint count=bytes/4u+uint(bytes%4u!=0u);
 if(word>=count)return;
 // The source owns this saved SSA representation before overwriting the stage from the seed.
 uint saved=input_data.words[word];
 uint replacement=0u;
 if(SIZE>=4u)replacement=seed_data.words[word%(SIZE/4u)];
 else{{
  uint lanes=4u/SIZE;
  uint mask=(1u<<(SIZE*8u))-1u;
  for(uint lane=0u;lane<lanes;lane++){{
   if(word*lanes+lane>=N)break;
   replacement|=(seed_data.words[0]&mask)<<(lane*SIZE*8u);
  }}
 }}
 stage_data.words[word]=replacement;
 output_data.words[word]=saved;
 status_data.words[word]=0u;
}}"
    );
    let base = std::env::temp_dir().join(format!(
        "pcu-independent-ordered-raw-{}-{element_bytes}-{extent}",
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
            "native raw compiler: {} {}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        )
        .into());
    }
    let bytes = std::fs::read(&output)?;
    std::fs::remove_file(output)?;
    if !bytes.len().is_multiple_of(4) {
        return Err("native raw malformed SPIR-V extent".into());
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect())
}
