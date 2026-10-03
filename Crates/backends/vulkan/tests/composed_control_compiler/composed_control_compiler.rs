//! External compiler/validator evidence only; this test never opens a device.
use pcu_facade::PcuScalarType;
type NativeResult<T> = Result<T, Box<dyn std::error::Error>>;
#[path = "../../benches/composed_float_maps/ffi/compile/compile.rs"]
mod compile;

fn validate(words: &[u32], index: usize) {
    assert_eq!(words[0], 0x0723_0203);
    let mut offset = 5;
    while offset < words.len() {
        let count = usize::try_from(words[offset] >> 16).unwrap();
        assert!(count > 0 && offset + count <= words.len());
        if words[offset] & 0xffff == 17 {
            // Only Shader capability; no native Float64/Int64 arithmetic dependency.
            assert_eq!(&words[offset + 1..offset + count], &[1]);
        }
        offset += count;
    }
    let path = std::env::temp_dir().join(format!(
        "pcu-native-composed-validation-{}-{index}.spv",
        std::process::id()
    ));
    let bytes: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    std::fs::write(&path, bytes).unwrap();
    let result = std::process::Command::new("spirv-val")
        .args(["--target-env", "vulkan1.0"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{} {}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    std::fs::remove_file(path).unwrap();
}

#[test]
#[ignore = "requires installed GLSL compiler and official SPIR-V validator; no GPU"]
fn independent_native_glsl_compiler_validates_all_formats_and_policies() {
    all_modules(false);
}

#[test]
#[ignore = "requires installed GLSL compiler and official validator; independent one-effect control"]
fn independent_one_effect_glsl_compiler_validates_all_formats_and_policies() {
    all_modules(true);
}

fn all_modules(one_effect: bool) {
    let mut index = 0;
    for scalar in [
        PcuScalarType::F16,
        PcuScalarType::BF16,
        PcuScalarType::F8E4M3FN,
        PcuScalarType::F8E5M2,
        PcuScalarType::F32,
        PcuScalarType::F64,
    ] {
        for policy in 0..3 {
            for range in 0..2 {
                for extent in [1, 65] {
                    let words = compile::shader(policy, range, scalar, extent, one_effect).unwrap();
                    validate(&words, index);
                    index += 1;
                }
            }
        }
    }
    assert_eq!(index, 72);
}
