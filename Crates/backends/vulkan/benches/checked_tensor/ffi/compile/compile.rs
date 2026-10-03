//! Independent native shader compilation; audited arithmetic bodies, no provider bytecode.
#[rustfmt::skip]
use pcu_facade::PcuScalarType;
use super::Result;

pub(super) fn shader(
    scalar: PcuScalarType,
    extent: u32,
    operation: u32,
    policy: u32,
) -> Result<Vec<u32>> {
    if extent == 0 || operation > 5 || policy > 2 {
        return Err("native pointwise profile unsupported".into());
    }
    let format = match scalar {
        PcuScalarType::F16 => Some(0),
        PcuScalarType::BF16 => Some(1),
        PcuScalarType::F8E4M3FN => Some(2),
        PcuScalarType::F8E5M2 => Some(3),
        PcuScalarType::F32 => Some(4),
        PcuScalarType::F64 => Some(5),
        _ => None,
    };
    let mut source = if let Some(format) = format {
        if operation == 5 {
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../spirv/checked_backward/shader/checked_backward.comp"
            ))
            .replace(
                "const uint FORMAT=0u",
                &format!("const uint FORMAT={format}u"),
            )
            .replace(
                "const uint POLICY=0u",
                &format!("const uint POLICY={policy}u"),
            )
            .replace(
                "const uint EXTENT=1u",
                &format!("const uint EXTENT={extent}u"),
            )
        } else if operation == 4 {
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../spirv/checked_unary/shader/checked_unary.comp"
            ))
            .replace(
                "const uint FORMAT=0u",
                &format!("const uint FORMAT={format}u"),
            )
            .replace("const uint OP=0u", "const uint OP=1u")
            .replace(
                "const uint POLICY=0u",
                &format!("const uint POLICY={policy}u"),
            )
            .replace(
                "const uint EXTENT=1u",
                &format!("const uint EXTENT={extent}u"),
            )
        } else {
            let body = match format {
                4 => include_str!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../spirv/checked_binary/shader/checked_binary.comp"
                )),
                5 => include_str!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../spirv/checked_binary/shader/checked_binary_f64.comp"
                )),
                _ => include_str!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../spirv/checked_binary/shader/checked_binary_low.comp"
                )),
            };
            body.replace("const uint OP = 0", &format!("const uint OP = {operation}"))
                .replace(
                    "const uint POLICY = 0",
                    &format!("const uint POLICY = {policy}"),
                )
                .replace(
                    "const uint FORMAT = 0",
                    &format!("const uint FORMAT = {format}"),
                )
                .replace(
                    "const uint EXTENT = 1",
                    &format!("const uint EXTENT = {extent}"),
                )
        }
    } else {
        integer(scalar, extent, operation)?
    };
    // Retain an explicit trailing newline in the standalone GLSL compilation unit.
    source.push('\n');
    let stem = std::env::temp_dir().join(format!(
        "pcu-native-owned-pointwise-{}-{scalar:?}-{extent}-{operation}-{policy}",
        std::process::id()
    ));
    compile(&source, &stem)
}

fn compile(source: &str, stem: &std::path::Path) -> Result<Vec<u32>> {
    let input = stem.with_extension("comp");
    let output = stem.with_extension("spv");
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
            "native compiler {} {}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        )
        .into());
    }
    let bytes = std::fs::read(&output)?;
    std::fs::remove_file(output)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect())
}

pub(super) fn compound(scalar: PcuScalarType, constants: [u32; 9]) -> Result<Vec<u32>> {
    let body = match scalar {
        PcuScalarType::F32 => include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../spirv/checked_compound/shader/checked_compound_f32.comp"
        )),
        PcuScalarType::F64 => include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../spirv/checked_compound/shader/checked_compound_f64.comp"
        )),
        _ => return Err("native compound scalar unsupported".into()),
    };
    let mut source = body.to_owned();
    for ((name, default), value) in [
        ("OP", 0),
        ("POLICY", 0),
        ("ROWS", 1),
        ("INNER", 1),
        ("COLUMNS", 1),
        ("TRANSPOSE_LEFT", 0),
        ("TRANSPOSE_RIGHT", 0),
        ("FACTOR_LOW", 0),
        ("FACTOR_HIGH", 0),
    ]
    .into_iter()
    .zip(constants)
    {
        source = source.replace(
            &format!("const uint {name} = {default};"),
            &format!("const uint {name} = {value};"),
        );
    }
    let stem = std::env::temp_dir().join(format!(
        "pcu-native-compound-{}-{scalar:?}",
        std::process::id()
    ));
    compile(&source, &stem)
}

fn integer(scalar: PcuScalarType, extent: u32, operation: u32) -> Result<String> {
    let signed = match scalar {
        PcuScalarType::I8
        | PcuScalarType::I16
        | PcuScalarType::I32
        | PcuScalarType::I64
        | PcuScalarType::I128
        | PcuScalarType::I256
        | PcuScalarType::I512 => true,
        PcuScalarType::U8
        | PcuScalarType::U16
        | PcuScalarType::U32
        | PcuScalarType::U64
        | PcuScalarType::U128
        | PcuScalarType::U256
        | PcuScalarType::U512 => false,
        _ => return Err("native integer carrier unsupported".into()),
    };
    if operation > 2 {
        return Err("native integer operation unsupported".into());
    }
    Ok(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../spirv/checked_integer/shader/checked_integer.comp"
    ))
    .replace("const uint OP=0u", &format!("const uint OP={operation}u"))
    .replace(
        "const uint SIGNED=0u",
        &format!("const uint SIGNED={}u", u32::from(signed)),
    )
    .replace(
        "const uint ELEMENT_BYTES=4u",
        &format!("const uint ELEMENT_BYTES={}u", scalar.bit_width() / 8),
    )
    .replace(
        "const uint EXTENT=1u",
        &format!("const uint EXTENT={extent}u"),
    ))
}

/// Full loss is retained as an observable checked effect, not removed from the workload.
pub(super) fn training(scalar: PcuScalarType, policy: u32) -> Result<Vec<u32>> {
    if policy > 2 {
        return Err("native training policy unsupported".into());
    }
    let body = match scalar {
        PcuScalarType::F32 => include_str!("../../../gradient/native/training_f32.comp"),
        PcuScalarType::F64 => include_str!("../../../gradient/native/training_f64.comp"),
        _ => return Err("native training scalar unsupported".into()),
    };
    let source = body.replace(
        "const uint POLICY = 0;",
        &format!("const uint POLICY = {policy};"),
    );
    let stem = std::env::temp_dir().join(format!(
        "pcu-native-training-{}-{scalar:?}-{policy}",
        std::process::id()
    ));
    compile(&source, &stem)
}
