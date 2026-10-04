//! Cold external composed template ownership; filesystem IO stays outside the `no_std` emitter.
#[rustfmt::skip]
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
};
#[rustfmt::skip]
use fusion_pcu_spirv::{
    PcuSpirvComposedTemplate,
    PcuSpirvComposedTemplateData,
    PcuSpirvComposedTemplateFamily as Family,
    PCU_COMPOSED_TEMPLATE_REVISION,
};
use crate::{PcuVulkanError, PcuVulkanShaderArtifactError};
#[cfg(feature = "embedded-composed")]
use std::io::{BufWriter, Write};

/// Runtime source choice is separate from generated-artifact cache retention.
#[derive(Clone, Debug, Default)]
pub enum PcuVulkanShaderSource {
    #[default]
    Embedded,
    /// Exact known composed arithmetic packages; other compiler families retain their source.
    ExternalComposed {
        directory: PathBuf,
        retain_in_memory: bool,
    },
}
impl PcuVulkanShaderSource {
    /// Selects `pcu-shader-packages` beside the executable.
    /// # Errors
    /// Returns executable path discovery failures.
    pub fn beside_executable() -> std::io::Result<Self> {
        let executable = std::env::current_exe()?;
        let parent = executable
            .parent()
            .ok_or_else(|| std::io::Error::other("executable has no parent"))?;
        Ok(Self::ExternalComposed {
            directory: parent.join("pcu-shader-packages"),
            retain_in_memory: true,
        })
    }
}
const fn family_slot(family: Family) -> usize {
    match family {
        Family::F32 => 0,
        Family::F64 => 1,
        Family::Low => 2,
        Family::IntegerWide => 3,
        Family::IntegerNarrow => 4,
    }
}
const fn family_name(family: Family) -> &'static str {
    match family {
        Family::F32 => "composed-f32",
        Family::F64 => "composed-f64",
        Family::Low => "composed-low",
        Family::IntegerWide => "composed-integer-wide",
        Family::IntegerNarrow => "composed-integer-narrow",
    }
}
struct OwnedTemplate {
    family: Family,
    words: Vec<u32>,
    functions: Vec<u32>,
    calls: Vec<usize>,
    arguments: Vec<u32>,
    defaults: Vec<usize>,
}
impl OwnedTemplate {
    fn data(&self) -> PcuSpirvComposedTemplateData<'_> {
        PcuSpirvComposedTemplateData {
            family: self.family,
            words: &self.words,
            function_ids: &self.functions,
            call_target_offsets: &self.calls,
            call_argument_ids: &self.arguments,
            specialization_offsets: &self.defaults,
        }
    }
    fn load(directory: &Path, family: Family) -> Result<Self, PcuVulkanError> {
        let path = directory.join(family_name(family));
        let module = read_bounded(&path.join("template.spv"), 128 * 1024)?;
        let (chunks, remainder) = module.as_chunks::<4>();
        if !remainder.is_empty() {
            return Err(invalid());
        }
        let words = chunks
            .iter()
            .map(|chunk| u32::from_le_bytes(*chunk))
            .collect();
        let metadata = read_bounded(&path.join("rewrite.bin"), 16 * 1024)?;
        let mut cursor = Cursor(&metadata);
        if cursor.take(PCU_COMPOSED_TEMPLATE_REVISION.len())? != PCU_COMPOSED_TEMPLATE_REVISION
            || cursor.u32()? != u32::try_from(family_slot(family)).map_err(|_| invalid())?
        {
            return Err(invalid());
        }
        let template = Self {
            family,
            words,
            functions: cursor.words()?,
            calls: cursor.offsets()?,
            arguments: cursor.words()?,
            defaults: cursor.offsets()?,
        };
        if !cursor.0.is_empty() {
            return Err(invalid());
        }
        PcuSpirvComposedTemplate::validate(template.data()).map_err(|_| invalid())?;
        Ok(template)
    }
}
const fn invalid() -> PcuVulkanError {
    PcuVulkanError::ShaderArtifact(PcuVulkanShaderArtifactError::InvalidTemplate)
}
fn read_bounded(path: &Path, limit: usize) -> Result<Vec<u8>, PcuVulkanError> {
    let mut bytes = Vec::new();
    fs::File::open(path)
        .and_then(|file| file.take((limit + 1) as u64).read_to_end(&mut bytes))
        .map_err(|error| PcuVulkanError::ShaderArtifact(error.into()))?;
    if bytes.len() > limit {
        return Err(invalid());
    }
    Ok(bytes)
}
struct Cursor<'a>(&'a [u8]);
impl<'a> Cursor<'a> {
    fn take(&mut self, length: usize) -> Result<&'a [u8], PcuVulkanError> {
        let (first, rest) = self.0.split_at_checked(length).ok_or_else(invalid)?;
        self.0 = rest;
        Ok(first)
    }
    fn u32(&mut self) -> Result<u32, PcuVulkanError> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().map_err(|_| invalid())?,
        ))
    }
    fn u64(&mut self) -> Result<u64, PcuVulkanError> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().map_err(|_| invalid())?,
        ))
    }
    fn words(&mut self) -> Result<Vec<u32>, PcuVulkanError> {
        let length = usize::try_from(self.u64()?).map_err(|_| invalid())?;
        if length > self.0.len() / 4 {
            return Err(invalid());
        }
        (0..length).map(|_| self.u32()).collect()
    }
    fn offsets(&mut self) -> Result<Vec<usize>, PcuVulkanError> {
        let length = usize::try_from(self.u64()?).map_err(|_| invalid())?;
        if length > self.0.len() / 8 {
            return Err(invalid());
        }
        (0..length)
            .map(|_| {
                self.u64()
                    .and_then(|value| usize::try_from(value).map_err(|_| invalid()))
            })
            .collect()
    }
}

#[derive(Default)]
pub struct SourceStore {
    pub(crate) source: PcuVulkanShaderSource,
    templates: [Option<OwnedTemplate>; 5],
}
impl SourceStore {
    pub(crate) fn configure(&mut self, source: PcuVulkanShaderSource) {
        self.source = source;
        self.clear();
    }
    pub(crate) fn clear(&mut self) {
        self.templates = [None, None, None, None, None];
    }
    pub(crate) fn lower(
        &mut self,
        kernel: &fusion_pcu_core::PcuDispatchKernelIr<'_>,
        family: Family,
        one_effect: bool,
        sink: &mut Vec<u32>,
    ) -> Result<fusion_pcu_spirv::PcuSpirvComposedProfile, PcuVulkanError> {
        let options = fusion_pcu_spirv::PcuSpirvLoweringOptions::minimal_shader();
        match &self.source {
            PcuVulkanShaderSource::Embedded => {
                #[cfg(feature = "embedded-composed")]
                {
                    let lower = if matches!(family, Family::IntegerWide | Family::IntegerNarrow) {
                        if one_effect {
                            fusion_pcu_spirv::lower_one_effect_integer_to_spirv
                        } else {
                            fusion_pcu_spirv::lower_composed_integer_to_spirv
                        }
                    } else if one_effect {
                        fusion_pcu_spirv::lower_one_effect_float_to_spirv
                    } else {
                        fusion_pcu_spirv::lower_composed_float_to_spirv
                    };
                    lower(kernel, options, sink)
                        .map(|(_, profile)| profile)
                        .map_err(|error| PcuVulkanError::SpirvLowering { error })
                }
                #[cfg(not(feature = "embedded-composed"))]
                {
                    let _ = (kernel, options, one_effect, sink);
                    Err(PcuVulkanError::ShaderArtifact(
                        PcuVulkanShaderArtifactError::MissingEmbeddedTemplates,
                    ))
                }
            }
            PcuVulkanShaderSource::ExternalComposed {
                directory,
                retain_in_memory,
            } => {
                let slot = family_slot(family);
                if *retain_in_memory {
                    if self.templates[slot].is_none() {
                        self.templates[slot] = Some(OwnedTemplate::load(directory, family)?);
                    }
                    let template = self.templates[slot].as_ref().ok_or_else(invalid)?;
                    // Revalidation remains cold, never a prepared-call operation.
                    lower_external(kernel, template, one_effect, sink)
                } else {
                    lower_external(
                        kernel,
                        &OwnedTemplate::load(directory, family)?,
                        one_effect,
                        sink,
                    )
                }
            }
        }
    }
}
fn lower_external(
    kernel: &fusion_pcu_core::PcuDispatchKernelIr<'_>,
    template: &OwnedTemplate,
    one_effect: bool,
    sink: &mut Vec<u32>,
) -> Result<fusion_pcu_spirv::PcuSpirvComposedProfile, PcuVulkanError> {
    let validated = PcuSpirvComposedTemplate::validate(template.data()).map_err(|_| invalid())?;
    if matches!(template.family, Family::IntegerWide | Family::IntegerNarrow) {
        return fusion_pcu_spirv::lower_composed_integer_with_template(
            kernel,
            fusion_pcu_spirv::PcuSpirvLoweringOptions::minimal_shader(),
            validated,
            one_effect,
            sink,
        )
        .map(|(_, profile)| profile)
        .map_err(|error| PcuVulkanError::SpirvLowering { error });
    }
    fusion_pcu_spirv::lower_composed_float_with_template(
        kernel,
        fusion_pcu_spirv::PcuSpirvLoweringOptions::minimal_shader(),
        validated,
        one_effect,
        sink,
    )
    .map(|(_, profile)| profile)
    .map_err(|error| PcuVulkanError::SpirvLowering { error })
}

/// Exports this compiler's exact built-in composed families to a deployment package directory.
/// # Errors
/// Returns explicit IO or compiler identity failures. Existing package files are never replaced.
#[cfg(feature = "embedded-composed")]
pub fn write_composed_shader_package(directory: &Path) -> Result<(), PcuVulkanError> {
    fs::create_dir_all(directory).map_err(|error| PcuVulkanError::ShaderArtifact(error.into()))?;
    for family in [
        Family::F32,
        Family::F64,
        Family::Low,
        Family::IntegerWide,
        Family::IntegerNarrow,
    ] {
        let data = fusion_pcu_spirv::embedded_composed_template(family);
        PcuSpirvComposedTemplate::validate(data).map_err(|_| invalid())?;
        let destination = directory.join(family_name(family));
        if destination.exists() {
            OwnedTemplate::load(directory, family)?;
            continue;
        }
        let temporary = directory.join(format!(".{}-{}", family_name(family), std::process::id()));
        fs::create_dir(&temporary).map_err(|error| PcuVulkanError::ShaderArtifact(error.into()))?;
        let result = write_template_files(&temporary, data).and_then(|()| {
            fs::rename(&temporary, &destination)
                .map_err(|error| PcuVulkanError::ShaderArtifact(error.into()))
        });
        if temporary.exists() {
            let _ = fs::remove_dir_all(&temporary);
        }
        result?;
    }
    Ok(())
}
#[cfg(feature = "embedded-composed")]
fn write_template_files(
    directory: &Path,
    data: PcuSpirvComposedTemplateData<'_>,
) -> Result<(), PcuVulkanError> {
    let result = (|| -> std::io::Result<()> {
        let mut shader = BufWriter::new(fs::File::create(directory.join("template.spv"))?);
        for word in data.words {
            shader.write_all(&word.to_le_bytes())?;
        }
        shader.flush()?;
        shader.get_ref().sync_all()?;
        let mut metadata = BufWriter::new(fs::File::create(directory.join("rewrite.bin"))?);
        metadata.write_all(PCU_COMPOSED_TEMPLATE_REVISION)?;
        metadata.write_all(
            &u32::try_from(family_slot(data.family))
                .map_err(std::io::Error::other)?
                .to_le_bytes(),
        )?;
        write_words(&mut metadata, data.function_ids)?;
        write_offsets(&mut metadata, data.call_target_offsets)?;
        write_words(&mut metadata, data.call_argument_ids)?;
        write_offsets(&mut metadata, data.specialization_offsets)?;
        metadata.flush()?;
        metadata.get_ref().sync_all()?;
        Ok(())
    })();
    result.map_err(|error| PcuVulkanError::ShaderArtifact(error.into()))
}
#[cfg(feature = "embedded-composed")]
fn write_words(writer: &mut impl Write, words: &[u32]) -> std::io::Result<()> {
    writer.write_all(&(words.len() as u64).to_le_bytes())?;
    for word in words {
        writer.write_all(&word.to_le_bytes())?;
    }
    Ok(())
}
#[cfg(feature = "embedded-composed")]
fn write_offsets(writer: &mut impl Write, offsets: &[usize]) -> std::io::Result<()> {
    writer.write_all(&(offsets.len() as u64).to_le_bytes())?;
    for offset in offsets {
        writer.write_all(&(*offset as u64).to_le_bytes())?;
    }
    Ok(())
}

#[cfg(all(test, feature = "embedded-composed"))]
mod tests {
    use super::*;
    #[test]
    fn exact_package_load_and_mutation_refusal() {
        let directory =
            std::env::temp_dir().join(format!("pcu-package-test-{}", std::process::id()));
        assert!(!directory.exists());
        write_composed_shader_package(&directory).unwrap();
        // Re-exporting a matching package is idempotent and never replaces deployed bytes.
        write_composed_shader_package(&directory).unwrap();
        for family in [
            Family::F32,
            Family::F64,
            Family::Low,
            Family::IntegerWide,
            Family::IntegerNarrow,
        ] {
            let loaded = OwnedTemplate::load(&directory, family).unwrap();
            let original = fusion_pcu_spirv::embedded_composed_template(family);
            assert_eq!(loaded.words, original.words);
            assert_eq!(loaded.functions, original.function_ids);
            assert_eq!(loaded.calls, original.call_target_offsets);
            assert_eq!(loaded.arguments, original.call_argument_ids);
            assert_eq!(loaded.defaults, original.specialization_offsets);
        }
        let path = directory
            .join(family_name(Family::F32))
            .join("template.spv");
        let mut words = fs::read(&path).unwrap();
        words[8] ^= 1;
        fs::write(&path, words).unwrap();
        assert!(OwnedTemplate::load(&directory, Family::F32).is_err());
        assert!(write_composed_shader_package(&directory).is_err());
        fs::remove_dir_all(directory).unwrap();
    }
}
