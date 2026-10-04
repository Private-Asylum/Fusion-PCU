//! Cold, bounded persistence of admitted PCU modules. This is not an arbitrary shader importer.
#[rustfmt::skip]
use std::{
    borrow::Cow,
    cell::RefCell,
    collections::HashMap,
    fmt,
    fs,
    hash::{Hash, Hasher},
    io::{self, BufWriter, Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};
use fusion_pcu::PcuDispatchKernelIr;
use crate::PcuVulkanCaps;

const REVISION: &[u8] = b"PCU-SPV-ARTIFACT-1-IR-HASH-LE-1-LOWERING-1";
const MAX_BYTES: usize = 32 * 1024 * 1024;
const MAX_KEY: usize = 1024 * 1024;
static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

/// Retention is independent of the shader source. Hits still require lowering and exact words.
#[derive(Clone, Debug, Default)]
pub enum PcuVulkanShaderCachePolicy {
    /// No artifact retention; native prepared pipelines still retain their owners.
    Disabled,
    /// Keep up to 256 cold artifacts in this device session.
    #[default]
    MemoryOnly,
    /// Persist generated modules, optionally retaining their words in RAM.
    Disk(PcuVulkanShaderDiskConfig),
}

/// Disk artifacts use a separate metadata file beside a standard little-endian `.spv` file.
#[derive(Clone, Debug)]
pub struct PcuVulkanShaderDiskConfig {
    pub directory: PathBuf,
    pub retain_in_memory: bool,
    /// Corrupt or mismatched cached modules are rebuilt only when explicitly enabled.
    pub rebuild_invalid: bool,
}
impl PcuVulkanShaderDiskConfig {
    /// Selects `pcu-shaders` beside the current executable.
    /// # Errors
    /// Returns an executable path discovery failure.
    pub fn beside_executable() -> io::Result<Self> {
        let executable = std::env::current_exe()?;
        let parent = executable
            .parent()
            .ok_or_else(|| io::Error::other("executable has no parent"))?;
        Ok(Self {
            directory: parent.join("pcu-shaders"),
            retain_in_memory: true,
            rebuild_invalid: false,
        })
    }
}

/// A complete producer-specific cold request identity. The digest is only a directory index hint.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct PcuVulkanShaderArtifactKey(Vec<u8>);
impl PcuVulkanShaderArtifactKey {
    /// Includes all typed IR fields, entry, shape, policies and effective device capabilities.
    /// This initial format is host and producer specific; Rust Hash is not a portable IR codec.
    /// The revision-bound encoding records each Hash write with its byte length; integers
    /// use little endian and pointer-sized integers use 64 bits. It is not a compiler bypass.
    /// # Errors
    /// Refuses nested/cyclic regions, oversized tables or a complete encoding beyond 1 MiB.
    pub fn for_kernel(
        kernel: &PcuDispatchKernelIr<'_>,
        caps: PcuVulkanCaps,
    ) -> Result<Self, PcuVulkanShaderArtifactError> {
        validate_ops(kernel.ops)?;
        if [
            kernel.bindings.len(),
            kernel.ports.len(),
            kernel.parameters.len(),
        ]
        .into_iter()
        .any(|length| length > 4096)
        {
            return Err(PcuVulkanShaderArtifactError::TooLarge);
        }
        let mut recorder = Recorder(REVISION.to_vec(), false);
        std::env::consts::ARCH.hash(&mut recorder);
        std::env::consts::OS.hash(&mut recorder);
        cfg!(target_endian = "little").hash(&mut recorder);
        kernel.hash(&mut recorder);
        caps.hash(&mut recorder);
        if recorder.1 || recorder.0.len() > MAX_KEY - 16 {
            return Err(PcuVulkanShaderArtifactError::TooLarge);
        }
        Ok(Self(recorder.0))
    }
    /// Disambiguates multiple lowered modules for the same typed request.
    #[must_use]
    pub fn with_module(&self, words: &[u32]) -> Self {
        let mut key = self.clone();
        key.0.extend_from_slice(&(words.len() as u64).to_le_bytes());
        let digest = words
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .fold(0xcbf2_9ce4_8422_2325_u64, |state, byte| {
                (state ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
            });
        key.0.extend_from_slice(&digest.to_le_bytes());
        key
    }
    /// Returns the persisted module path. Metadata is always checked independently.
    #[must_use]
    pub fn module_path(&self, root: &Path) -> PathBuf {
        self.path(root).join("shader.spv")
    }
    fn path(&self, root: &Path) -> PathBuf {
        let digest = self
            .0
            .iter()
            .fold(0xcbf2_9ce4_8422_2325_u64, |state, byte| {
                (state ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
            });
        root.join(format!("{digest:016x}"))
    }
}
pub fn validate_ops(
    ops: &[fusion_pcu::PcuDispatchOp<'_>],
) -> Result<(), PcuVulkanShaderArtifactError> {
    if ops.len() > 4096 {
        return Err(PcuVulkanShaderArtifactError::TooLarge);
    }
    // An iterative one-region scan rejects nesting, including cyclic borrowed IR,
    // before invoking the recursively derived Hash implementation.
    let mut remaining = 4096 - ops.len();
    for operation in ops {
        if let fusion_pcu::PcuDispatchOp::GridStrideLoop { body, .. } = operation {
            if body.len() > remaining {
                return Err(PcuVulkanShaderArtifactError::TooLarge);
            }
            remaining -= body.len();
            if body
                .iter()
                .any(|nested| matches!(nested, fusion_pcu::PcuDispatchOp::GridStrideLoop { .. }))
            {
                return Err(PcuVulkanShaderArtifactError::InvalidRequest);
            }
        }
    }
    Ok(())
}
struct Recorder(Vec<u8>, bool);
macro_rules! record_integer {
    ($method:ident, $ty:ty) => {
        fn $method(&mut self, value: $ty) {
            self.write(&value.to_le_bytes());
        }
    };
}
impl Hasher for Recorder {
    fn finish(&self) -> u64 {
        0
    }
    fn write(&mut self, bytes: &[u8]) {
        let remaining = MAX_KEY.saturating_sub(self.0.len());
        if self.1 || remaining < 8 || bytes.len() > remaining - 8 {
            self.1 = true;
            return;
        }
        self.0
            .extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        self.0.extend_from_slice(bytes);
    }
    record_integer!(write_u8, u8);
    record_integer!(write_u16, u16);
    record_integer!(write_u32, u32);
    record_integer!(write_u64, u64);
    record_integer!(write_u128, u128);
    record_integer!(write_i8, i8);
    record_integer!(write_i16, i16);
    record_integer!(write_i32, i32);
    record_integer!(write_i64, i64);
    record_integer!(write_i128, i128);
    fn write_usize(&mut self, value: usize) {
        self.write_u64(value as u64);
    }
    fn write_isize(&mut self, value: isize) {
        self.write_i64(value as i64);
    }
}

/// Cold artifact failure. No failure silently changes numerical or ownership contracts.
#[derive(Debug)]
pub enum PcuVulkanShaderArtifactError {
    Io(io::Error),
    InvalidModule,
    InvalidTemplate,
    InvalidRequest,
    MissingEmbeddedTemplates,
    Mismatched,
    TooLarge,
    MissingRequest,
}
impl fmt::Display for PcuVulkanShaderArtifactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "shader artifact IO: {error}"),
            Self::InvalidRequest => {
                f.write_str("shader artifact request has unsupported nested regions")
            }
            Self::InvalidTemplate => f.write_str(
                "external composed package does not match the exact known compiler asset",
            ),
            Self::MissingEmbeddedTemplates => f.write_str(
                "embedded composed shaders excluded; configure an external package source",
            ),
            Self::InvalidModule => f.write_str("invalid shader artifact module"),
            Self::Mismatched => f.write_str("shader artifact request or lowered words mismatch"),
            Self::TooLarge => f.write_str("shader artifact exceeds bounded storage limit"),
            Self::MissingRequest => {
                f.write_str("disk shader artifacts require an admitted typed PCU request")
            }
        }
    }
}
impl std::error::Error for PcuVulkanShaderArtifactError {}
impl From<io::Error> for PcuVulkanShaderArtifactError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

fn validate(words: &[u32]) -> Result<(), PcuVulkanShaderArtifactError> {
    if words.len() > MAX_BYTES / 4 {
        return Err(PcuVulkanShaderArtifactError::TooLarge);
    }
    if words.len() < 5
        || words[0] != 0x0723_0203
        || words[1] & 0xff00_00ff != 0
        || !(0x0001_0000..=0x0001_0600).contains(&words[1])
        || words[3] == 0
        || words[4] != 0
    {
        return Err(PcuVulkanShaderArtifactError::InvalidModule);
    }
    let mut offset = 5;
    while offset < words.len() {
        let count = (words[offset] >> 16) as usize;
        if count == 0 || count > words.len() - offset {
            return Err(PcuVulkanShaderArtifactError::InvalidModule);
        }
        offset += count;
    }
    Ok(())
}
fn read_bounded(path: &Path, limit: usize) -> Result<Vec<u8>, PcuVulkanShaderArtifactError> {
    let mut bytes = Vec::new();
    fs::File::open(path)?
        .take((limit + 1) as u64)
        .read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        return Err(PcuVulkanShaderArtifactError::TooLarge);
    }
    Ok(bytes)
}
/// Reads a standard `.spv` artifact and requires exact complete metadata and expected words.
/// # Errors
/// Returns IO, bounded-format, identity or module mismatch failures.
pub fn load_shader_artifact(
    root: &Path,
    key: &PcuVulkanShaderArtifactKey,
    expected: &[u32],
) -> Result<Vec<u32>, PcuVulkanShaderArtifactError> {
    validate(expected)?;
    let directory = key.path(root);
    if read_bounded(&directory.join("request.bin"), MAX_KEY)? != key.0 {
        return Err(PcuVulkanShaderArtifactError::Mismatched);
    }
    let bytes = read_bounded(&directory.join("shader.spv"), MAX_BYTES)?;
    if !bytes.len().is_multiple_of(4) {
        return Err(PcuVulkanShaderArtifactError::InvalidModule);
    }
    let words: Vec<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    validate(&words)?;
    if words != expected {
        return Err(PcuVulkanShaderArtifactError::Mismatched);
    }
    Ok(words)
}
/// Atomically publishes a complete directory containing metadata and a standard `.spv` file.
/// Existing matching artifacts are retained; collisions or concurrent differing writes fail.
/// # Errors
/// Returns IO, invalid module, bounded storage or existing artifact mismatch failures.
pub fn write_shader_artifact(
    root: &Path,
    key: &PcuVulkanShaderArtifactKey,
    words: &[u32],
) -> Result<(), PcuVulkanShaderArtifactError> {
    validate(words)?;
    if key.0.len() > MAX_KEY {
        return Err(PcuVulkanShaderArtifactError::TooLarge);
    }
    fs::create_dir_all(root)?;
    let destination = key.path(root);
    if destination.exists() {
        load_shader_artifact(root, key, words)?;
        return Ok(());
    }
    let temporary = root.join(format!(
        ".tmp-{}-{}",
        std::process::id(),
        NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&temporary)?;
    let result = (|| {
        let mut metadata = fs::File::create(temporary.join("request.bin"))?;
        metadata.write_all(&key.0)?;
        metadata.sync_all()?;
        let mut module = BufWriter::new(fs::File::create(temporary.join("shader.spv"))?);
        for word in words {
            module.write_all(&word.to_le_bytes())?;
        }
        module.flush()?;
        module.get_ref().sync_all()?;
        match fs::rename(&temporary, &destination) {
            Ok(()) => Ok(()),
            Err(error) if destination.exists() => {
                load_shader_artifact(root, key, words)?;
                drop(error);
                Ok(())
            }
            Err(error) => Err(error.into()),
        }
    })();
    if temporary.exists() {
        let _ = fs::remove_dir_all(&temporary);
    }
    result
}
/// Removes persisted bytes only. Already prepared native owners remain valid.
/// # Errors
/// Returns metadata mismatch or filesystem failures.
pub fn unload_shader_artifact(
    root: &Path,
    key: &PcuVulkanShaderArtifactKey,
) -> Result<(), PcuVulkanShaderArtifactError> {
    let directory = key.path(root);
    if read_bounded(&directory.join("request.bin"), MAX_KEY)? != key.0 {
        return Err(PcuVulkanShaderArtifactError::Mismatched);
    }
    fs::remove_dir_all(directory)?;
    Ok(())
}

/// Cold cache observations; no counters or artifact lookups occur during prepared calls.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PcuVulkanShaderCacheStats {
    pub memory_hits: u64,
    pub disk_hits: u64,
    pub disk_misses: u64,
    pub rebuilds: u64,
}

#[derive(Default)]
pub struct ArtifactStore {
    pub(crate) policy: PcuVulkanShaderCachePolicy,
    pub(crate) request: Option<PcuVulkanShaderArtifactKey>,
    memory: HashMap<PcuVulkanShaderArtifactKey, Vec<u32>>,
    pub(crate) stats: PcuVulkanShaderCacheStats,
}
impl ArtifactStore {
    pub(crate) fn configure(&mut self, policy: PcuVulkanShaderCachePolicy) {
        self.policy = policy;
        self.memory.clear();
        self.stats = PcuVulkanShaderCacheStats::default();
    }
    pub(crate) fn resolve<'a>(
        &mut self,
        words: &'a [u32],
    ) -> Result<Cow<'a, [u32]>, PcuVulkanShaderArtifactError> {
        if matches!(self.policy, PcuVulkanShaderCachePolicy::Disabled) {
            return Ok(Cow::Borrowed(words));
        }
        let Some(request) = self.request.as_ref() else {
            return if matches!(self.policy, PcuVulkanShaderCachePolicy::Disk(_)) {
                Err(PcuVulkanShaderArtifactError::MissingRequest)
            } else {
                Ok(Cow::Borrowed(words))
            };
        };
        // Full lowered module bytes disambiguate multiple modules emitted for one typed request.
        validate(words)?;
        let key = request.with_module(words);
        if key.0.len() > MAX_KEY {
            return Err(PcuVulkanShaderArtifactError::TooLarge);
        }
        if let Some(found) = self.memory.get(&key) {
            if found != words {
                return Err(PcuVulkanShaderArtifactError::Mismatched);
            }
            self.stats.memory_hits += 1;
            return Ok(Cow::Owned(found.clone()));
        }
        let (resolved, retain) = match &self.policy {
            PcuVulkanShaderCachePolicy::Disk(config) => {
                let loaded = load_shader_artifact(&config.directory, &key, words);
                let resolved = match loaded {
                    Ok(found) => {
                        self.stats.disk_hits += 1;
                        found
                    }
                    Err(PcuVulkanShaderArtifactError::Io(error))
                        if error.kind() == io::ErrorKind::NotFound
                            && !key.path(&config.directory).exists() =>
                    {
                        write_shader_artifact(&config.directory, &key, words)?;
                        self.stats.disk_misses += 1;
                        words.to_vec()
                    }
                    Err(
                        PcuVulkanShaderArtifactError::InvalidModule
                        | PcuVulkanShaderArtifactError::Mismatched
                        | PcuVulkanShaderArtifactError::TooLarge,
                    ) if config.rebuild_invalid => {
                        // Only an exact matching request can be removed; a filename collision refuses.
                        unload_shader_artifact(&config.directory, &key)?;
                        write_shader_artifact(&config.directory, &key, words)?;
                        self.stats.rebuilds += 1;
                        words.to_vec()
                    }
                    Err(error) => return Err(error),
                };
                (resolved, config.retain_in_memory)
            }
            _ => (words.to_vec(), true),
        };
        let retained_bytes: usize = self
            .memory
            .iter()
            .map(|(stored_key, module)| stored_key.0.len() + module.len() * 4)
            .sum();
        if retain
            && self.memory.len() < 256
            && key.0.len() + resolved.len() * 4 <= MAX_BYTES.saturating_sub(retained_bytes)
        {
            self.memory.insert(key, resolved.clone());
        }
        Ok(Cow::Owned(resolved))
    }
}
/// Restores cold request context even when preparation fails or unwinds.
pub struct RequestGuard<'a> {
    store: &'a RefCell<ArtifactStore>,
    previous: Option<PcuVulkanShaderArtifactKey>,
}
impl<'a> RequestGuard<'a> {
    pub(crate) fn new(
        store: &'a RefCell<ArtifactStore>,
        request: Option<PcuVulkanShaderArtifactKey>,
    ) -> Self {
        let previous = std::mem::replace(&mut store.borrow_mut().request, request);
        Self { store, previous }
    }
}
impl Drop for RequestGuard<'_> {
    fn drop(&mut self) {
        self.store.borrow_mut().request = self.previous.take();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (PathBuf, PcuVulkanShaderArtifactKey, Vec<u32>) {
        let root = std::env::temp_dir().join(format!(
            "pcu-artifact-test-{}-{}",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        // A bounded structural fixture is sufficient for persistence tests; native gates use
        // the actual admitted PCU emitter and execute the resulting shader on a device.
        (
            root,
            PcuVulkanShaderArtifactKey(b"test-producer-key".to_vec()),
            vec![0x0723_0203, 0x0001_0000, 0, 1, 0, 0x0001_0000],
        )
    }
    #[test]
    fn recursive_regions_and_unbounded_recordings_are_refused() {
        use fusion_pcu::PcuDispatchOp;
        static CYCLIC: [PcuDispatchOp<'static>; 1] = [PcuDispatchOp::GridStrideLoop {
            extent: 1,
            body: &CYCLIC,
        }];
        assert!(matches!(
            validate_ops(&CYCLIC),
            Err(PcuVulkanShaderArtifactError::InvalidRequest)
        ));
        let excessive = vec![
            PcuDispatchOp::GridStrideLoop {
                extent: 1,
                body: &[]
            };
            4097
        ];
        assert!(matches!(
            validate_ops(&excessive),
            Err(PcuVulkanShaderArtifactError::TooLarge)
        ));
        let body = vec![
            PcuDispatchOp::GridStrideLoop {
                extent: 1,
                body: &[]
            };
            4096
        ];
        let repeated = [PcuDispatchOp::GridStrideLoop {
            extent: 1,
            body: &body,
        }];
        assert!(matches!(
            validate_ops(&repeated),
            Err(PcuVulkanShaderArtifactError::TooLarge)
        ));
        let mut recorder = Recorder(Vec::new(), false);
        recorder.write(&vec![0; MAX_KEY]);
        assert!(recorder.1);
        assert!(recorder.0.is_empty());
        recorder.write(b"later metadata");
        assert!(recorder.0.is_empty());
        let mut full = Recorder(vec![0; MAX_KEY], false);
        full.write(&[]);
        assert!(full.1);
        assert_eq!(full.0.len(), MAX_KEY);
    }
    #[test]
    fn persistence_checks_metadata_bytes_and_instruction_boundaries() {
        let (root, key, words) = fixture();
        write_shader_artifact(&root, &key, &words).unwrap();
        assert_eq!(load_shader_artifact(&root, &key, &words).unwrap(), words);
        let bytes = fs::read(key.module_path(&root)).unwrap();
        assert_eq!(&bytes[..4], &[3, 2, 35, 7]);
        let mut wrong = words.clone();
        wrong[2] = 7;
        assert!(matches!(
            load_shader_artifact(&root, &key, &wrong),
            Err(PcuVulkanShaderArtifactError::Mismatched)
        ));
        fs::write(key.path(&root).join("request.bin"), b"collision").unwrap();
        assert!(matches!(
            load_shader_artifact(&root, &key, &words),
            Err(PcuVulkanShaderArtifactError::Mismatched)
        ));
        assert!(unload_shader_artifact(&root, &key).is_err());
        fs::write(key.path(&root).join("request.bin"), &key.0).unwrap();
        fs::write(key.module_path(&root), [1, 2, 3]).unwrap();
        assert!(matches!(
            load_shader_artifact(&root, &key, &words),
            Err(PcuVulkanShaderArtifactError::InvalidModule)
        ));
        unload_shader_artifact(&root, &key).unwrap();
        assert!(fs::read_dir(&root).unwrap().next().is_none());
        fs::remove_dir(&root).unwrap();
        wrong[5] = 0;
        assert!(matches!(
            validate(&wrong),
            Err(PcuVulkanShaderArtifactError::InvalidModule)
        ));
    }
    #[test]
    fn cache_recovery_is_explicit_and_memory_owners_survive_file_removal() {
        let (root, request, words) = fixture();
        let key = request.with_module(&words);
        let mut store = ArtifactStore {
            request: Some(request),
            ..Default::default()
        };
        let config = PcuVulkanShaderDiskConfig {
            directory: root.clone(),
            retain_in_memory: false,
            rebuild_invalid: false,
        };
        store.configure(PcuVulkanShaderCachePolicy::Disk(config.clone()));
        assert_eq!(store.resolve(&words).unwrap(), words);
        fs::write(key.module_path(&root), [0; 4]).unwrap();
        assert!(store.resolve(&words).is_err());
        store.configure(PcuVulkanShaderCachePolicy::Disk(
            PcuVulkanShaderDiskConfig {
                rebuild_invalid: true,
                retain_in_memory: true,
                ..config
            },
        ));
        assert_eq!(store.resolve(&words).unwrap(), words);
        unload_shader_artifact(&root, &key).unwrap();
        assert_eq!(store.resolve(&words).unwrap(), words);
        assert!(!key.module_path(&root).exists());
        fs::remove_dir(&root).unwrap();
    }
    #[test]
    fn request_context_restores_on_failure_and_absent_request_refuses_disk() {
        let store = RefCell::new(ArtifactStore::default());
        {
            let _guard = RequestGuard::new(&store, Some(PcuVulkanShaderArtifactKey(vec![1])));
            assert!(store.borrow().request.is_some());
        }
        assert!(store.borrow().request.is_none());
        let (root, _, words) = fixture();
        store
            .borrow_mut()
            .configure(PcuVulkanShaderCachePolicy::Disk(
                PcuVulkanShaderDiskConfig {
                    directory: root.clone(),
                    retain_in_memory: true,
                    rebuild_invalid: false,
                },
            ));
        assert!(matches!(
            store.borrow_mut().resolve(&words),
            Err(PcuVulkanShaderArtifactError::MissingRequest)
        ));
        assert!(!root.exists());
    }
}
