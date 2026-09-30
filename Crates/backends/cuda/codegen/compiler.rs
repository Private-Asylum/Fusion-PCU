//! Offline CUDA source compilation with toolkit discovery.

#[rustfmt::skip]
use std::{
    fmt,
    fs,
    io,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT_BUILD: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
pub enum CudaCompileError {
    Io(io::Error),
    CompilerNotFound { searched: Vec<PathBuf> },
    CompilerFailed { status: Option<i32>, output: String },
    InvalidArchitecture,
}

impl fmt::Display for CudaCompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "CUDA compilation I/O failed: {error}"),
            Self::CompilerNotFound { searched } => write!(
                f,
                "nvcc was not found (searched: {})",
                searched
                    .iter()
                    .map(|path| path.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Self::CompilerFailed { status, output } => {
                write!(f, "nvcc exited with {status:?}: {output}")
            }
            Self::InvalidArchitecture => f.write_str("invalid CUDA architecture target"),
        }
    }
}

impl std::error::Error for CudaCompileError {}

impl From<io::Error> for CudaCompileError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

struct BuildDirectory(PathBuf);

impl Drop for BuildDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Compile CUDA source to PTX with the locally installed NVIDIA toolkit.
///
/// `architecture` accepts CUDA targets such as `sm_86` or `compute_86`. `CUDA_NVCC` can point
/// directly at the compiler; otherwise the toolkit is searched via `CUDA_HOME`, `CUDA_PATH`, the
/// conventional `/usr/local/cuda` location, and `PATH`.
///
/// # Errors
///
/// Returns an I/O error, invalid target error, missing-toolkit error, or nvcc diagnostics.
pub fn compile_cuda_source(source: &str, architecture: &str) -> Result<Vec<u8>, CudaCompileError> {
    let architecture =
        normalize_architecture(architecture).ok_or(CudaCompileError::InvalidArchitecture)?;
    let compiler = find_nvcc()?;
    let directory = create_build_directory()?;
    let input = directory.0.join("kernel.cu");
    let output = directory.0.join("kernel.ptx");
    fs::write(&input, source)?;

    let result = Command::new(compiler)
        .arg("--ptx")
        .arg(format!("--gpu-architecture={architecture}"))
        .args([
            "--fmad=false",
            "--ftz=false",
            "--prec-div=true",
            "--prec-sqrt=true",
        ])
        .arg(&input)
        .arg("-o")
        .arg(&output)
        .output()?;
    if !result.status.success() {
        return Err(CudaCompileError::CompilerFailed {
            status: result.status.code(),
            output: format!(
                "{}{}",
                String::from_utf8_lossy(&result.stdout),
                String::from_utf8_lossy(&result.stderr)
            ),
        });
    }
    Ok(fs::read(output)?)
}

fn normalize_architecture(architecture: &str) -> Option<String> {
    let digits = architecture
        .strip_prefix("sm_")
        .or_else(|| architecture.strip_prefix("compute_"))
        .or_else(|| architecture.strip_prefix("sm"))?;
    if digits.len() < 2 || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let version = digits.parse::<u16>().ok()?;
    let major = version / 10;
    if !(5..=12).contains(&major) || version % 10 > 9 {
        return None;
    }
    Some(format!("compute_{digits}"))
}

fn find_nvcc() -> Result<PathBuf, CudaCompileError> {
    if let Some(path) = std::env::var_os("CUDA_NVCC") {
        let path = PathBuf::from(path);
        if executable_exists(&path) {
            return Ok(path);
        }
        return Err(CudaCompileError::CompilerNotFound {
            searched: vec![path],
        });
    }

    let mut candidates = Vec::new();
    for variable in ["CUDA_HOME", "CUDA_PATH"] {
        if let Some(root) = std::env::var_os(variable) {
            candidates.push(PathBuf::from(root).join("bin/nvcc"));
        }
    }
    candidates.push(PathBuf::from("/usr/local/cuda/bin/nvcc"));
    if let Ok(entries) = fs::read_dir("/usr/local") {
        candidates.extend(entries.flatten().filter_map(|entry| {
            let name = entry.file_name();
            name.to_string_lossy()
                .starts_with("cuda-")
                .then(|| entry.path().join("bin/nvcc"))
        }));
    }
    if let Some(path) = std::env::var_os("PATH") {
        candidates.extend(std::env::split_paths(&path).map(|dir| dir.join("nvcc")));
    }
    if let Some(found) = candidates
        .iter()
        .find(|candidate| executable_exists(candidate))
    {
        return Ok(found.clone());
    }
    Err(CudaCompileError::CompilerNotFound {
        searched: candidates,
    })
}

fn executable_exists(path: &Path) -> bool {
    path.is_file()
}

fn create_build_directory() -> Result<BuildDirectory, CudaCompileError> {
    loop {
        let id = NEXT_BUILD.fetch_add(1, Ordering::Relaxed);
        let candidate =
            std::env::temp_dir().join(format!("fusion-pcu-cuda-{}-{id}", std::process::id()));
        match fs::create_dir(&candidate) {
            Ok(()) => return Ok(BuildDirectory(candidate)),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(CudaCompileError::Io(error)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::normalize_architecture;

    #[test]
    fn accepts_cuda_architecture_spellings() {
        assert_eq!(
            normalize_architecture("sm_86").as_deref(),
            Some("compute_86")
        );
        assert_eq!(
            normalize_architecture("compute_120").as_deref(),
            Some("compute_120")
        );
        assert_eq!(normalize_architecture("vendor1030"), None);
        assert_eq!(normalize_architecture("sm_8a"), None);
    }
}
