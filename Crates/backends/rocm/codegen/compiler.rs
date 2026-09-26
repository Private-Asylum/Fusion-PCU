//! Host-side HIP source compilation for a selected AMD GPU architecture.

use std::{
    fmt,
    fs,
    io,
    path::PathBuf,
    process::Command,
    sync::atomic::{
        AtomicU64,
        Ordering,
    },
};

static NEXT_BUILD: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
pub enum HipCompileError {
    Io(io::Error),
    CompilerFailed { status: Option<i32>, output: String },
    InvalidArchitecture,
}

impl fmt::Display for HipCompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "HIP compilation I/O failed: {error}"),
            Self::CompilerFailed { status, output } => {
                write!(f, "hipcc exited with {status:?}: {output}")
            }
            Self::InvalidArchitecture => f.write_str("invalid HIP architecture target"),
        }
    }
}

impl std::error::Error for HipCompileError {}

impl From<io::Error> for HipCompileError {
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

/// Compile HIP source into an AMD GPU code object without executing it.
///
/// `architecture` must be an AMD LLVM target such as `gfx1030`. The compiler is selected by
/// `HIPCC` or defaults to `hipcc`. Each call uses a private temporary directory and passes all
/// arguments directly to the process without a shell.
///
/// # Errors
///
/// Returns an I/O error, invalid target error, or compiler diagnostics.
pub fn compile_hip_source(source: &str, architecture: &str) -> Result<Vec<u8>, HipCompileError> {
    if !architecture.starts_with("gfx")
        || architecture.len() < 6
        || !architecture
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric())
    {
        return Err(HipCompileError::InvalidArchitecture);
    }

    let directory = loop {
        let id = NEXT_BUILD.fetch_add(1, Ordering::Relaxed);
        let candidate =
            std::env::temp_dir().join(format!("fusion-pcu-hip-{}-{id}", std::process::id()));
        match fs::create_dir(&candidate) {
            Ok(()) => break BuildDirectory(candidate),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(HipCompileError::Io(error)),
        }
    };
    let input = directory.0.join("kernel.hip");
    let output = directory.0.join("kernel.hsaco");
    fs::write(&input, source)?;

    let compiler = std::env::var_os("HIPCC").unwrap_or_else(|| "hipcc".into());
    let result = Command::new(compiler)
        .arg("--genco")
        .arg(format!("--offload-arch={architecture}"))
        .arg(&input)
        .arg("-o")
        .arg(&output)
        .output()?;
    if !result.status.success() {
        return Err(HipCompileError::CompilerFailed {
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
