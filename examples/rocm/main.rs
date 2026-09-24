use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::path::Path;
use std::process::{
    Command,
    ExitCode,
};

const HIP_SOURCE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/smoke.hip");

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("ERROR fusion-rocm-smoke: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), RunError> {
    let arch = env::var("FUSION_ROCM_ARCH").unwrap_or_else(|_| "gfx1030".to_owned());
    if !Path::new(HIP_SOURCE).is_file() {
        return Err(RunError::Failure(format!(
            "HIP source not found: {HIP_SOURCE}"
        )));
    }

    run_compiled(&arch)
}

fn run_compiled(arch: &str) -> Result<(), RunError> {
    let output = env::temp_dir().join(format!("fusion-rocm-smoke-{}", std::process::id()));
    let mut compiler =
        Command::new(env::var_os("HIPCC").unwrap_or_else(|| OsString::from("hipcc")));
    compiler
        .arg("-O2")
        .arg(format!("--offload-arch={arch}"))
        .arg(HIP_SOURCE)
        .arg("-lrocblas")
        .arg("-o")
        .arg(&output);
    let compile = compiler.output().map_err(|error| {
        RunError::Failure(format!(
            "failed to start hipcc (set HIPCC to its path): {error}"
        ))
    })?;
    if !compile.status.success() {
        return Err(RunError::Failure(format!(
            "hipcc failed for {arch}:\n{}{}",
            String::from_utf8_lossy(&compile.stdout),
            String::from_utf8_lossy(&compile.stderr)
        )));
    }

    let execution = Command::new(&output).output().map_err(|error| {
        RunError::Failure(format!("could not launch compiled HIP probe: {error}"))
    });
    let _ = std::fs::remove_file(&output);
    let execution = execution?;
    if !execution.status.success() {
        return Err(RunError::Failure(format!(
            "HIP/rocBLAS probe exited with {}:\n{}{}",
            execution.status,
            String::from_utf8_lossy(&execution.stdout),
            String::from_utf8_lossy(&execution.stderr)
        )));
    }
    print!("{}", String::from_utf8_lossy(&execution.stdout));
    eprint!("{}", String::from_utf8_lossy(&execution.stderr));
    Ok(())
}

enum RunError {
    Failure(String),
}

impl std::fmt::Display for RunError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Failure(message) => formatter.write_str(message),
        }
    }
}

impl std::fmt::Debug for RunError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, formatter)
    }
}

impl Error for RunError {}
