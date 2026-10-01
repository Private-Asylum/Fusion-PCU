//! Pinned archive acquisition. `CMake` receives only verified local sources.
#[rustfmt::skip]
use std::{
    fs,
    path::{
        Component,
        Path,
        PathBuf,
    },
    process::Command,
};

use crate::prepare;
use crate::output as generated;
pub struct Pin<'a> {
    pub name: &'a str,
    pub url: &'a str,
    pub archive_sha256: &'a str,
    pub directory: &'a str,
    pub inventory: &'a str,
}
pub const PINS: &[Pin<'static>] = &[
    Pin {
        name: "mlx",
        url: "https://codeload.github.com/ml-explore/mlx/tar.gz/64ea011cb65f14d9ce2737e60db9a4ae91ed7441",
        archive_sha256: "428070f9b74ab39b5f65ae90a0a409c14b1a7a75911d041864788bfbdb9f7ef0",
        directory: "mlx-64ea011cb65f14d9ce2737e60db9a4ae91ed7441",
        inventory: include_str!("pins/mlx.sha256"),
    },
    Pin {
        name: "mlx-c",
        url: "https://codeload.github.com/ml-explore/mlx-c/tar.gz/a341b4925024b88b2c593468f16e12f5e17315da",
        archive_sha256: "4424dd3f6225708d111b691be4041bba9bc6da09719ae8d32e6900744852c7f6",
        directory: "mlx-c-a341b4925024b88b2c593468f16e12f5e17315da",
        inventory: include_str!("../../../native/patches/upstream.sha256"),
    },
    Pin {
        name: "metal",
        url: "https://developer.apple.com/metal/cpp/files/metal-cpp_26.zip",
        archive_sha256: "4df3c078b9aadcb516212e9cb03004cbc5ce9a3e9c068fa3144d021db585a3a4",
        directory: "metal-cpp",
        inventory: include_str!("pins/metal.sha256"),
    },
    Pin {
        name: "json",
        url: "https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz",
        archive_sha256: "d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d",
        directory: "json",
        inventory: include_str!("pins/json.sha256"),
    },
    Pin {
        name: "fmt",
        url: "https://codeload.github.com/fmtlib/fmt/tar.gz/407c905e45ad75fc29bf0f9bb7c5c2fd3475976f",
        archive_sha256: "2bc1fe4a5b6d5d6a614239b4ca1d520e66e152a02d3d262684d26dfd6ab3438a",
        directory: "fmt-407c905e45ad75fc29bf0f9bb7c5c2fd3475976f",
        inventory: include_str!("pins/fmt.sha256"),
    },
    Pin {
        name: "gguf",
        url: "https://codeload.github.com/antirez/gguf-tools/tar.gz/8fa6eb65236618e28fd7710a0fba565f7faa1848",
        archive_sha256: "9e30bc1eb82cc2231150d39ce37dcdd6f844d6994fba18da83fc537a487ba86f",
        directory: "gguf-tools-8fa6eb65236618e28fd7710a0fba565f7faa1848",
        inventory: include_str!("pins/gguf.sha256"),
    },
];
pub fn validate_members(names: &str, kinds: &str, prefix: &str) -> prepare::Result<()> {
    let names: Vec<_> = names.lines().collect();
    let kinds: Vec<_> = kinds.lines().collect();
    if names.is_empty() || names.len() != kinds.len() {
        return Err("archive member inventory mismatch".into());
    }
    for (name, kind) in names.iter().zip(kinds) {
        let path = Path::new(name);
        if !path.components().all(|c| matches!(c, Component::Normal(_)))
            || !path.starts_with(prefix)
            || !(matches!(kind.as_bytes().first(), Some(b'-' | b'd'))
                || (prefix == "mlx-64ea011cb65f14d9ce2737e60db9a4ae91ed7441"
                    && *name == format!("{prefix}/CLAUDE.md")
                    && kind.starts_with('l')
                    && kind.ends_with(" -> AGENTS.md")))
        {
            return Err(format!("unsafe archive member: {name}"));
        }
    }
    Ok(())
}
fn output(command: &mut Command) -> prepare::Result<String> {
    let out = command.output().map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).into_owned());
    }
    String::from_utf8(out.stdout).map_err(|e| e.to_string())
}
pub fn acquire(cache: &Path, pin: &Pin<'_>) -> prepare::Result<PathBuf> {
    let offline =
        std::env::var("CARGO_NET_OFFLINE").is_ok_and(|value| value == "true" || value == "1");
    acquire_with_network(cache, pin, !offline)
}
pub fn acquire_with_network(
    cache: &Path,
    pin: &Pin<'_>,
    network_allowed: bool,
) -> prepare::Result<PathBuf> {
    generated::directory(cache)?;
    let archive = cache.join(format!("{}.archive", pin.name));
    generated::regular(&archive)?;
    if !archive.exists() {
        if !network_allowed {
            return Err(format!(
                "offline native source cache missing {} (expected archive SHA256 {}): {}",
                pin.name,
                pin.archive_sha256,
                archive.display()
            ));
        }
        let temporary = cache.join(format!("{}.download", pin.name));
        let reserved = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|e| {
                format!(
                    "exclusive download output refused {}: {e}",
                    temporary.display()
                )
            })?;
        drop(reserved);
        let download = output(
            Command::new("curl")
                .args([
                    "--fail",
                    "--location",
                    "--silent",
                    "--show-error",
                    "--proto",
                    "=https",
                    "--proto-redir",
                    "=https",
                    "--tlsv1.2",
                    "--connect-timeout",
                    "30",
                    "--max-time",
                    "600",
                    "--output",
                ])
                .arg(&temporary)
                .arg(pin.url),
        );
        if let Err(error) = download {
            let _ = fs::remove_file(&temporary);
            return Err(error);
        }
        if prepare::sha256(&fs::read(&temporary).map_err(|e| e.to_string())?) != pin.archive_sha256
        {
            let _ = fs::remove_file(temporary);
            return Err(format!("pinned download hash mismatch: {}", pin.name));
        }
        fs::rename(temporary, &archive).map_err(|e| e.to_string())?;
    }
    if prepare::sha256(&fs::read(&archive).map_err(|e| e.to_string())?) != pin.archive_sha256 {
        return Err(format!("cached archive hash mismatch: {}", pin.name));
    }
    let extracted = cache.join(format!("{}-source", pin.name));
    let source = extracted.join(pin.directory);
    match fs::symlink_metadata(&extracted) {
        Ok(metadata) if metadata.file_type().is_dir() => (),
        Ok(_) => return Err("cached extracted source must be a regular directory".into()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
        Err(error) => return Err(error.to_string()),
    }
    if !extracted.exists() {
        let names = output(Command::new("tar").arg("-tf").arg(&archive))?;
        let kinds = output(Command::new("tar").arg("-tvf").arg(&archive))?;
        validate_members(&names, &kinds, pin.directory)?;
        fs::create_dir(&extracted).map_err(|e| e.to_string())?;
        let result = output(
            Command::new("tar")
                .arg("-xf")
                .arg(&archive)
                .arg("-C")
                .arg(&extracted),
        );
        if let Err(error) = result {
            let _ = fs::remove_dir_all(&extracted);
            return Err(error);
        }
    }
    prepare::verify(&source, pin.inventory)?;
    Ok(source)
}
