//! Generated outputs never follow preexisting filesystem links.
#[rustfmt::skip]
use std::{
    fs,
    path::Path,
};
pub fn regular(path: &Path) -> Result<(), String> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_file() => Ok(()),
        Ok(_) => Err(format!(
            "generated output must be regular: {}",
            path.display()
        )),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.to_string()),
    }
}
pub fn directory(path: &Path) -> Result<(), String> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_dir() => Ok(()),
        Ok(_) => Err(format!(
            "generated directory must be regular: {}",
            path.display()
        )),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            fs::create_dir_all(path).map_err(|e| e.to_string())
        }
        Err(error) => Err(error.to_string()),
    }
}
pub fn write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    regular(path)?;
    if fs::read(path).ok().as_deref() == Some(bytes) {
        return Ok(());
    }
    atomic(path, |file| {
        std::io::Write::write_all(file, bytes).map_err(|e| e.to_string())
    })
}
pub fn copy(source: &Path, destination: &Path) -> Result<(), String> {
    regular(destination)?;
    atomic(destination, |file| {
        let mut source = fs::File::open(source).map_err(|e| e.to_string())?;
        std::io::copy(&mut source, file).map_err(|e| e.to_string())?;
        Ok(())
    })
}
fn atomic(
    path: &Path,
    write: impl FnOnce(&mut fs::File) -> Result<(), String>,
) -> Result<(), String> {
    let temporary = path.with_extension("pcu-temporary");
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|e| e.to_string())?;
    let result = (|| {
        write(&mut file)?;
        file.sync_all().map_err(|e| e.to_string())?;
        fs::rename(&temporary, path).map_err(|e| e.to_string())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}
