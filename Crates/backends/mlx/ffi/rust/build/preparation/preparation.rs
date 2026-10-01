//! Exact offline source preparation shared by Cargo and canonical tests.
#[rustfmt::skip]
use std::{
    fmt::Write as _,
    collections::BTreeMap,
    fs,
    path::{
        Component,
        Path,
    },
};

#[rustfmt::skip]
use sha2::{
    Digest,
    Sha256,
};

use crate::output;
pub const REVISION: &str = "a341b4925024b88b2c593468f16e12f5e17315da";
const ORIGINAL: &str = include_str!("../../../native/patches/upstream.sha256");
const PREPARED: &str = include_str!("../../../native/patches/prepared.sha256");
const PATCH: &str = include_str!("../../../native/patches/safety.patch");
pub type Result<T> = std::result::Result<T, String>;
pub fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn safe_path(s: &str) -> Result<()> {
    if s.is_empty()
        || s.contains('\\')
        || !Path::new(s)
            .components()
            .all(|c| matches!(c, Component::Normal(_)))
    {
        return Err(format!("unsafe source path {s:?}"));
    }
    Ok(())
}
pub fn manifest(text: &str) -> Result<BTreeMap<String, String>> {
    let mut out = BTreeMap::new();
    for line in text.lines() {
        let (hash, path) = line.split_once("  ").ok_or("invalid hash manifest")?;
        safe_path(path)?;
        if hash.len() != 64
            || !hash
                .bytes()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
            || out.insert(path.into(), hash.into()).is_some()
        {
            return Err("invalid or duplicate manifest record".into());
        }
    }
    if out.is_empty() {
        return Err("empty source manifest".into());
    }
    Ok(out)
}
fn inventory(root: &Path, rel: &Path, out: &mut BTreeMap<String, String>) -> Result<()> {
    for entry in fs::read_dir(root.join(rel)).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = rel.join(entry.file_name());
        if rel.as_os_str().is_empty() && entry.file_name() == ".git" {
            continue;
        }
        let kind = entry.file_type().map_err(|e| e.to_string())?;
        if kind.is_symlink() {
            let target = fs::read_link(entry.path()).map_err(|e| e.to_string())?;
            let target_name = target.to_str().ok_or("non UTF-8 symlink")?;
            safe_path(target_name)?;
            let resolved = entry.path().canonicalize().map_err(|e| e.to_string())?;
            if !resolved.starts_with(root.canonicalize().map_err(|e| e.to_string())?) {
                return Err("source symlink escape refused".into());
            }
            let name = path.to_str().ok_or("non UTF-8 path")?.to_string();
            out.insert(name, sha256(format!("symlink:{target_name}").as_bytes()));
            continue;
        }
        if kind.is_dir() {
            inventory(root, &path, out)?;
        } else if kind.is_file() {
            let name = path
                .to_str()
                .ok_or("non UTF-8 source path")?
                .replace('\\', "/");
            out.insert(
                name,
                sha256(&fs::read(entry.path()).map_err(|e| e.to_string())?),
            );
        } else {
            return Err("non regular source entry".into());
        }
    }
    Ok(())
}
pub fn verify(root: &Path, text: &str) -> Result<()> {
    let expected = manifest(text)?;
    let mut actual = BTreeMap::new();
    inventory(root, Path::new(""), &mut actual)?;
    if actual != expected {
        let differences: Vec<_> = expected
            .keys()
            .chain(actual.keys())
            .filter(|p| expected.get(*p) != actual.get(*p))
            .take(8)
            .collect();
        return Err(format!(
            "exact source inventory/hash mismatch: {differences:?}"
        ));
    }
    Ok(())
}
fn range(s: &str) -> Result<(usize, usize)> {
    let s = s.get(1..).ok_or("bad hunk range")?;
    let (start, count) = s.split_once(',').unwrap_or((s, "1"));
    Ok((
        start.parse().map_err(|_| "bad hunk start")?,
        count.parse().map_err(|_| "bad hunk count")?,
    ))
}
/// Apply unified hunks with exact line positions, context and counts. No fuzz or offsets.
pub fn apply_patch(source: &str, patch: &str) -> Result<String> {
    let original: Vec<_> = source.split_inclusive('\n').collect();
    let lines: Vec<_> = patch.split_inclusive('\n').collect();
    let mut output = String::new();
    let mut cursor = 0;
    let mut i = 0;
    while i < lines.len() {
        let header = lines[i].trim_end_matches('\n');
        if !header.starts_with("@@ ") {
            return Err("expected unified hunk".into());
        }
        let fields: Vec<_> = header.split_whitespace().collect();
        if fields.len() != 4
            || fields[0] != "@@"
            || fields[3] != "@@"
            || !fields[1].starts_with('-')
            || !fields[2].starts_with('+')
        {
            return Err("invalid hunk header".into());
        }
        let (old_start, old_count) = range(fields[1])?;
        let (new_start, new_count) = range(fields[2])?;
        let start = if old_count == 0 {
            old_start
        } else {
            old_start.checked_sub(1).ok_or("zero hunk start")?
        };
        if start < cursor || start > original.len() {
            return Err("hunk overlap or outside source".into());
        }
        for line in &original[cursor..start] {
            output.push_str(line);
        }
        cursor = start;
        let expected_new = if new_count == 0 {
            new_start
        } else {
            new_start.checked_sub(1).ok_or("zero new start")?
        };
        if output.split_inclusive('\n').count() != expected_new {
            return Err("new hunk position mismatch".into());
        }
        i += 1;
        let mut consumed = 0;
        let mut produced = 0;
        while i < lines.len() && !lines[i].starts_with("@@ ") {
            let line = lines[i];
            let body = line.get(1..).ok_or("empty patch line")?;
            match line.as_bytes()[0] {
                b' ' | b'-' => {
                    if original.get(cursor).copied() != Some(body) {
                        return Err(format!(
                            "hunk context mismatch at source line {}",
                            cursor + 1
                        ));
                    }
                    cursor += 1;
                    consumed += 1;
                    if line.starts_with(' ') {
                        output.push_str(body);
                        produced += 1;
                    }
                }
                b'+' => {
                    output.push_str(body);
                    produced += 1;
                }
                _ => return Err("unsupported unified patch line".into()),
            }
            i += 1;
        }
        if consumed != old_count || produced != new_count {
            return Err("hunk line count mismatch".into());
        }
    }
    for line in &original[cursor..] {
        output.push_str(line);
    }
    Ok(output)
}
fn patch_files() -> Result<BTreeMap<String, String>> {
    let mut out = BTreeMap::new();
    let mut lines = PATCH.split_inclusive('\n').peekable();
    while let Some(old) = lines.next() {
        let path = old
            .strip_prefix("--- a/")
            .ok_or("bad original patch file header")?
            .trim_end_matches('\n');
        safe_path(path)?;
        let new = lines.next().ok_or("missing new patch header")?;
        if new != format!("+++ b/{path}\n") {
            return Err("patch file rename refused".into());
        }
        let mut body = String::new();
        while lines.peek().is_some_and(|line| !line.starts_with("--- a/")) {
            body.push_str(lines.next().ok_or("patch exhausted")?);
        }
        if body.is_empty() || out.insert(path.into(), body).is_some() {
            return Err("empty or duplicate file patch".into());
        }
    }
    Ok(out)
}
/// Resolve existing ancestors before any write; neither tree may contain the other.
pub fn disjoint(source: &Path, destination: &Path) -> Result<()> {
    if destination
        .components()
        .any(|c| matches!(c, Component::ParentDir))
    {
        return Err("destination traversal refused".into());
    }
    let source = source.canonicalize().map_err(|e| e.to_string())?;
    let mut ancestor = destination.to_path_buf();
    let mut suffix = Vec::new();
    while !ancestor.exists() {
        suffix.push(
            ancestor
                .file_name()
                .ok_or("destination lacks existing ancestor")?
                .to_os_string(),
        );
        ancestor = ancestor
            .parent()
            .ok_or("destination lacks parent")?
            .to_path_buf();
    }
    let mut destination = ancestor.canonicalize().map_err(|e| e.to_string())?;
    for component in suffix.iter().rev() {
        destination.push(component);
    }
    if source.starts_with(&destination) || destination.starts_with(&source) {
        return Err("source and prepared destination must be disjoint".into());
    }
    Ok(())
}
pub fn verify_prepared(root: &Path) -> Result<()> {
    validate_metadata(root)?;
    let expected = manifest(PREPARED)?;
    let mut actual = BTreeMap::new();
    inventory(root, Path::new(""), &mut actual)?;
    for metadata in [
        "provenance.json",
        "source-hashes.txt",
        "extension-hashes.txt",
    ] {
        actual.remove(metadata);
    }
    if actual != expected {
        return Err("prepared source drift refused".into());
    }
    Ok(())
}
/// Reuse an admitted generated tree without changing its source bytes or timestamps.
pub fn ensure_prepared(source: &Path, destination: &Path) -> Result<()> {
    disjoint(source, destination)?;
    verify(source, ORIGINAL)?;
    match fs::symlink_metadata(destination) {
        Ok(metadata) if metadata.file_type().is_dir() => verify_prepared(destination),
        Ok(_) => Err("prepared destination must be a regular directory".into()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => prepare(source, destination),
        Err(error) => Err(error.to_string()),
    }
}
/// Source is immutable. Destination must not exist: failed preparation cannot be admitted.
pub fn prepare(source: &Path, destination: &Path) -> Result<()> {
    disjoint(source, destination)?;
    verify(source, ORIGINAL)?;
    match fs::symlink_metadata(destination) {
        Ok(_) => return Err("prepared destination already exists".into()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
        Err(error) => return Err(error.to_string()),
    }
    let original = manifest(ORIGINAL)?;
    let expected = manifest(PREPARED)?;
    let patches = patch_files()?;
    if original.keys().ne(expected.keys()) || patches.keys().any(|p| !original.contains_key(p)) {
        return Err("patch inventory mismatch".into());
    }
    // Validate all transformations before creating the output tree.
    let mut files = BTreeMap::new();
    for path in original.keys() {
        let bytes = fs::read(source.join(path)).map_err(|e| e.to_string())?;
        let result = if let Some(patch) = patches.get(path) {
            apply_patch(
                std::str::from_utf8(&bytes).map_err(|e| e.to_string())?,
                patch,
            )?
            .into_bytes()
        } else {
            bytes
        };
        if expected.get(path) != Some(&sha256(&result)) {
            return Err(format!("prepared hash mismatch: {path}"));
        }
        files.insert(path, result);
    }
    verify(source, ORIGINAL)?;
    for (path, bytes) in files {
        let target = destination.join(path);
        fs::create_dir_all(target.parent().ok_or("missing parent")?).map_err(|e| e.to_string())?;
        fs::write(target, bytes).map_err(|e| e.to_string())?;
    }
    verify(destination, PREPARED)
}
pub fn write_provenance(destination: &Path, extension_root: &Path) -> Result<()> {
    verify_prepared(destination)?;
    let mut extensions = BTreeMap::new();
    inventory(extension_root, Path::new(""), &mut extensions)?;
    let mut hashes = String::new();
    for (path, hash) in &extensions {
        writeln!(hashes, "{hash}  {path}").map_err(|e| e.to_string())?;
    }
    write_if_changed(&destination.join("source-hashes.txt"), PREPARED.as_bytes())?;
    write_if_changed(&destination.join("extension-hashes.txt"), hashes.as_bytes())?;
    let json = format!(
        "{{\n  \"source_revision\": \"{REVISION}\",\n  \"upstream_manifest_sha256\": \"{}\",\n  \"prepared_manifest_sha256\": \"{}\",\n  \"patch_sha256\": \"{}\",\n  \"extension_manifest_sha256\": \"{}\",\n  \"source_original_unchanged\": true\n}}\n",
        sha256(ORIGINAL.as_bytes()),
        sha256(PREPARED.as_bytes()),
        sha256(PATCH.as_bytes()),
        sha256(hashes.as_bytes())
    );
    write_if_changed(&destination.join("provenance.json"), json.as_bytes())
}

fn validate_metadata(root: &Path) -> Result<()> {
    for name in [
        "provenance.json",
        "source-hashes.txt",
        "extension-hashes.txt",
    ] {
        let path = root.join(name);
        match fs::symlink_metadata(&path) {
            Ok(metadata) if metadata.file_type().is_file() => (),
            Ok(_) => {
                return Err(format!(
                    "prepared metadata must be regular: {}",
                    path.display()
                ));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
            Err(error) => return Err(error.to_string()),
        }
    }
    Ok(())
}
fn write_if_changed(path: &Path, bytes: &[u8]) -> Result<()> {
    output::write(path, bytes)
}
