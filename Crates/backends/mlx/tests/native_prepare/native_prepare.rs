#[path = "../../ffi/rust/build/acquisition/acquisition.rs"]
mod acquisition;
#[path = "../../ffi/rust/build/output/output.rs"]
mod output;
#[path = "../../ffi/rust/build/platform/platform.rs"]
mod platform;
#[path = "../../ffi/rust/build/preparation/preparation.rs"]
mod prepare;
#[rustfmt::skip]
use std::{
    fs,
    path::PathBuf,
};
fn scratch(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!("pcu-native-{label}-{}", std::process::id()))
}
#[test]
fn standard_sha256_vectors() {
    assert_eq!(
        prepare::sha256(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        prepare::sha256(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}
#[test]
fn exact_hunks_reproduce_and_refuse_drift() {
    let source = "one\ntwo\nthree\nfour\n";
    let patch = "@@ -1,3 +1,4 @@\n one\n-two\n+second\n+extra\n three\n";
    let expected = "one\nsecond\nextra\nthree\nfour\n";
    assert_eq!(prepare::apply_patch(source, patch).unwrap(), expected);
    assert_eq!(prepare::apply_patch(source, patch).unwrap(), expected);
    assert!(prepare::apply_patch("ONE\ntwo\nthree\nfour\n", patch).is_err());
    assert!(prepare::apply_patch(source, &patch.replace("-1,3", "-1,4")).is_err());
    assert!(prepare::apply_patch(source, &patch.replace("+1,4", "+2,4")).is_err());
}
#[test]
fn manifests_refuse_missing_extra_changed_and_unsafe_files() {
    let root = scratch("inventory");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    fs::write(root.join("a"), b"exact").unwrap();
    let manifest = format!("{}  a\n", prepare::sha256(b"exact"));
    prepare::verify(&root, &manifest).unwrap();
    fs::write(root.join("b"), b"extra").unwrap();
    assert!(prepare::verify(&root, &manifest).is_err());
    fs::remove_file(root.join("b")).unwrap();
    fs::write(root.join("a"), b"drift").unwrap();
    assert!(prepare::verify(&root, &manifest).is_err());
    fs::remove_file(root.join("a")).unwrap();
    assert!(prepare::verify(&root, &manifest).is_err());
    assert!(prepare::manifest(&format!("{}  ../escape\n", prepare::sha256(b"x"))).is_err());
    fs::remove_dir_all(root).unwrap();
}
#[test]
fn pinned_dependencies_and_archive_boundaries() {
    assert_eq!(acquisition::PINS.len(), 6);
    for pin in acquisition::PINS {
        assert!(!prepare::manifest(pin.inventory).unwrap().is_empty());
        assert_eq!(pin.archive_sha256.len(), 64);
        assert!(pin.url.starts_with("https://"));
    }
    acquisition::validate_members("root/\nroot/a\n", "drwx root\n-rw root/a\n", "root").unwrap();
    for name in ["/root/a", "root/../escape", "other/a"] {
        assert!(acquisition::validate_members(&format!("{name}\n"), "-rw a\n", "root").is_err());
    }
    assert!(acquisition::validate_members("root/a\n", "lrwx a\n", "root").is_err());
}
#[test]
#[ignore = "requires exact local pristine C source; no download or SDK needed"]
fn exact_local_source_matches_golden_and_provenance() {
    let source = std::env::var_os("PCU_MLX_TEST_C_SOURCE")
        .or_else(|| option_env!("PCU_MLX_PINNED_C_SOURCE").map(Into::into))
        .map(PathBuf::from)
        .expect("requires the automatic Apple native build or PCU_MLX_TEST_C_SOURCE");
    let destination = std::env::var_os("PCU_MLX_TEST_PREPARED_OUT")
        .map_or_else(|| scratch("exact"), PathBuf::from);
    prepare::disjoint(&source, &destination).unwrap();
    let second = scratch("second");
    let _ = fs::remove_dir_all(&destination);
    let _ = fs::remove_dir_all(&second);
    prepare::prepare(&source, &destination).unwrap();
    prepare::prepare(&source, &second).unwrap();
    let inventory = include_str!("../../ffi/native/patches/prepared.sha256");
    for path in prepare::manifest(inventory).unwrap().keys() {
        let actual = fs::read(destination.join(path)).unwrap();
        assert_eq!(actual, fs::read(second.join(path)).unwrap());
        if let Some(golden) = std::env::var_os("PCU_MLX_TEST_C_GOLDEN") {
            assert_eq!(
                actual,
                fs::read(PathBuf::from(golden).join(path)).unwrap(),
                "golden: {path}"
            );
        }
    }
    prepare::write_provenance(
        &destination,
        &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("ffi/native"),
    )
    .unwrap();
    assert!(
        fs::read_to_string(destination.join("provenance.json"))
            .unwrap()
            .contains(prepare::REVISION)
    );
    let mut timestamps = std::collections::BTreeMap::new();
    for path in prepare::manifest(inventory).unwrap().keys() {
        timestamps.insert(
            path.clone(),
            fs::metadata(destination.join(path))
                .unwrap()
                .modified()
                .unwrap(),
        );
    }
    let metadata_before = fs::metadata(destination.join("provenance.json"))
        .unwrap()
        .modified()
        .unwrap();
    prepare::ensure_prepared(&source, &destination).unwrap();
    prepare::write_provenance(
        &destination,
        &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("ffi/native"),
    )
    .unwrap();
    for (path, before) in timestamps {
        assert_eq!(
            fs::metadata(destination.join(path))
                .unwrap()
                .modified()
                .unwrap(),
            before
        );
    }
    assert_eq!(
        fs::metadata(destination.join("provenance.json"))
            .unwrap()
            .modified()
            .unwrap(),
        metadata_before
    );
    let drift = destination.join("mlx/c/array.cpp");
    fs::write(&drift, b"refuse cached drift\n").unwrap();
    assert!(prepare::ensure_prepared(&source, &destination).is_err());
    assert_eq!(fs::read(&drift).unwrap(), b"refuse cached drift\n");
    fs::remove_dir_all(second).unwrap();
}

#[test]
fn disjoint_guard_precedes_output_and_cache_refuses_drift() {
    let root = scratch("boundaries");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(root.join("source")).unwrap();
    assert!(prepare::disjoint(&root.join("source"), &root.join("source/child")).is_err());
    assert!(prepare::disjoint(&root.join("source"), &root).is_err());
    assert!(prepare::disjoint(&root.join("source"), &root.join("source")).is_err());
    prepare::disjoint(&root.join("source"), &root.join("output")).unwrap();
    let hash = prepare::sha256(b"archive");
    let inventory = format!("{}  file\n", prepare::sha256(b"exact"));
    let pin = acquisition::Pin {
        name: "test",
        directory: "root",
        url: "https://invalid.invalid",
        archive_sha256: &hash,
        inventory: &inventory,
    };
    assert!(
        acquisition::acquire_with_network(&root, &pin, false)
            .unwrap_err()
            .contains("offline native source cache missing test")
    );
    assert!(!root.join("test.download").exists());
    assert!(!root.join("test.archive").exists());
    fs::write(root.join("test.archive"), b"archive").unwrap();
    fs::create_dir_all(root.join("test-source/root")).unwrap();
    fs::write(root.join("test-source/root/file"), b"exact").unwrap();
    acquisition::acquire_with_network(&root, &pin, false).unwrap();
    fs::write(root.join("test-source/root/file"), b"drift").unwrap();
    assert!(acquisition::acquire(&root, &pin).is_err());
    fs::write(root.join("test.archive"), b"wrong").unwrap();
    assert!(acquisition::acquire(&root, &pin).is_err());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn target_host_decision_precedes_all_native_side_effects() {
    assert!(
        !platform::native_build("x86_64-unknown-linux-gnu", "x86_64-unknown-linux-gnu").unwrap()
    );
    assert!(!platform::native_build("aarch64-apple-darwin", "x86_64-unknown-linux-gnu").unwrap());
    assert!(!platform::native_build("x86_64-apple-darwin", "x86_64-apple-darwin").unwrap());
    assert!(platform::native_build("aarch64-apple-darwin", "aarch64-apple-darwin").unwrap());
    assert!(platform::native_build("x86_64-unknown-linux-gnu", "aarch64-apple-darwin").is_err());
    assert!(platform::native_build("x86_64-apple-darwin", "aarch64-apple-darwin").is_err());
}

#[cfg(unix)]
#[test]
fn cached_metadata_symlink_cannot_mutate_verified_source() {
    let root = scratch("metadata-symlink");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(root.join("mlx/c")).unwrap();
    let source = root.join("mlx/c/array.cpp");
    let original = b"original source bytes\n";
    fs::write(&source, original).unwrap();
    std::os::unix::fs::symlink("mlx/c/array.cpp", root.join("source-hashes.txt")).unwrap();
    assert!(
        prepare::verify_prepared(&root)
            .unwrap_err()
            .contains("metadata must be regular")
    );
    assert!(prepare::write_provenance(&root, &root).is_err());
    assert_eq!(fs::read(&source).unwrap(), original);
    assert_eq!(fs::read(root.join("source-hashes.txt")).unwrap(), original);
    assert!(
        fs::symlink_metadata(root.join("source-hashes.txt"))
            .unwrap()
            .file_type()
            .is_symlink()
    );
    fs::remove_dir_all(root).unwrap();
}

#[cfg(unix)]
#[test]
fn closure_links_refused_without_mutating_external_file() {
    let root = scratch("closure-symlink");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let outside = root.join("outside");
    fs::write(&outside, b"preserve").unwrap();
    let from = root.join("source");
    fs::write(&from, b"replacement").unwrap();
    let closure = root.join("closure");
    fs::create_dir(&closure).unwrap();
    let linked = closure.join("libmlx.dylib");
    std::os::unix::fs::symlink(&outside, &linked).unwrap();
    assert!(output::copy(&from, &linked).is_err());
    assert!(output::write(&linked, b"metadata").is_err());
    assert_eq!(fs::read(&outside).unwrap(), b"preserve");
    let linked_directory = root.join("linked-closure");
    std::os::unix::fs::symlink(&closure, &linked_directory).unwrap();
    assert!(output::directory(&linked_directory).is_err());
    fs::remove_dir_all(root).unwrap();
}

#[cfg(unix)]
#[test]
fn acquisition_links_refused_before_network_or_mutation() {
    let root = scratch("acquisition-symlink");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let outside = root.join("outside");
    fs::write(&outside, b"preserve").unwrap();
    let hash = prepare::sha256(b"archive");
    let inventory = format!("{}  file\n", prepare::sha256(b"exact"));
    let pin = acquisition::Pin {
        name: "test",
        directory: "root",
        url: "https://invalid.invalid",
        archive_sha256: &hash,
        inventory: &inventory,
    };
    let cache = root.join("cache");
    fs::create_dir(&cache).unwrap();
    let archive = cache.join("test.archive");
    std::os::unix::fs::symlink(&outside, &archive).unwrap();
    assert!(acquisition::acquire(&cache, &pin).is_err());
    fs::remove_file(&archive).unwrap();
    let download = cache.join("test.download");
    std::os::unix::fs::symlink(&outside, &download).unwrap();
    assert!(acquisition::acquire(&cache, &pin).is_err());
    fs::remove_file(&download).unwrap();
    fs::write(&archive, b"archive").unwrap();
    let extracted = cache.join("test-source");
    std::os::unix::fs::symlink(&root, &extracted).unwrap();
    assert!(acquisition::acquire(&cache, &pin).is_err());
    let linked_cache = root.join("linked-cache");
    std::os::unix::fs::symlink(&cache, &linked_cache).unwrap();
    assert!(acquisition::acquire(&linked_cache, &pin).is_err());
    assert_eq!(fs::read(&outside).unwrap(), b"preserve");
    fs::remove_dir_all(root).unwrap();
}
