//! Explicit selection must not touch an unrelated provider's compiler/runtime probes.

#[rustfmt::skip]
use std::{
    fs,
    os::unix::fs::PermissionsExt,
    process::Command,
};

const CHILD: &str = "PCU_PARITY_UNSELECTED_SDK_CHILD";
const MARKER: &str = "PCU_PARITY_UNSELECTED_SDK_MARKER";

pub fn verify() {
    if std::env::var_os(CHILD).is_some() {
        super::binary::verify(fusion_pcu::global::PcuBackendChoice::Cpu);
        super::integer::verify(fusion_pcu::global::PcuBackendChoice::Cpu);
        return;
    }

    // Environment overrides are child-local: concurrently running tests and the
    // engineer's SDK configuration are unaffected. A native probe must leave a
    // marker even when discovery swallows its nonzero status.
    let directory =
        std::env::temp_dir().join(format!("pcu-unselected-sdk-probe-{}", std::process::id()));
    fs::create_dir(&directory).unwrap();
    let executable = directory.join("probe.sh");
    let marker = directory.join("calls");
    fs::write(
        &executable,
        "#!/bin/sh\nprintf x >> \"$PCU_PARITY_UNSELECTED_SDK_MARKER\"\nexit 1\n",
    )
    .unwrap();
    fs::set_permissions(&executable, fs::Permissions::from_mode(0o700)).unwrap();
    let child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "cpu_explicit_selection_does_not_probe_other_sdks",
            "--nocapture",
        ])
        .env(CHILD, "1")
        .env(MARKER, &marker)
        .env("HIPCC", &executable)
        .env("NVCC", &executable)
        .env("CUDA_NVCC", &executable)
        .output();
    let probed = marker.try_exists().unwrap();
    fs::remove_dir_all(&directory).unwrap();
    let child = child.unwrap();
    assert!(
        child.status.success(),
        "CPU child failed:\n{}\n{}",
        String::from_utf8_lossy(&child.stdout),
        String::from_utf8_lossy(&child.stderr)
    );
    assert!(
        !probed,
        "explicit CPU source selection invoked an unselected SDK compiler"
    );
}
