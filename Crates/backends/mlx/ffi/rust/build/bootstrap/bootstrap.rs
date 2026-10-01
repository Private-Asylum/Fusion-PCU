#[rustfmt::skip]
use std::{
    fmt::Write as _,
    env,
    fs,
    path::{
        Path,
        PathBuf,
    },
    process::Command,
};

#[rustfmt::skip]
use crate::{
    acquisition,
    output,
    platform,
    prepare,
};
fn run(command: &mut Command) -> Result<(), String> {
    let status = command
        .status()
        .map_err(|e| format!("native build command failed: {e}"))?;
    if !status.success() {
        return Err(format!("native build command returned {status}"));
    }
    Ok(())
}
fn cmake_configure(cmake: &std::ffi::OsStr, source: &Path, build: &Path) -> Command {
    let mut command = Command::new(cmake);
    command.arg("-S").arg(source).arg("-B").arg(build).args([
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_OSX_DEPLOYMENT_TARGET=26.2",
        "-DBUILD_SHARED_LIBS=ON",
        "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
    ]);
    command
}
pub fn build() -> Result<(), String> {
    for name in [
        "PCU_MLX_CMAKE",
        "PCU_MLX_NATIVE_JOBS",
        "CARGO_NET_OFFLINE",
        "DEVELOPER_DIR",
        "SDKROOT",
        "MACOSX_DEPLOYMENT_TARGET",
        "HOST",
        "TARGET",
        "PATH",
        "CC",
        "CXX",
        "CFLAGS",
        "CXXFLAGS",
        "LDFLAGS",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    for path in ["ffi/rust/build", "ffi/native", "ffi/CMakeLists.txt"] {
        println!("cargo:rerun-if-changed={path}");
    }
    let host = env::var("HOST").map_err(|e| e.to_string())?;
    let target = env::var("TARGET").map_err(|e| e.to_string())?;
    if !platform::native_build(&host, &target)? {
        return Ok(());
    }
    capture(Command::new("xcrun").args(["metal", "--version"]))
        .map_err(|error| format!("Apple Metal compiler prerequisite unavailable; install the official Metal Toolchain with xcodebuild -downloadComponent MetalToolchain before building MLX. {error}"))?;
    let cmake = env::var_os("PCU_MLX_CMAKE").unwrap_or_else(|| "cmake".into());
    capture(Command::new(&cmake).arg("--version"))
        .map_err(|error| format!("CMake build prerequisite unavailable: {error}"))?;
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").ok_or("missing manifest dir")?);
    let out =
        PathBuf::from(env::var_os("OUT_DIR").ok_or("missing OUT_DIR")?).join("pcu-mlx-native");
    output::directory(&out)?;
    let mut sources = std::collections::BTreeMap::new();
    for pin in acquisition::PINS {
        sources.insert(pin.name, acquisition::acquire(&out.join("cache"), pin)?);
    }
    let staged = out.join("prepared-c");
    prepare::ensure_prepared(&sources["mlx-c"], &staged)?;
    prepare::write_provenance(&staged, &root.join("ffi/native"))?;
    let jobs = env::var("PCU_MLX_NATIVE_JOBS").unwrap_or_else(|_| "2".into());
    if !matches!(jobs.parse::<usize>(), Ok(1..=64)) {
        return Err("PCU_MLX_NATIVE_JOBS must be 1..64".into());
    }
    let prefix = build_mlx(&cmake, &jobs, &out, &sources)?;
    let c_build = build_c(&cmake, &jobs, &out, &root, &prefix, &staged)?;
    let library = stage_closure(&out, &prefix, &c_build, &staged)?;
    write_build_provenance(&out.join("closure"), &cmake, &host, &target)?;
    for (name, path) in [
        ("mlx-cmake-cache.txt", out.join("mlx-build/CMakeCache.txt")),
        (
            "mlx-compile-commands.json",
            out.join("mlx-build/compile_commands.json"),
        ),
        ("c-cmake-cache.txt", out.join("c-build/CMakeCache.txt")),
    ] {
        output::copy(&path, &out.join("closure").join(name))?;
    }
    for pin in acquisition::PINS {
        prepare::verify(&sources[pin.name], pin.inventory)?;
    }
    println!(
        "cargo:rustc-env=PCU_MLX_DEFAULT_LIBRARY={}",
        library.display()
    );
    println!(
        "cargo:rustc-env=PCU_MLX_PINNED_C_SOURCE={}",
        sources["mlx-c"].display()
    );
    Ok(())
}
fn build_mlx(
    cmake: &std::ffi::OsStr,
    jobs: &str,
    out: &Path,
    sources: &std::collections::BTreeMap<&str, PathBuf>,
) -> Result<PathBuf, String> {
    let native_build = out.join("mlx-build");
    let prefix = out.join("mlx-prefix");
    output::directory(&native_build)?;
    output::directory(&prefix)?;
    let mut configure = cmake_configure(cmake, &sources["mlx"], &native_build);
    configure
        .arg(format!("-DCMAKE_INSTALL_PREFIX={}", prefix.display()))
        .args([
            "-DMLX_BUILD_TESTS=OFF",
            "-DMLX_BUILD_EXAMPLES=OFF",
            "-DMLX_BUILD_BENCHMARKS=OFF",
            "-DMLX_BUILD_PYTHON_BINDINGS=OFF",
            "-DMLX_BUILD_PYTHON_STUBS=OFF",
            "-DMLX_BUILD_METAL=ON",
            "-DMLX_METAL_JIT=OFF",
            "-DMLX_BUILD_CPU=ON",
            "-DMLX_BUILD_CUDA=OFF",
            "-DMLX_USE_CCACHE=OFF",
            "-DUSE_SYSTEM_FMT=OFF",
        ]);
    for (name, key) in [
        ("metal", "METAL_CPP"),
        ("json", "JSON"),
        ("fmt", "FMT"),
        ("gguf", "GGUFLIB"),
    ] {
        configure.arg(format!(
            "-DFETCHCONTENT_SOURCE_DIR_{key}={}",
            sources[name].display()
        ));
    }
    run(&mut configure)?;
    run(Command::new(cmake)
        .arg("--build")
        .arg(&native_build)
        .arg("--parallel")
        .arg(jobs))?;
    run(Command::new(cmake).arg("--install").arg(&native_build))?;
    Ok(prefix)
}
fn build_c(
    cmake: &std::ffi::OsStr,
    jobs: &str,
    out: &Path,
    root: &Path,
    prefix: &Path,
    staged: &Path,
) -> Result<PathBuf, String> {
    let c_build = out.join("c-build");
    output::directory(&c_build)?;
    run(cmake_configure(cmake, &root.join("ffi"), &c_build)
        .arg(format!("-DCMAKE_PREFIX_PATH={}", prefix.display()))
        .arg(format!("-DMLX_DIR={}/share/cmake/MLX", prefix.display()))
        .args([
            "-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF",
            "-DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF",
        ])
        .arg(format!("-DPCU_MLX_C_PREPARED_SOURCE={}", staged.display())))?;
    run(Command::new(cmake)
        .arg("--build")
        .arg(&c_build)
        .arg("--parallel")
        .arg(jobs))?;
    Ok(c_build)
}
fn stage_closure(
    out: &Path,
    prefix: &Path,
    c_build: &Path,
    staged: &Path,
) -> Result<PathBuf, String> {
    // Closure contains only these source-built libraries and actual installed shaders.
    let closure = out.join("closure");
    output::directory(&closure)?;
    for name in [
        "libmlx.dylib",
        "libjaccl.dylib",
        "mlx.metallib",
        "libpcu_mlx_c_direct.dylib",
        "artifact-hashes.txt",
        "provenance.json",
        "build-provenance.txt",
        "mlx-cmake-cache.txt",
        "mlx-compile-commands.json",
        "c-cmake-cache.txt",
    ] {
        output::regular(&closure.join(name))?;
    }
    for name in ["libmlx.dylib", "libjaccl.dylib", "mlx.metallib"] {
        let from = prefix.join("lib").join(name);
        if !from.is_file() {
            return Err(format!("source-built closure missing {}", from.display()));
        }
        output::copy(&from, &closure.join(name))?;
    }
    let library = closure.join("libpcu_mlx_c_direct.dylib");
    output::copy(
        &c_build.join("native/upstream/libpcu_mlx_c_direct.dylib"),
        &library,
    )?;
    for name in [
        "libmlx.dylib",
        "libjaccl.dylib",
        "libpcu_mlx_c_direct.dylib",
    ] {
        relocate(&closure.join(name), name)?;
    }
    let mut hashes = String::new();
    for name in [
        "libmlx.dylib",
        "libjaccl.dylib",
        "mlx.metallib",
        "libpcu_mlx_c_direct.dylib",
    ] {
        writeln!(
            hashes,
            "{}  {name}",
            prepare::sha256(&fs::read(closure.join(name)).map_err(|e| e.to_string())?)
        )
        .map_err(|e| e.to_string())?;
    }
    output::write(&closure.join("artifact-hashes.txt"), hashes.as_bytes())?;
    output::copy(
        &staged.join("provenance.json"),
        &closure.join("provenance.json"),
    )?;
    Ok(library)
}
fn capture(command: &mut Command) -> Result<String, String> {
    let output = command.output().map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).into_owned());
    }
    String::from_utf8(output.stdout).map_err(|e| e.to_string())
}
fn relocate(file: &Path, name: &str) -> Result<(), String> {
    let dependencies = capture(Command::new("otool").arg("-L").arg(file))?;
    for line in dependencies.lines().skip(2) {
        let dependency = line
            .split_whitespace()
            .next()
            .ok_or("empty dylib dependency")?;
        let leaf = Path::new(dependency)
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or("invalid dylib dependency")?;
        if ["libmlx.dylib", "libjaccl.dylib"].contains(&leaf) {
            run(Command::new("install_name_tool")
                .arg("-change")
                .arg(dependency)
                .arg(format!("@loader_path/{leaf}"))
                .arg(file))?;
        } else if !dependency.starts_with("/usr/lib/")
            && !dependency.starts_with("/System/Library/")
        {
            return Err(format!("unowned native dependency refused: {dependency}"));
        }
    }
    run(Command::new("install_name_tool")
        .arg("-id")
        .arg(format!("@rpath/{name}"))
        .arg(file))?;
    let load = capture(Command::new("otool").arg("-l").arg(file))?;
    let mut rpath_command = false;
    for line in load.lines() {
        let line = line.trim();
        if line.starts_with("cmd ") {
            rpath_command = line == "cmd LC_RPATH";
        }
        if rpath_command && line.starts_with("path ") {
            let path = line
                .strip_prefix("path ")
                .and_then(|p| p.split_once(" (offset"))
                .map(|(p, _)| p)
                .ok_or("invalid native rpath")?;
            if path != "@loader_path" {
                run(Command::new("install_name_tool")
                    .arg("-delete_rpath")
                    .arg(path)
                    .arg(file))?;
            }
            rpath_command = false;
        }
    }
    if !load.contains("path @loader_path (offset") {
        run(Command::new("install_name_tool")
            .arg("-add_rpath")
            .arg("@loader_path")
            .arg(file))?;
    }
    run(Command::new("codesign")
        .args(["--force", "--sign", "-"])
        .arg(file))
}

fn write_build_provenance(
    closure: &Path,
    cmake: &std::ffi::OsStr,
    host: &str,
    target: &str,
) -> Result<(), String> {
    let mut text = format!(
        "host={host}\ntarget={target}\nmlx_revision=64ea011cb65f14d9ce2737e60db9a4ae91ed7441\nmlx_c_revision={}\n",
        prepare::REVISION
    );
    for pin in acquisition::PINS {
        writeln!(
            text,
            "source={} url={} archive_sha256={} inventory_sha256={}",
            pin.name,
            pin.url,
            pin.archive_sha256,
            prepare::sha256(pin.inventory.as_bytes())
        )
        .map_err(|e| e.to_string())?;
    }
    for (label, command) in [
        ("cmake", Command::new(cmake).arg("--version")),
        ("xcode", Command::new("xcodebuild").arg("-version")),
        (
            "sdk",
            Command::new("xcrun").args(["--sdk", "macosx", "--show-sdk-version"]),
        ),
        ("metal", Command::new("xcrun").args(["metal", "--version"])),
    ] {
        writeln!(text, "{label}: {}", capture(command)?.trim()).map_err(|e| e.to_string())?;
    }
    output::write(&closure.join("build-provenance.txt"), text.as_bytes())
}
