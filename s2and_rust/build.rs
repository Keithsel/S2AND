#[cfg(windows)]
fn main() {
    use std::env;
    use std::fs;
    use std::path::{Path, PathBuf};

    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
    println!("cargo:rerun-if-env-changed=PYTHON_SYS_EXECUTABLE");
    println!("cargo:rerun-if-env-changed=VIRTUAL_ENV");

    let Some(python_exe) = python_executable() else {
        println!(
            "cargo:warning=Unable to locate Python executable for Windows cargo-test DLL staging"
        );
        return;
    };
    rerun_if_changed(&python_exe);
    if let Ok(Some(python_home)) = python_home(&python_exe) {
        println!(
            "cargo:rustc-env=S2AND_RUST_PYTHONHOME={}",
            python_home.display()
        );
    }
    let Ok(runtime_dlls) = python_runtime_dlls(&python_exe) else {
        println!(
            "cargo:warning=Unable to query Python runtime DLLs from {}",
            python_exe.display()
        );
        return;
    };
    if runtime_dlls.is_empty() {
        println!(
            "cargo:warning=Python runtime DLL query returned no files for {}",
            python_exe.display()
        );
        return;
    }

    let Ok(out_dir) = env::var("OUT_DIR") else {
        println!("cargo:warning=OUT_DIR is unavailable; cannot stage Python runtime DLLs");
        return;
    };
    let Some(profile_dir) = PathBuf::from(out_dir)
        .ancestors()
        .nth(3)
        .map(Path::to_path_buf)
    else {
        println!("cargo:warning=Unable to resolve Cargo profile directory from OUT_DIR");
        return;
    };
    let deps_dir = profile_dir.join("deps");
    if let Err(err) = fs::create_dir_all(&deps_dir) {
        println!(
            "cargo:warning=Unable to create Cargo deps directory {}: {err}",
            deps_dir.display()
        );
        return;
    }

    for source in runtime_dlls {
        rerun_if_changed(&source);
        let Some(file_name) = source.file_name() else {
            continue;
        };
        let destination = deps_dir.join(file_name);
        if let Err(err) = fs::copy(&source, &destination) {
            println!(
                "cargo:warning=Unable to stage Python runtime DLL {} to {}: {err}",
                source.display(),
                destination.display()
            );
        }
    }
}

#[cfg(not(windows))]
fn main() {}

#[cfg(windows)]
fn rerun_if_changed(path: &std::path::Path) {
    let normalized = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    println!("cargo:rerun-if-changed={}", normalized.display());
}

#[cfg(windows)]
fn python_executable() -> Option<std::path::PathBuf> {
    use std::env;
    use std::path::PathBuf;

    for key in ["PYO3_PYTHON", "PYTHON_SYS_EXECUTABLE"] {
        if let Ok(value) = env::var(key) {
            let candidate = PathBuf::from(value);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }

    if let Ok(virtual_env) = env::var("VIRTUAL_ENV") {
        let candidate = PathBuf::from(virtual_env)
            .join("Scripts")
            .join("python.exe");
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    let repo_venv_python = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|repo_root| repo_root.join(".venv").join("Scripts").join("python.exe"));
    if let Some(candidate) = repo_venv_python {
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    None
}

#[cfg(windows)]
fn python_home(python_exe: &std::path::Path) -> std::io::Result<Option<std::path::PathBuf>> {
    use std::path::PathBuf;
    use std::process::Command;

    let output = Command::new(python_exe)
        .arg("-c")
        .arg("import sys; print(sys.base_prefix)")
        .output()?;
    if !output.status.success() {
        return Ok(None);
    }
    let value = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if value.is_empty() {
        return Ok(None);
    }
    let path = PathBuf::from(value);
    Ok(path.is_dir().then_some(path))
}

#[cfg(windows)]
fn python_runtime_dlls(python_exe: &std::path::Path) -> std::io::Result<Vec<std::path::PathBuf>> {
    use std::path::PathBuf;
    use std::process::Command;

    let script = r#"
import pathlib
import sys

root = pathlib.Path(sys.base_prefix)
names = [
    f"python{sys.version_info.major}{sys.version_info.minor}.dll",
    "python3.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
]
print("\n".join(str(root / name) for name in names if (root / name).exists()))
"#;
    let output = Command::new(python_exe).arg("-c").arg(script).output()?;
    if !output.status.success() {
        return Ok(Vec::new());
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(PathBuf::from)
        .filter(|path| path.is_file())
        .collect())
}
