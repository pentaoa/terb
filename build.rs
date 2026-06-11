use std::{env, fs, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=macos/SystemAudioHelper.swift");
    println!("cargo:rerun-if-changed=macos/AudioProcessHelper.swift");
    println!("cargo:rerun-if-changed=macos/HelperInfo.plist");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "macos" {
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let plist = PathBuf::from("macos/HelperInfo.plist");

    let audio_helper_path = compile_swift_helper(
        &out_dir,
        "terb-audio-helper",
        "macos/SystemAudioHelper.swift",
        &[
            "ScreenCaptureKit",
            "AVFoundation",
            "CoreGraphics",
            "CoreMedia",
        ],
        Some(&plist),
    );
    let process_helper_path = compile_swift_helper(
        &out_dir,
        "terb-audio-process-helper",
        "macos/AudioProcessHelper.swift",
        &["CoreAudio", "AppKit"],
        Some(&plist),
    );

    println!(
        "cargo:rustc-env=TERB_AUDIO_HELPER={}",
        audio_helper_path.display()
    );
    println!(
        "cargo:rustc-env=TERB_AUDIO_PROCESS_HELPER={}",
        process_helper_path.display()
    );
}

fn compile_swift_helper(
    out_dir: &PathBuf,
    name: &str,
    source: &str,
    frameworks: &[&str],
    plist: Option<&PathBuf>,
) -> PathBuf {
    let helper_path = out_dir.join(name);
    let mut command = Command::new("swiftc");
    command.arg(source).arg("-parse-as-library").arg("-O");
    for framework in frameworks {
        command.arg("-framework").arg(framework);
    }
    if let Some(plist) = plist {
        command
            .arg("-Xlinker")
            .arg("-sectcreate")
            .arg("-Xlinker")
            .arg("__TEXT")
            .arg("-Xlinker")
            .arg("__info_plist")
            .arg("-Xlinker")
            .arg(plist);
    }
    let status = command
        .arg("-o")
        .arg(&helper_path)
        .status()
        .expect("failed to run swiftc");
    if !status.success() {
        panic!("failed to compile {name}");
    }
    let _ = Command::new("codesign")
        .arg("--force")
        .arg("--sign")
        .arg("-")
        .arg(&helper_path)
        .status();

    fs::metadata(&helper_path).expect("helper was not produced");
    helper_path
}
