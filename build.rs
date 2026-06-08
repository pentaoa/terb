use std::{env, fs, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=macos/SystemAudioHelper.swift");
    println!("cargo:rerun-if-changed=macos/HelperInfo.plist");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "macos" {
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let helper_path = out_dir.join("terb-audio-helper");
    let source = PathBuf::from("macos/SystemAudioHelper.swift");
    let plist = PathBuf::from("macos/HelperInfo.plist");

    let status = Command::new("swiftc")
        .arg(source)
        .arg("-parse-as-library")
        .arg("-O")
        .arg("-framework")
        .arg("ScreenCaptureKit")
        .arg("-framework")
        .arg("AVFoundation")
        .arg("-framework")
        .arg("CoreGraphics")
        .arg("-framework")
        .arg("CoreMedia")
        .arg("-Xlinker")
        .arg("-sectcreate")
        .arg("-Xlinker")
        .arg("__TEXT")
        .arg("-Xlinker")
        .arg("__info_plist")
        .arg("-Xlinker")
        .arg(plist)
        .arg("-o")
        .arg(&helper_path)
        .status()
        .expect("failed to run swiftc");

    if !status.success() {
        panic!("failed to compile macOS audio helper");
    }

    let _ = Command::new("codesign")
        .arg("--force")
        .arg("--sign")
        .arg("-")
        .arg(&helper_path)
        .status();

    fs::metadata(&helper_path).expect("audio helper was not produced");
    println!(
        "cargo:rustc-env=TERB_AUDIO_HELPER={}",
        helper_path.display()
    );
}
