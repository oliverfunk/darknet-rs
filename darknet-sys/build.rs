extern crate cc;
extern crate bindgen;

use std::env;
use std::path::{PathBuf, Path};
use std::process::{Command};

/// Generate Rust FFI bindings to the C library
fn bindgen_darknet() {
    let bindings = bindgen::Builder::default()
        .header("darknet/include/darknet.h")
        .generate()
        .expect("unable to generate darknet bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings.write_to_file(out_path.join("bindings.rs"))
        .expect("unable to write darknet bindings");
}

/// Build the static library
fn build_darknet() {
    let mut config = cc::Build::new();
    config.include("darknet/include/");
    config.include("darknet/src/");

    let lib_sources = include_str!("darknet_lib_sources.txt")
        .split(" ")
        .collect::<Vec<&'static str>>();
 
    for file in lib_sources {
        let file = "darknet/".to_string() + file;
        config.file(&file);
    }

    config.flag("-Wno-unused-result");
    config.flag("-Wno-unknown-pragmas");
    config.flag("-Wfatal-errors");
    config.extra_warnings(false);

    if cfg!(feature = "opencv") {
        config.define("OPENCV", Some("1"));
        config.file("darknet/src/image_opencv.cpp");
        config.cpp_link_stdlib("stdc++");        
    }
    if cfg!(feature = "gpu") {
        config.define("GPU", Some("1"));
        config.include("/usr/local/cuda/include/");
        let lib_gpu_sources = include_str!("darknet_lib_gpu_sources.txt")
            .split(" ")
            .collect::<Vec<&'static str>>();
        for file in lib_gpu_sources {
            let file = "darknet/".to_string() + file;
            config.file(&file);
        }
        config.cpp_link_stdlib("stdc++");
        config.cuda(true);
    }
    if cfg!(feature = "cudnn") {
        config.define("CUDNN", Some("1"));
        config.cuda(true);
    }

    if cfg!(feature = "static") {
        config.static_flag(true);
    }
    config.compile("libdarknet.a");
}

fn try_to_find_and_link_lib() -> bool {
    if let Ok(lib_dir) = env::var("DARKNET_LIB_DIR") {
        println!("cargo:rustc-link-search=native={}", lib_dir);
        let mode = match env::var_os("DARKNET_STATIC") {
            Some(_) => "static",
            None => "dylib",
        };
        println!("cargo:rustc-link-lib={}=darknet", mode);
        return true;
    }
    false
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=darknet/");

    if !Path::new("darknet/.git").exists() {
        let _ = Command::new("git").args(&["submodule", "update", "--init"]).status();
    }

    // Generate Rust bindings to the C library
    bindgen_darknet();

    // Check if library is available and can be linked
    if !try_to_find_and_link_lib() {
        // Build the static library
        build_darknet();
    }
}
