//! Rust wrapper for darknet convolutional neural networks.

extern crate darknet_sys as ffi;
extern crate libc;

mod errors;

use crate::errors::Error;
use std::ffi::CString;
use std::path::Path;
use std::ptr;

/// A wrapper for a darknet network
pub struct Network(*mut ffi::network);

// Free the underlying network when it goes out of scope
impl Drop for Network {
    fn drop(&mut self) {
        unsafe { ffi::free_network(self.0) }
    }
}

/// Load the network from a configuration file
pub fn load_network<P: AsRef<Path>>(
    config_path: P,
    weights_path: Option<P>,
    clear: bool,
) -> Result<Network, Error> {
    let config_path_c =
        CString::new(config_path.as_ref().to_string_lossy().as_bytes()).map_err(|_| {
            Error::new(
                "Failed to convert config file path to CString when loading network.".to_owned(),
            )
        })?;

    let weights_path_c = match weights_path {
        Some(w_path) => CString::new(w_path.as_ref().to_string_lossy().as_bytes())
            .map_err(|_| {
                Error::new(
                    "Failed to convert weight file path to CString when loading network."
                        .to_string(),
                )
            })?.as_ptr(),
        None => ptr::null_mut(),
    };

    let network = unsafe {
        ffi::load_network(
            config_path_c.as_ptr() as *mut _,
            weights_path_c as *mut _,
            clear as i32,
        )
    };
    Ok(Network(network))
}

pub fn forward_network(network: &Network) {
    unsafe { ffi::forward_network(network.0) }
}

pub fn backward_network(network: &Network) {
    unsafe { ffi::backward_network(network.0) }
}

pub fn update_network(network: &Network) {
    unsafe { ffi::update_network(network.0) }
}
