//! Rust wrapper for darknet convolutional neural networks.

extern crate darknet_sys as ffi;
extern crate libc;

mod errors;

use errors::Error;
use std::ffi::CString;
use std::path::Path;
use std::ptr;

pub struct Metadata(pub ffi::metadata);

/// Load metadata from a file
pub fn load_metadata<P: AsRef<Path>>(metadata_file_path: P) -> Result<Metadata, Error> {
    let metadata_file_path_c =
        CString::new(metadata_file_path.as_ref().to_string_lossy().as_bytes()).map_err(|_| {
            Error::new(
                "Failed to convert config file path to CString when loading metadata.".to_owned(),
            )
        })?;

    let md = unsafe { ffi::get_metadata(metadata_file_path_c.as_ptr() as *mut _) };

    Ok(Metadata(md))
}

/// A wrapper for a darknet network
pub struct Network(*mut ffi::network);

/// Free the underlying network when it goes out of scope
impl Drop for Network {
    fn drop(&mut self) {
        unsafe { ffi::free_network(self.0) }
    }
}

pub fn set_batch_network(network: &mut Network, batch: i32) {
    unsafe { ffi::set_batch_network(network.0, batch) }
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

    let network = match weights_path {
        Some(w_path) => {
            let weights_path_c = CString::new(w_path.as_ref().to_string_lossy().as_bytes())
                .map_err(|_| {
                    Error::new(
                        "Failed to convert weight file path to CString when loading network."
                            .to_string(),
                    )
                })?;

            unsafe {
                ffi::load_network(
                    config_path_c.as_ptr() as *mut _,
                    weights_path_c.as_ptr() as *mut _,
                    clear as i32,
                )
            }
        }
        None => unsafe {
            ffi::load_network(
                config_path_c.as_ptr() as *mut _,
                ptr::null_mut() as *mut _,
                clear as i32,
            )
        },
    };

    Ok(Network(network))
}

//pub fn load

pub fn predict_image(network: &mut Network, image: &Image) {
    unsafe { ffi::network_predict_image(network.0, image.0) };
}

pub fn forward_network(network: &mut Network) {
    unsafe { ffi::forward_network(network.0) }
}

pub fn backward_network(network: &mut Network) {
    unsafe { ffi::backward_network(network.0) }
}

pub fn update_network(network: &mut Network) {
    unsafe { ffi::update_network(network.0) }
}

/// A wrapper for a darknet image
pub struct Image(pub ffi::image);

impl Drop for Image {
    fn drop(&mut self) {
        unsafe { ffi::free_image(self.0) }
    }
}

pub fn load_image_color<P: AsRef<Path>>(
    image_filepath: P,
    width: i32,
    height: i32,
) -> Result<Image, Error> {
    let image_filepath_c = CString::new(image_filepath.as_ref().to_string_lossy().as_bytes())
        .map_err(|_| Error::new("Error converting image_filepath into a CString".to_string()))?;

    let image =
        unsafe { ffi::load_image_color(image_filepath_c.as_ptr() as *mut _, width, height) };

    Ok(Image(image))
}

pub fn resize_image(im: &Image, w: i32, h: i32) -> Image {
    let resize_image = unsafe { ffi::resize_image(im.0, w, h) };

    Image(resize_image)
}

pub fn save_image(im: &Image, image_file_name: &str) {
    let image_fn_c = CString::new(image_file_name.to_string().as_bytes())
        .map_err(|_| Error::new("Error converting image name into a CString".to_string()))
        .unwrap();

    unsafe { ffi::save_image(im.0, image_fn_c.as_ptr()) }
}

pub struct Detection(*mut ffi::detection);

pub fn do_nms_obj(dets: &mut Detection, total: i32, classes: i32, thresh: f32) {
    unsafe { ffi::do_nms_obj(dets.0, total, classes, thresh) }
}

pub fn get_network_boxes(
    net: Network,
    w: i32,
    h: i32,
    thresh: f32,
    hier: f32,
    map: &mut i32,
    relative: i32,
    num: &mut i32,
) -> Detection {
    let m = map as *mut i32;
    let n = num as *mut i32;

    let det = unsafe { ffi::get_network_boxes(net.0, w, h, thresh, hier, m, relative, n) };

    Detection(det)
}

pub struct Alphabet(*mut *mut ffi::image);

pub fn load_alphabet() -> Alphabet {
    Alphabet(unsafe { ffi::load_alphabet() })
}

pub struct Names(*mut *mut ::std::os::raw::c_char);

pub fn load_names(names_arr: Vec<&str>) -> Result<Names, Error> {
    let mut names_c: Vec<*mut u8> = Vec::new();

    for s in names_arr {
        let name_c = CString::new(s.to_string().as_bytes())
            .map_err(|_| Error::new("Error converting name into a CString".to_string()))?
            .as_ptr();

        names_c.push(name_c as *mut _);
    }

    Ok(Names(names_c.as_ptr() as *mut _))
}

pub fn draw_detections(
    img: &Image,
    dets: Detection,
    num: i32,
    thresh: f32,
    names: Names,
    alphabet: Alphabet,
    classes: i32,
) {
    unsafe { ffi::draw_detections(img.0, dets.0, num, thresh, names.0, alphabet.0, classes) }
}

pub fn free_detections(dets: &mut Detection, n: i32) {
    unsafe { ffi::free_detections(dets.0, n) }
}
