extern crate darknet;

const metadata_file_path: &'static str = "darknet-sys/darknet/cfg/coco.data";
const config_path: &'static str = "darknet-sys/darknet/cfg/yolov3.cfg";
const weight_file_path: &'static str = "trained_weights/yolov3.weights";
const voc_names: Vec<&'static str> = vec![
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
];
const classes: u32 = 20;
const thresh: f32 = 0.001;
const hier: f32 = 0.5;
const nms:f32 = 0.45;

fn main() {
    let alphabet = darknet::load_alphabet();
    let mut net = darknet::load_network(config_path, Ok(weight_file_path), false).unwrap();
    let metadata = darknet::load_metadata(metadata_file_path).unwrap();

    let mut im = darknet::load_image_color("darknet-sys/darknet/data/horses.jpg", 0, 0).unwrap();
    im = darknet::resize_image(im, im.0.w, im.0.h);

    darknet::predict_image(&mut net, im);

    let num: &mut i32 = &mut 0;
    let mut detections = darknet::get_network_boxes(net, im.0.w, im.0.h,thresh, hier, &mut 0, 0, num);

    // non-max suppression
    darknet::do_nms_obj(&mut detections, *num, metadata.0.classes, nms);

    darknet::draw_detections(im, detections, *num, thresh, )
}
