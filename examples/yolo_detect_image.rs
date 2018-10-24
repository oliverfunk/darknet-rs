extern crate darknet;

const CONFIG_PATH: &'static str = "darknet-sys/darknet/cfg/yolov3-voc.cfg";
const WEIGHT_FILE_PATH: &'static str = "trained_weights/yolov3.weights";

const CLASSES: i32 = 20;
const THRESH: f32 = 0.001;
const HIER: f32 = 0.5;
const NMS: f32 = 0.45;

fn main() {
    let voc_names: Vec<&'static str> = vec![
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

    println!("HERE 1");
    let alphabet = darknet::load_alphabet();
    let mut net =
        darknet::load_network(CONFIG_PATH, Option::Some(WEIGHT_FILE_PATH), false).unwrap();

    println!("HERE 2");
    let mut im = darknet::load_image_color("darknet-sys/darknet/data/horses.jpg", 0, 0).unwrap();
    im = darknet::resize_image(&im, im.0.w, im.0.h);

    println!("HERE 3");
    darknet::predict_image(&mut net, &im);

    println!("HERE 4");
    let num: &mut i32 = &mut 0;
    let mut detections =
        darknet::get_network_boxes(net, im.0.w, im.0.h, THRESH, HIER, &mut 0, 0, num);

    println!("HERE 5");
    // non-max suppression
    darknet::do_nms_obj(&mut detections, *num, CLASSES, NMS);

    println!("HERE 6");
    darknet::draw_detections(
        &im,
        detections,
        *num,
        THRESH,
        darknet::load_names(voc_names).unwrap(),
        alphabet,
        CLASSES,
    );
    darknet::save_image(&im, "detection");
}
