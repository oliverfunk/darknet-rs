# Rust bindings for darknet

[![Build Status](https://travis-ci.com/oliverfunk/darknet-rs.svg?branch=master)](https://travis-ci.com/oliverfunk/darknet-rs)

[Darknet: Convolutional Neural Networks](https://pjreddie.com/darknet/)


## todo
- rewrite the demo function used in yolo.c in rust

## Examples

Link existing files and download training weights:

```shell script
ln -s darknet-sys/darknet/data .
( mkdir trained_weights && cd trained_weights && \
  wget https://pjreddie.com/media/files/yolov3.weights )
```

Run the example

```shell script
cargo run --example yolo_detect_image
```

