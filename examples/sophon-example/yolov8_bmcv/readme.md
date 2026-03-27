export CUR_PWD=$(pwd)
../yolov8_bmcv \
    --input=/data/workspace/sophon-toolchain-rs/examples/sophon-example/yolov8_bmcv/images \
    --bmodel=/data/workspace/sophon-toolchain-rs/examples/sophon-example/coco_yolov11s_int8_1b.bmodel \
    --classnames=//data/workspace/sophon-toolchain-rs/examples/sophon-example/yolov8_bmcv/coco.names \
    --dev_id=0 \
    --conf_thresh=0.25 \
    --nms_thresh=0.45