示例：使用 Rust 封装的 Sophon Runtime 跑 YOLOv11/YOLOv8 bmodel（含预处理 + 后处理 + 结果落盘）。

目录推理（与官方 yolov8_bmcv demo 对齐）：

```bash
cargo run -- \
  --bmodel ./coco_yolov11s_int8_1b.bmodel \
  --input ./yolov8_bmcv/images \
  --classnames ./yolov8_bmcv/coco.names \
  --devid 0 \
  --conf_thresh 0.25 \
  --nms_thresh 0.45
```

如果 Sophon SDK 在 /opt/sophon 下但未加入系统链接器路径，可显式指定：

```bash
cargo run -- \
  --libdir /opt/sophon/libsophon-current/lib \
  --bmodel ./coco_yolov11s_int8_1b.bmodel \
  --input ./yolov8_bmcv/images \
  --classnames ./yolov8_bmcv/coco.names
```

输出：
- results/images 下保存可视化图片（画框）
- results/<bmodel>_<dataset>_rust_result.json 保存检测结果 JSON
