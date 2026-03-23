# yolov8-detect

Runs YOLOv8 object detection on a video file and writes an annotated output video.

## Build

```bash
docker compose build
docker compose run --rm gstreamer_custom bash
```

Then inside the container, compile and install:

```bash
cd /app
make
cp bin/detect /usr/local/bin/detect   # run from any directory
```

## Usage

```bash
bin/detect <input.mp4> <model.onnx> <output.mp4> [confidence]
```

```bash
# examples
bin/detect videos/street.mp4 models/yolov8n.onnx videos/out.mp4
bin/detect videos/street.mp4 models/yolov8n.onnx videos/out.mp4 0.5
```

## Notes

- Default confidence threshold is `0.45`
- Files in the project folder are mounted at `/app` inside the container
- Model input shape: `[1, 3, 640, 640]` — float32, RGB, normalized to [0, 1]
- Model output shape: `[1, 300, 6]` — 300 detections, each `[x1, y1, x2, y2, score, class_id]`
- Model must have NMS built-in (default ultralytics ONNX export)