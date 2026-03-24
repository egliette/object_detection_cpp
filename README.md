# yolov8-detect

Runs YOLOv8 object detection on a video file and writes an annotated output video.
Inference runs on GPU via ONNX Runtime CUDA EP. NMS is performed in C on the CPU.

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

## Model

Export from Ultralytics **without** baked-in NMS (default export):

```python
from ultralytics import YOLO
YOLO("yolov8n.pt").export(format="onnx")   # produces yolov8n.onnx
```

- Input shape:  `[1, 3, 640, 640]` — float32, RGB, normalized to [0, 1]
- Output shape: `[1, 84, 8400]` — 4 box coords (cx, cy, w, h) + 80 COCO class scores per anchor

> **Do not use a model exported with `nms=True`** (`[1, 300, 6]` output).
> ORT's CUDA EP cannot execute the NMS graph ops on GPU and injects 55+ CPU↔GPU
> memcpy nodes, causing a pointer-as-integer corruption in the `Expand` node.

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

- Default confidence threshold: `0.45`
- NMS IoU threshold: `0.45`
- Files in the project folder are mounted at `/app` inside the container
- Progress and FPS are printed every 100 frames
