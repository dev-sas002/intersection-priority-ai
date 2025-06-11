# `/predict`: captured session

Real requests against the Docker Compose stack (`docker compose up --build`,
host port 8250), captured on 2026-11-07. The image was built from a clean
copy of the repository, so no ambulance weights were present and
`ITCS_DETECTOR=auto` chose the public COCO checkpoint baked in at build time.
That is why each response carries a `notice` and the label is `"bus"`.

Compose sets `ITCS_MAX_FRAMES=60`, `ITCS_FRAME_STRIDE=3` and
`ITCS_BATCH_SIZE=8`. The `frame` field (a base64 JPEG) is replaced by its
length; nothing else is edited.

## Health

```console
$ curl -s localhost:8250/health
{
  "status": "ok",
  "version": "1.0.0",
  "environment": "production",
  "detector": {
    "name": "coco",
    "performs_real_detection": true,
    "identifies_ambulances": false,
    "description": "Public Ultralytics yolov8n.pt (COCO). COCO has no ambulance class; truck, bus are treated as candidate emergency vehicles."
  },
  "inference_slots_free": 8
}
```

## A clip with an emergency vehicle

`samples/ambulance.mp4` is three seconds of an ambulance driving ahead of the
camera. The first batch of eight sampled frames (22 decoded frames at stride
3) already holds a `bus` above the 0.75 confidence threshold, so the scan stops
there.

```console
$ curl -s -F "file=@samples/ambulance.mp4" localhost:8250/predict
{
  "ambulance": true,
  "frame": "<74652 base64 chars>",
  "confidence": 0.8124947547912598,
  "label": "bus",
  "frames_read": 22,
  "frames_inferred": 8,
  "detector": "coco",
  "notice": "The active model has no ambulance class: it reports generic vehicle classes and cannot confirm that the vehicle is an ambulance. Load the purpose-trained weights (ITCS_WEIGHTS_PATH) for a real answer.",
  "priority": null
}
```

The response body was 75,010 bytes and took 1.4 s inside the Docker Desktop
VM on an Apple Silicon Mac, model already loaded.

## A clip without one

`samples/highway-suv.mp4` shows an SUV and a pickup on a highway. All 72
frames are read, every third goes through the model, and nothing qualifies.
The returned frame is the first sampled frame with a "No Ambulance Detected"
overlay.

```console
$ curl -s -F "file=@samples/highway-suv.mp4" localhost:8250/predict
{
  "ambulance": false,
  "frame": "<54104 base64 chars>",
  "confidence": 0.0,
  "label": null,
  "frames_read": 72,
  "frames_inferred": 24,
  "detector": "coco",
  "notice": "The active model has no ambulance class: it reports generic vehicle classes and cannot confirm that the vehicle is an ambulance. Load the purpose-trained weights (ITCS_WEIGHTS_PATH) for a real answer.",
  "priority": null
}
```

## Something that is not a video

```console
$ curl -s -w "\nHTTP %{http_code}\n" -F "file=@LICENSE" localhost:8250/predict
{"detail":"Uploaded file could not be read as a video."}
HTTP 400
```

## Misconfiguration stops the process at startup

`ITCS_DETECTOR=yolo` with no `best.pt` on disk (exit code 3, traceback
trimmed):

```console
$ docker run --rm -e ITCS_DETECTOR=yolo p2-intelligent-traffic-control-system:api
app.errors.DetectorUnavailableError: Ambulance weights not found at /app/weights/best.pt.
These weights are trained per-deployment and are deliberately NOT committed to this repository.
Options:
  1. Train or obtain a single-class YOLOv8 ambulance checkpoint and place it at /app/weights/best.pt (or point ITCS_WEIGHTS_PATH at it).
  2. Run with ITCS_DETECTOR=coco to use the auto-downloaded public YOLOv8n checkpoint. It detects generic COCO vehicle classes only and cannot identify an ambulance.
  3. Leave ITCS_DETECTOR=auto, which picks option 2 for you.
ERROR:    Application startup failed. Exiting.
```

`ITCS_DETECTOR=stub` under the default `ITCS_ENV=production` (exit code 1):

```console
$ docker run --rm -e ITCS_DETECTOR=stub p2-intelligent-traffic-control-system:api
app.errors.ConfigurationError: ITCS_DETECTOR=stub performs NO detection and is refused while ITCS_ENV='production'. Set ITCS_ENV=development (or 'test') as a second, deliberate opt-in if you really want the fake detector.
```
