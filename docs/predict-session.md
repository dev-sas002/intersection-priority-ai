# `/predict` — captured session

Real request/response pairs, captured on 2026-09-23 against the service running
on `http://localhost:8250` with the settings `docker-compose.yml` ships
(`ITCS_MAX_FRAMES=60`, `ITCS_FRAME_STRIDE=3`, `ITCS_BATCH_SIZE=8`,
`ITCS_MAX_CONCURRENT_INFERENCES=2`).

No ambulance weights were present, so `ITCS_DETECTOR=auto` selected the public
COCO fallback — which is why every response carries a `notice` and reports
`label: "bus"` rather than `"ambulance"`. That is the honest answer for that
model, not a bug.

The `frame` field is a base64 JPEG. It has been replaced with its length here;
nothing else has been edited.

## Health

```console
$ curl -s http://localhost:8250/health | jq
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

## A clip that contains an emergency vehicle

```console
$ curl -s -F "file=@test_files/ambulance_video1.mp4" -F "approach=north" \
    http://localhost:8250/predict | jq
{
  "ambulance": true,
  "frame": "<base64 JPEG, 157168 chars>",
  "confidence": 0.7709326148033142,
  "label": "bus",
  "frames_read": 70,
  "frames_inferred": 22,
  "detector": "coco",
  "notice": "The active model has no ambulance class: it reports generic vehicle classes and cannot confirm that the vehicle is an ambulance. Load the purpose-trained weights (ITCS_WEIGHTS_PATH) for a real answer.",
  "priority": {
    "action": "no_change",
    "approach": "north",
    "reason": "below_confidence_floor",
    "held_for_s": null,
    "state": {
      "priority_active": false,
      "active_approach": null,
      "held_for_s": null,
      "cooldowns": {}
    }
  }
}
```

`priority.reason` is `below_confidence_floor`: the scan's 0.771 is under the
controller's `ITCS_PRIORITY_MIN_CONFIDENCE` of 0.80, so the sighting is not
treated as evidence. The detection and the priority decision are deliberately
held to separate bars.

## A clip that does not

```console
$ curl -s -F "file=@test_files/car_sample2_yolo.mp4" http://localhost:8250/predict | jq
{
  "ambulance": false,
  "frame": "<base64 JPEG, 96004 chars>",
  "confidence": 0.0,
  "label": null,
  "frames_read": 178,
  "frames_inferred": 60,
  "detector": "coco",
  "notice": "The active model has no ambulance class: it reports generic vehicle classes and cannot confirm that the vehicle is an ambulance. Load the purpose-trained weights (ITCS_WEIGHTS_PATH) for a real answer.",
  "priority": null
}
```

## Something that is not a video

```console
$ curl -s -w "\nHTTP %{http_code}\n" -F "file=@README.md" http://localhost:8250/predict
{"detail":"Uploaded file could not be read as a video."}
HTTP 400
```

## Six simultaneous uploads

`ITCS_MAX_CONCURRENT_INFERENCES=2`, so the six requests complete in three
pairs rather than fighting over the CPU. None is dropped: the queue only
returns 503 past `ITCS_INFERENCE_QUEUE_SLOTS` (8).

```console
$ for i in $(seq 1 6); do
>   curl -s -o /dev/null -w "%{http_code}  %{time_total}s\n" \
>     -F "file=@test_files/ambulance_video1.mp4" http://localhost:8250/predict &
> done; wait
200  1.837622s
200  2.023958s
200  3.577318s
200  3.783004s
200  5.429247s
200  5.599744s
```

## What the response-size cap is worth

The same request, same clip, same JPEG quality — only `ITCS_MAX_FRAME_EDGE`
differs. The source frame is 4096x2160.

```console
ITCS_MAX_FRAME_EDGE=1280 (default)  response bytes: 157527
ITCS_MAX_FRAME_EDGE=0    (uncapped)  response bytes: 928463
```

Before the cap existed the response was **1,757,211 bytes** (no resize, OpenCV's
default JPEG quality of 95). The shipped defaults — a 1280 px longest edge at
quality 85 — bring the same response to 157,527 bytes, **11x smaller**.

## When the weights are missing or the fake detector is asked for

```console
### ITCS_DETECTOR=yolo with no weights on disk
DetectorUnavailableError:
Ambulance weights not found at /srv/itcs/weights/best.pt.
These weights are trained per-deployment and are deliberately NOT committed to this repository.
Options:
  1. Train or obtain a single-class YOLOv8 ambulance checkpoint and place it at /srv/itcs/weights/best.pt (or point ITCS_WEIGHTS_PATH at it).
  2. Run with ITCS_DETECTOR=coco to use the auto-downloaded public YOLOv8n checkpoint. It detects generic COCO vehicle classes only and cannot identify an ambulance.
  3. Leave ITCS_DETECTOR=auto, which picks option 2 for you.

### ITCS_DETECTOR=stub with the default ITCS_ENV=production
ConfigurationError:
ITCS_DETECTOR=stub performs NO detection and is refused while ITCS_ENV='production'. Set ITCS_ENV=development (or 'test') as a second, deliberate opt-in if you really want the fake detector.
```
