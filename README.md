# 👁️ Real-time Object & Person Counter (YOLOv8)

Count people and objects in real time from a **webcam, a video file, or a stream**. YOLOv8 detects the objects, ByteTrack gives each one a persistent ID, and a counting engine turns that into useful numbers: how many are in view now, how many different ones have appeared, and how many crossed a virtual line **IN** or **OUT**.

Two ways to use it:
- **Command line** (`count.py`): an OpenCV window with live boxes, trails and counts. It can also save the annotated video and a CSV log.
- **Web dashboard** (`app.py`): live video in your browser with a counter panel, a chart and settings (choose classes, confidence, counting line, upload a video).

## Features
- Detects 80 object types (person, car, bus, truck, bicycle, dog, backpack... see `python count.py --list-classes`)
- **Live counts per class**, **total unique objects seen** (each tracker ID counts once), and **peak** counts
- **Line crossing counter** with IN / OUT directions (horizontal, vertical or custom line) and a net count
- **Zone counting**: how many objects are inside a polygon area
- Movement trails, FPS display, class filter, confidence control
- Save the annotated video and a per-second CSV log
- Works with any webcam, video file (mp4, avi, mov, mkv), or RTSP / HTTP stream
- Runs on CPU, and much faster on an NVIDIA GPU
- Counting logic is separate from the model and fully unit-tested (26 tests)

## How it works
```
video frame ──► YOLOv8 (detect) ──► ByteTrack (IDs) ──► CountingEngine ──► annotated frame + numbers
                                                         ├─ in view now (per class)
                                                         ├─ unique objects seen (per class)
                                                         ├─ line crossings IN / OUT
                                                         └─ objects inside zones
```
A crossing is counted when the movement of an object's reference point (box centre, or the bottom of the box with `--anchor bottom`) between two frames intersects the counting line segment. For people walking toward a camera, `--anchor bottom` (feet) is usually more accurate.

## Quick start

Requires **Python 3.9 to 3.12** (check PyTorch support if you use a newer version).

```bash
# 1. (optional) create a virtual environment
py -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate        # macOS / Linux

# 2. install dependencies (includes PyTorch, so this download is large)
pip install -r requirements.txt

# 3a. web dashboard
python app.py                    # then open http://127.0.0.1:5000

# 3b. or the command-line app with your webcam
python count.py
```
The first run downloads the model weights (`yolov8n.pt`, about 6 MB) automatically. On Windows, use `py` instead of `python` if `python` doesn't work.

## Command-line examples
```bash
python count.py                                          # webcam 0, count everything
python count.py --source people.mp4 --classes person --line h
python count.py --source people.mp4 --classes person --line h --anchor bottom --save out.mp4 --csv counts.csv --no-show
python count.py --source 0 --classes person car --zone 0.1,0.1,0.9,0.1,0.9,0.9,0.1,0.9
python count.py --source "rtsp://user:pass@192.168.1.10/stream"
python count.py --list-classes
```

| Option | Meaning |
|---|---|
| `--source` | Webcam number (`0`), video file, or stream URL |
| `--model` | `yolov8n.pt` (fastest), `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt` (most accurate) |
| `--conf` | Minimum confidence, 0 to 1 (default 0.35). Raise to reduce false detections |
| `--classes` | Names or numbers to count, e.g. `person car` |
| `--line` | `h`, `v`, `none`, or `x1,y1,x2,y2` as fractions of the frame |
| `--flip` | Swap the IN / OUT direction |
| `--zone` | Polygon `x1,y1,x2,y2,x3,y3,...` (fractions). Repeat for more zones |
| `--anchor` | `center` or `bottom` of each box, used for crossings and zones |
| `--device` | `cpu`, or `0` for the first GPU (default: automatic) |
| `--save`, `--csv` | Save the annotated video / a per-second CSV log |
| `--no-show` | Headless mode (no window) |
| `--loop`, `--max-frames` | Repeat a video / stop after N frames |

**Line direction:** IN is the direction the arrow on the line points to. For the horizontal line that is downward, and for the vertical line it is rightward. Use `--flip` to swap.

**Window keys:** `q` or `Esc` quit, `space` pause, `r` reset counts, `s` save a screenshot.

## Web dashboard
Run `python app.py` and open http://127.0.0.1:5000. Choose **Webcam** or **Video file**, pick what to count, adjust confidence and the counting line, then press **Start**. The panel shows live counts, a chart, a per-class table and the latest line crossings. The server uses the webcam of the computer it runs on.

## Choosing a model
| Model | Speed | Accuracy | Good for |
|---|---|---|---|
| `yolov8n.pt` | fastest | lowest | CPU, laptops, demos |
| `yolov8s.pt` | fast | good | balanced |
| `yolov8m/l/x.pt` | slower | higher | GPU, crowded scenes |

## Tips for accurate counts
- Mount the camera high and at an angle where people do not overlap heavily
- Place the line where objects pass one at a time and cross it fully
- Use `--classes person` to avoid counting objects you don't care about
- Raise `--conf` to reduce false positives, lower it if real objects are missed
- Use a larger model (`--model yolov8s.pt` or bigger) and `--imgsz 960` for small or distant objects

## Limitations
- Tracking can lose or swap IDs when people are hidden behind others or leave and re-enter the frame. A person who returns may get a new ID and be counted again in "total seen"
- Crowds with heavy overlap lower the accuracy of any detector
- Standard models detect the 80 COCO classes. For other objects, train a custom YOLOv8 model and pass it with `--model path/to/best.pt`
- Performance depends on your hardware. Expect low FPS on CPU with larger models

## Troubleshooting
- **Webcam does not open:** try `--source 1`, close other apps that use the camera, and check camera privacy permissions in Windows settings
- **Low FPS:** use `yolov8n.pt`, lower `--imgsz` (for example 480), or install the CUDA build of PyTorch for an NVIDIA GPU (see pytorch.org)
- **Tracking error about `lap`:** run `pip install lapx` (it is already listed in requirements.txt)
- **No window appears in `count.py`:** install `opencv-python` (not `opencv-python-headless`), or use `--no-show` with `--save`

## Tests
```bash
python -m unittest discover -s tests -v
```
The tests use a fake detector, so they run without PyTorch or a model download. They cover line crossings in both directions, unique counting, zones, class filtering, the drawing code, the conversion of YOLO results, and the web API with a real video file.

## Project structure
```
object-person-counter/
├── count.py                 # command-line app
├── app.py                   # web dashboard (Flask)
├── counter/
│   ├── engine.py            # counting logic (no ML dependencies)
│   ├── detector.py          # YOLOv8 + ByteTrack wrapper
│   ├── pipeline.py          # frame -> detections -> counts -> annotated frame
│   ├── draw.py              # boxes, trails, line, zones, info panel
│   ├── stream.py            # background worker for the web dashboard
│   ├── sources.py           # webcam / file / stream helpers
│   └── coco.py              # class names
├── templates/ static/       # dashboard page
├── tests/
├── requirements.txt
└── LICENSE
```

## Licensing note
This project's own code is MIT licensed (see [LICENSE](LICENSE)). It uses **Ultralytics YOLOv8**, which is released under **AGPL-3.0** (with a separate enterprise license available from Ultralytics). If you distribute this project as part of a product or run it as a network service, review the Ultralytics license terms and make sure you comply with them.

## Ideas to extend
- Heatmap of where people spend time, or dwell-time per zone
- Send counts to a database, MQTT, or a dashboard such as Grafana
- Alerts when a zone exceeds a limit
- Custom-trained model for your own objects
- Multi-camera support
