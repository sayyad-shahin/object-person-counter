"""Runs the counter in a background thread and keeps the latest annotated frame for the web page."""
import os
import threading
import time
import uuid

import cv2
import numpy as np

from .coco import COCO_NAMES, resolve_classes
from .engine import CountingEngine, parse_line
from .pipeline import CounterPipeline
from .sources import is_file, open_capture

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALLOWED_VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
MAX_WIDTH = 960


def _default_factory(**kw):
    from .detector import YoloTracker
    return YoloTracker(**kw)


def placeholder_jpeg(text, size=(640, 360)):
    img = np.full((size[1], size[0], 3), (28, 20, 14), np.uint8)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(img, text, ((size[0] - tw) // 2, (size[1] + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (210, 210, 210), 2, cv2.LINE_AA)
    return cv2.imencode(".jpg", img)[1].tobytes()


class StreamManager:
    def __init__(self, tracker_factory=None, upload_dir=None):
        self.tracker_factory = tracker_factory or _default_factory
        self.upload_dir = upload_dir or os.path.join(ROOT, "uploads")
        os.makedirs(self.upload_dir, exist_ok=True)
        self.uploads = {}
        self._lock = threading.Lock()          # protects the shared state below
        self._work = threading.Lock()          # held while a frame is processed or counts are reset
        self._stop = threading.Event()
        self._thread = None
        self._tracker, self._tracker_model = None, None
        self._pipeline = None
        self._snap = CountingEngine().snapshot()
        self._jpeg, self._jpeg_id = placeholder_jpeg("Choose a source and press Start"), 0
        self.status, self.error, self.source_name, self.fps = "idle", "", "", 0.0

    # ------------------------------------------------------------------ uploads
    def save_upload(self, file_storage):
        ext = os.path.splitext(file_storage.filename or "")[1].lower()
        if ext not in ALLOWED_VIDEO:
            raise ValueError("Please upload a video file (mp4, avi, mov, mkv, webm).")
        self._cleanup_uploads()
        uid = uuid.uuid4().hex[:12]
        path = os.path.join(self.upload_dir, uid + ext)
        file_storage.save(path)
        self.uploads[uid] = (path, os.path.basename(file_storage.filename))
        return uid

    def _cleanup_uploads(self, max_age=24 * 3600):
        for name in os.listdir(self.upload_dir):
            p = os.path.join(self.upload_dir, name)
            if os.path.isfile(p) and time.time() - os.path.getmtime(p) > max_age:
                os.remove(p)

    def resolve_source(self, spec):
        kind, _, value = str(spec).partition(":")
        if kind == "webcam" and value.isdigit() and int(value) < 10:
            return int(value), f"Webcam {value}"
        if kind == "upload" and value in self.uploads:
            path, name = self.uploads[value]
            return path, name
        raise ValueError("Unknown video source. Choose a webcam or upload a video first.")

    # ------------------------------------------------------------------ control
    def start(self, source_spec, conf=0.35, classes=None, line="h", flip=False, model="yolov8n.pt", loop=False):
        source, name = self.resolve_source(source_spec)
        parse_line(line)                                   # validate early (raises ValueError)
        resolve_classes(classes)
        self.stop()
        with self._lock:
            self.status, self.error, self.source_name, self.fps = "starting", "", name, 0.0
            self._snap = CountingEngine().snapshot()
            self._set_jpeg(placeholder_jpeg("Loading..."))
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(source, conf, classes, line, flip, model, loop), daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread and self._thread.is_alive():
            self._stop.set()
            self._thread.join(timeout=15)
        self._thread = None

    def reset(self):
        with self._work:
            if self._pipeline:
                self._pipeline.reset()
                self._snap = self._pipeline.engine.snapshot()

    # ------------------------------------------------------------------ results
    def _set_jpeg(self, data):
        self._jpeg, self._jpeg_id = data, self._jpeg_id + 1

    def latest(self):
        with self._lock:
            return self._jpeg, self._jpeg_id

    def snapshot(self):
        with self._lock:
            out = dict(self._snap)
            out.update(status=self.status, error=self.error, source=self.source_name, fps=round(self.fps, 1))
            return out

    # ------------------------------------------------------------------ worker thread
    def _run(self, source, conf, classes, line, flip, model, loop):
        cap = None
        try:
            with self._lock:
                self.status = "loading model"
            if self._tracker is None or self._tracker_model != model:
                self._tracker = self.tracker_factory(model=model, conf=conf, classes=classes)
                self._tracker_model = model
            else:
                self._tracker.configure(conf=conf, classes=classes)
                self._tracker.reset()
            names = getattr(self._tracker, "names", COCO_NAMES)
            ids = resolve_classes(classes, names)
            engine = CountingEngine(line=parse_line(line), flip=flip,
                                    classes=None if ids is None else [names[i] for i in ids])
            self._pipeline = CounterPipeline(self._tracker, engine)
            cap = open_capture(source)
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            file_mode = is_file(source)
            with self._lock:
                self.status = "running"
            while not self._stop.is_set():
                started = time.perf_counter()
                ok, frame = cap.read()
                if not ok:
                    if file_mode and loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                if frame.shape[1] > MAX_WIDTH:
                    scale = MAX_WIDTH / frame.shape[1]
                    frame = cv2.resize(frame, (MAX_WIDTH, int(frame.shape[0] * scale)))
                with self._work:
                    out, snap = self._pipeline.process(frame)
                data = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
                with self._lock:
                    self._snap, self.fps = snap, self._pipeline.fps
                    self._set_jpeg(data)
                if file_mode:                              # play video files at their normal speed
                    time.sleep(max(0.0, 1.0 / src_fps - (time.perf_counter() - started)))
            with self._lock:
                self.status = "stopped" if self._stop.is_set() else "finished"
        except Exception as e:                             # noqa: BLE001  (show any problem on the page)
            with self._lock:
                self.status, self.error = "error", str(e)
                self._set_jpeg(placeholder_jpeg("Error - see message on the page"))
        finally:
            if cap is not None:
                cap.release()
