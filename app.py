"""Web dashboard.   python app.py   then open http://127.0.0.1:5000
Shows the live video with detections and live counts. Use a webcam or upload a video."""
import argparse
import os
import time

from flask import Flask, Response, jsonify, render_template, request

from counter.coco import COCO_NAMES, resolve_classes
from counter.engine import parse_line
from counter.stream import StreamManager

MODELS = {"yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"}
BOUNDARY = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"


def create_app(manager=None):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024
    mgr = manager or StreamManager()
    app.manager = mgr

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        return jsonify(status="ok")

    @app.get("/api/classes")
    def classes():
        return jsonify(COCO_NAMES)

    @app.get("/api/stats")
    def stats():
        return jsonify(mgr.snapshot())

    @app.post("/api/upload")
    def upload():
        f = request.files.get("video")
        if not f or not f.filename:
            return jsonify(error="No video file received."), 400
        try:
            return jsonify(upload_id=mgr.save_upload(f), name=f.filename)
        except ValueError as e:
            return jsonify(error=str(e)), 400

    @app.post("/api/start")
    def start():
        d = request.get_json(silent=True) or {}
        try:
            conf = float(d.get("conf", 0.35))
            if not 0.05 <= conf <= 0.95:
                raise ValueError("Confidence must be between 0.05 and 0.95.")
            classes = d.get("classes") or None
            if classes is not None and not isinstance(classes, list):
                raise ValueError("'classes' must be a list of class names.")
            model = d.get("model", "yolov8n.pt")
            if model not in MODELS:
                raise ValueError("Unknown model.")
            line = d.get("line", "h")
            parse_line(line)
            resolve_classes(classes)
            mgr.start(d.get("source", ""), conf=conf, classes=classes, line=line, flip=bool(d.get("flip", False)), model=model)
        except (ValueError, TypeError) as e:
            return jsonify(error=str(e)), 400
        return jsonify(status="starting")

    @app.post("/api/stop")
    def stop():
        mgr.stop()
        return jsonify(status="stopped")

    @app.post("/api/reset")
    def reset():
        mgr.reset()
        return jsonify(status="reset")

    @app.get("/video_feed")
    def video_feed():
        def frames():
            last = -1
            while True:
                jpeg, jid = mgr.latest()
                if jid != last:
                    last = jid
                    yield BOUNDARY + jpeg + b"\r\n"
                time.sleep(0.03)
        return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    return app


app = create_app()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5000)))
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    print(f"Open http://{a.host}:{a.port} in your browser (Ctrl+C to stop)")
    app.run(host=a.host, port=a.port, debug=False, threaded=True)
