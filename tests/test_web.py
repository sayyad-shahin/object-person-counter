import io, os, sys, tempfile, time, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
from app import create_app
from counter.engine import Detection
from counter.stream import StreamManager


def make_video(path, frames=20, size=(160, 120)):
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 25, size)
    for _ in range(frames):
        w.write(np.full((size[1], size[0], 3), 90, np.uint8))
    w.release()


class MovingPerson:
    """Pretend detector: one person walks from the top of the frame to the bottom, one car stays still."""
    names = ["person", "bicycle", "car"]

    def __init__(self, **kw): self.i = 0
    def configure(self, **kw): pass
    def reset(self): self.i = 0
    def detect(self, frame):
        h, w = frame.shape[:2]
        y = 10 + self.i * (h - 20) / 19
        self.i += 1
        return [Detection(0, "person", 0.9, (60, y - 8, 80, y + 8), 1), Detection(2, "car", 0.8, (10, 10, 40, 30), 2)]


class TestWeb(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.mgr = StreamManager(tracker_factory=MovingPerson, upload_dir=self.dir.name)
        self.c = create_app(self.mgr).test_client()

    def tearDown(self):
        self.mgr.stop(); self.dir.cleanup()

    def wait_for(self, status, timeout=15):
        end = time.time() + timeout
        while time.time() < end:
            s = self.c.get("/api/stats").get_json()
            if s["status"] == status:
                return s
            time.sleep(0.1)
        self.fail(f"never reached '{status}', last: {s['status']} {s['error']}")

    def test_basic_pages(self):
        self.assertEqual(self.c.get("/").status_code, 200)
        self.assertEqual(self.c.get("/api/health").get_json()["status"], "ok")
        self.assertEqual(len(self.c.get("/api/classes").get_json()), 80)
        s = self.c.get("/api/stats").get_json()
        self.assertEqual((s["status"], s["total_current"]), ("idle", 0))

    def test_upload_validation(self):
        self.assertEqual(self.c.post("/api/upload", data={}).status_code, 400)
        r = self.c.post("/api/upload", data={"video": (io.BytesIO(b"x"), "notes.txt")}, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400)

    def test_start_validation(self):
        bad = [{"source": "webcam:99"}, {"source": "upload:nope"}, {"source": "rtsp://x"}, {"source": "webcam:0", "conf": 5},
               {"source": "webcam:0", "classes": ["unicorn"]}, {"source": "webcam:0", "line": "diagonal"},
               {"source": "webcam:0", "model": "evil.pt"}, {"source": "webcam:0", "classes": "person"}]
        for body in bad:
            self.assertEqual(self.c.post("/api/start", json=body).status_code, 400, body)

    def test_full_run_counts_a_crossing(self):
        path = os.path.join(self.dir.name, "t.avi"); make_video(path)
        with open(path, "rb") as f:
            up = self.c.post("/api/upload", data={"video": (f, "t.avi")}, content_type="multipart/form-data").get_json()
        r = self.c.post("/api/start", json={"source": "upload:" + up["upload_id"], "line": "h", "classes": ["person"]})
        self.assertEqual(r.status_code, 200)
        s = self.wait_for("finished")
        self.assertEqual(s["line"]["in_total"], 1); self.assertEqual(s["line"]["out_total"], 0)
        self.assertEqual(s["unique"], {"person": 1})          # the car was filtered out
        self.assertEqual(s["frame"], 20)
        jpeg, _ = self.mgr.latest()
        self.assertTrue(jpeg.startswith(b"\xff\xd8"))        # a real JPEG frame
        self.c.post("/api/reset")
        self.assertEqual(self.c.get("/api/stats").get_json()["total_unique"], 0)

    def test_stop_and_missing_file_error(self):
        path = os.path.join(self.dir.name, "t.avi"); make_video(path, frames=200)
        with open(path, "rb") as f:
            up = self.c.post("/api/upload", data={"video": (f, "t.avi")}, content_type="multipart/form-data").get_json()
        self.c.post("/api/start", json={"source": "upload:" + up["upload_id"]})
        self.wait_for("running")
        self.c.post("/api/stop")
        self.assertEqual(self.c.get("/api/stats").get_json()["status"], "stopped")
        os.remove(self.mgr.uploads[up["upload_id"]][0])          # the saved upload is gone -> clear error message
        self.c.post("/api/start", json={"source": "upload:" + up["upload_id"]})
        s = self.wait_for("error")
        self.assertIn("Could not open", s["error"])


if __name__ == "__main__":
    unittest.main()
