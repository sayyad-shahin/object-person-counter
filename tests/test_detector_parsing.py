"""Checks how YOLO results are converted into Detection objects, using stand-in objects (no PyTorch needed)."""
import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from counter.detector import YoloTracker


class T:                      # behaves like a torch tensor for .cpu().numpy()
    def __init__(self, a): self.a = np.array(a)
    def cpu(self): return self
    def numpy(self): return self.a


class Boxes:
    def __init__(self, xyxy, cls, conf, ids): self.xyxy, self.cls, self.conf, self.id = T(xyxy), T(cls), T(conf), (None if ids is None else T(ids))
    def __len__(self): return len(self.cls.a)


class Result:
    names = {0: "person", 2: "car"}
    def __init__(self, boxes): self.boxes = boxes


class FakeModel:
    def __init__(self, result): self.result, self.kwargs, self.predictor = result, None, "old"
    def track(self, frame, **kw): self.kwargs = kw; return [self.result]


def make(result):
    t = YoloTracker.__new__(YoloTracker)          # skip __init__: it would import ultralytics and load weights
    t.model, t.names = FakeModel(result), {0: "person", 2: "car"}
    t.iou, t.imgsz, t.device, t.tracker_cfg = 0.5, 640, None, "bytetrack.yaml"
    t.configure(conf=0.4, classes=["person"])
    return t


class TestParsing(unittest.TestCase):
    def test_converts_boxes_with_ids(self):
        t = make(Result(Boxes([[10, 20, 30, 60], [5, 5, 15, 15]], [0, 2], [0.91, 0.6], [7, 8])))
        d = t.detect(np.zeros((100, 100, 3), np.uint8))
        self.assertEqual([(x.label, x.track_id, x.cls_id) for x in d], [("person", 7, 0), ("car", 8, 2)])
        self.assertEqual(d[0].box, (10.0, 20.0, 30.0, 60.0)); self.assertAlmostEqual(d[0].conf, 0.91, places=2)

    def test_settings_are_passed_to_model(self):
        t = make(Result(Boxes([[0, 0, 1, 1]], [0], [0.9], [1])))
        t.detect(np.zeros((10, 10, 3), np.uint8))
        kw = t.model.kwargs
        self.assertEqual((kw["conf"], kw["classes"], kw["persist"]), (0.4, [0], True))

    def test_no_ids_and_no_boxes(self):
        d = make(Result(Boxes([[0, 0, 5, 5]], [0], [0.8], None))).detect(np.zeros((10, 10, 3), np.uint8))
        self.assertIsNone(d[0].track_id)
        self.assertEqual(make(Result(Boxes(np.zeros((0, 4)), [], [], None))).detect(np.zeros((10, 10, 3), np.uint8)), [])

    def test_reset_clears_tracker_state(self):
        t = make(Result(None)); t.reset(); self.assertIsNone(t.model.predictor)


if __name__ == "__main__":
    unittest.main()
