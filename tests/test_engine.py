import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from counter import CountingEngine, CounterPipeline, Detection, parse_line, parse_zone
from counter.coco import COCO_NAMES, resolve_classes
from counter.engine import crossing_direction, point_in_polygon

SIZE = (200, 100)   # width, height


def det(tid, cx, cy, label="person", cls=0, half=5):
    return Detection(cls, label, 0.9, (cx - half, cy - half, cx + half, cy + half), tid)


class TestGeometry(unittest.TestCase):
    def test_crossing_directions(self):
        a, b = (0, 50), (200, 50)                       # horizontal line, left to right
        self.assertEqual(crossing_direction((100, 40), (100, 60), a, b), "in")
        self.assertEqual(crossing_direction((100, 60), (100, 40), a, b), "out")
        self.assertIsNone(crossing_direction((100, 10), (100, 40), a, b))      # stays above
        self.assertIsNone(crossing_direction((250, 40), (250, 60), a, b))      # passes beyond the segment end

    def test_landing_exactly_on_line_counts_once(self):
        a, b = (0, 50), (200, 50)
        self.assertEqual(crossing_direction((100, 40), (100, 50), a, b), "in")
        self.assertIsNone(crossing_direction((100, 50), (100, 60), a, b))

    def test_polygon(self):
        sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
        self.assertTrue(point_in_polygon((5, 5), sq)); self.assertFalse(point_in_polygon((15, 5), sq))

    def test_parsers(self):
        self.assertEqual(parse_line("h"), (0.0, 0.5, 1.0, 0.5)); self.assertIsNone(parse_line("none"))
        self.assertEqual(parse_line("0.1,0.2,0.9,0.8"), (0.1, 0.2, 0.9, 0.8))
        for bad in ("1,2,3", "abc", "0,0,0,0", "0,0,2,1"):
            with self.assertRaises(ValueError): parse_line(bad)
        self.assertEqual(len(parse_zone("0,0,1,0,1,1")), 3)
        with self.assertRaises(ValueError): parse_zone("0,0,1,1")

    def test_resolve_classes(self):
        self.assertEqual(resolve_classes(["person", "Car", "5"]), [0, 2, 5]); self.assertIsNone(resolve_classes(None))
        self.assertIsNone(resolve_classes(["all"])); self.assertEqual(len(COCO_NAMES), 80)
        with self.assertRaises(ValueError): resolve_classes(["unicorn"])


class TestEngine(unittest.TestCase):
    def test_counts_in_view_and_unique(self):
        e = CountingEngine()
        s = e.update([det(1, 20, 20), det(2, 60, 20), det(3, 90, 20, "car", 2)], SIZE)
        self.assertEqual(s["current"], {"person": 2, "car": 1}); self.assertEqual(s["total_current"], 3)
        s = e.update([det(1, 22, 20), det(4, 150, 20)], SIZE)
        self.assertEqual(s["current"], {"person": 2}); self.assertEqual(s["unique"], {"person": 3, "car": 1})
        self.assertEqual(s["peak"]["person"], 2); self.assertEqual(s["total_unique"], 4)

    def test_same_id_is_not_counted_twice(self):
        e = CountingEngine()
        for _ in range(20): s = e.update([det(7, 50, 20)], SIZE)
        self.assertEqual(s["unique"], {"person": 1})

    def test_line_in_and_out(self):
        e = CountingEngine(line=parse_line("h"))
        e.update([det(1, 50, 30), det(2, 100, 70)], SIZE)
        s = e.update([det(1, 50, 60), det(2, 100, 40)], SIZE)            # 1 moves down (in), 2 moves up (out)
        self.assertEqual(s["line"]["in_total"], 1); self.assertEqual(s["line"]["out_total"], 1)
        self.assertEqual(s["line"]["net"], 0)
        s = e.update([det(1, 50, 70), det(2, 100, 20)], SIZE)            # no new crossing
        self.assertEqual(s["line"]["in_total"], 1)
        self.assertEqual(len(s["events"]), 2)

    def test_flip_swaps_direction(self):
        e = CountingEngine(line=parse_line("h"), flip=True)
        e.update([det(1, 50, 30)], SIZE); s = e.update([det(1, 50, 60)], SIZE)
        self.assertEqual((s["line"]["in_total"], s["line"]["out_total"]), (0, 1))

    def test_line_counts_per_class(self):
        e = CountingEngine(line=parse_line("v"))
        e.update([det(1, 90, 50, "car", 2)], SIZE); s = e.update([det(1, 110, 50, "car", 2)], SIZE)
        self.assertEqual(s["line"]["in"], {"car": 1})

    def test_bottom_anchor(self):
        e = CountingEngine(line=parse_line("h"), anchor="bottom")           # feet cross at y=50 when centre is at y=45
        e.update([det(1, 50, 40)], SIZE); s = e.update([det(1, 50, 46)], SIZE)
        self.assertEqual(s["line"]["in_total"], 1)

    def test_class_filter(self):
        e = CountingEngine(classes=["person"])
        s = e.update([det(1, 20, 20), det(2, 60, 20, "car", 2)], SIZE)
        self.assertEqual(s["current"], {"person": 1}); self.assertNotIn("car", s["unique"])

    def test_zone(self):
        e = CountingEngine(zones=[parse_zone("0,0,0.5,0,0.5,1,0,1")])         # left half
        s = e.update([det(1, 40, 50), det(2, 150, 50)], SIZE)
        self.assertEqual(s["zones"][0]["current"], {"person": 1})
        s = e.update([det(3, 30, 50)], SIZE)
        self.assertEqual(s["zones"][0]["unique"], {"person": 2})

    def test_no_track_ids_still_counts_current(self):
        e = CountingEngine(line=parse_line("h"))
        s = e.update([det(None, 50, 30), det(None, 90, 30)], SIZE)
        self.assertEqual(s["total_current"], 2); self.assertEqual(s["total_unique"], 0)

    def test_reset(self):
        e = CountingEngine(line=parse_line("h"))
        e.update([det(1, 50, 30)], SIZE); e.update([det(1, 50, 60)], SIZE); e.reset()
        s = e.snapshot()
        self.assertEqual((s["frame"], s["total_unique"], s["line"]["in_total"]), (0, 0, 0))

    def test_old_tracks_are_forgotten(self):
        e = CountingEngine(forget_after=5)
        e.update([det(1, 50, 30)], SIZE)
        for _ in range(10): e.update([], SIZE)
        self.assertNotIn(1, e._last)


class FakeTracker:
    def __init__(self, script): self.script, self.i = script, 0
    def detect(self, frame):
        d = self.script[min(self.i, len(self.script) - 1)]; self.i += 1; return d
    def reset(self): self.i = 0
    def configure(self, **kw): pass


class TestPipeline(unittest.TestCase):
    def test_pipeline_draws_and_counts(self):
        script = [[det(1, 100, 30)], [det(1, 100, 70)]]
        pipe = CounterPipeline(FakeTracker(script), CountingEngine(line=parse_line("h")))
        blank = np.zeros((100, 200, 3), np.uint8)
        out1, _ = pipe.process(blank.copy()); out2, snap = pipe.process(blank.copy())
        self.assertEqual(snap["line"]["in_total"], 1)
        self.assertGreater(int(out2.sum()), 0)                  # something was drawn
        pipe.reset(); self.assertEqual(pipe.engine.snapshot()["frame"], 0)


if __name__ == "__main__":
    unittest.main()
