"""One frame in, one annotated frame and the latest counts out."""
import time
from collections import deque

from .draw import annotate


class CounterPipeline:
    def __init__(self, tracker, engine, show_trails=True, trail_length=30):
        self.tracker, self.engine = tracker, engine
        self.show_trails, self.trail_length = show_trails, trail_length
        self.trails = {}
        self.fps = 0.0
        self._last = None

    def reset(self):
        self.engine.reset()
        self.tracker.reset()
        self.trails.clear()

    def process(self, frame):
        h, w = frame.shape[:2]
        detections = self.tracker.detect(frame)
        snap = self.engine.update(detections, (w, h))
        detections = [d for d in detections if self.engine.classes is None or d.label in self.engine.classes]

        now = time.perf_counter()
        if self._last is not None and now > self._last:
            inst = 1.0 / (now - self._last)
            self.fps = inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst
        self._last = now

        if self.show_trails:
            alive = set()
            for d in detections:
                if d.track_id is None:
                    continue
                cx, cy = d.anchor(self.engine.anchor)
                self.trails.setdefault(d.track_id, deque(maxlen=self.trail_length)).append((int(cx), int(cy)))
                alive.add(d.track_id)
            for tid in [t for t in self.trails if t not in alive and len(self.trails[t]) and self.engine.frame % 30 == 0]:
                del self.trails[tid]
        annotate(frame, detections, self.engine, snap, self.fps, self.trails if self.show_trails else None)
        return frame, snap
