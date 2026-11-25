"""All the counting logic. Pure Python: it needs no deep learning library, so it is easy to test.

Three kinds of numbers are produced:
  * IN VIEW   : how many objects of each class are visible in the current frame.
  * UNIQUE    : how many different tracked objects have been seen since the start (each tracker id counts once).
  * LINE      : how many objects crossed a virtual line in each direction (IN / OUT), plus optional ZONES.
"""
from collections import Counter, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    cls_id: int
    label: str
    conf: float
    box: tuple              # (x1, y1, x2, y2) in pixels
    track_id: object = None  # tracker id (int) or None when tracking is not available

    def anchor(self, mode="center"):
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0) if mode == "center" else ((x1 + x2) / 2.0, float(y2))


# ----------------------------------------------------------------------------- geometry
def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def crossing_direction(prev, cur, a, b):
    """Return 'in', 'out' or None when the move prev->cur crosses the line segment a->b.

    'in' means moving to the side the arrow points at. For the default lines that is: downward across a horizontal
    line ('h') and rightward across a vertical line ('v'). Use flip to swap the meaning.
    """
    d1, d2 = _orient(a, b, prev), _orient(a, b, cur)
    if (d1 < 0) == (d2 < 0):
        return None                                  # both points on the same side
    if _orient(prev, cur, a) * _orient(prev, cur, b) > 0:
        return None                                  # the move passes beyond the end of the segment
    return "in" if d1 < 0 else "out"


def point_in_polygon(pt, poly):
    x, y = pt
    inside = False
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            inside = not inside
    return inside


def parse_line(text):
    """'h' / 'v' / 'none' or 'x1,y1,x2,y2' with values from 0 to 1 (fractions of the frame)."""
    if text is None or str(text).strip().lower() in ("", "none", "off"):
        return None
    t = str(text).strip().lower()
    if t in ("h", "horizontal"):
        return (0.0, 0.5, 1.0, 0.5)
    if t in ("v", "vertical"):
        return (0.5, 1.0, 0.5, 0.0)      # drawn bottom to top, so moving right = IN
    try:
        vals = tuple(float(v) for v in t.replace(" ", "").split(","))
    except ValueError:
        vals = ()
    if len(vals) != 4 or any(not 0.0 <= v <= 1.0 for v in vals) or vals[:2] == vals[2:]:
        raise ValueError("Line must be 'h', 'v', 'none' or 'x1,y1,x2,y2' with values between 0 and 1, e.g. 0,0.5,1,0.5")
    return vals


def parse_zone(text):
    """'x1,y1,x2,y2,x3,y3,...' (at least 3 points, values from 0 to 1) -> list of (x, y)."""
    try:
        vals = [float(v) for v in str(text).replace(" ", "").split(",") if v != ""]
    except ValueError:
        vals = []
    if len(vals) < 6 or len(vals) % 2 or any(not 0.0 <= v <= 1.0 for v in vals):
        raise ValueError("Zone needs at least 3 points as x1,y1,x2,y2,x3,y3,... with values between 0 and 1")
    return [(vals[i], vals[i + 1]) for i in range(0, len(vals), 2)]


# ----------------------------------------------------------------------------- engine
class CountingEngine:
    def __init__(self, line=None, flip=False, zones=None, classes=None, anchor="center", forget_after=90):
        """line  : (x1, y1, x2, y2) as fractions of the frame, or None
        zones : list of polygons, each a list of (x, y) fractions
        classes: only count these class names (None = everything)
        """
        self.line, self.flip = line, flip
        self.zones = list(zones or [])
        self.classes = set(classes) if classes else None
        self.anchor = anchor
        self.forget_after = forget_after
        self.reset()

    def reset(self):
        self.frame = 0
        self.current = Counter()
        self.peak = Counter()
        self.unique = {}
        self.line_counts = {"in": Counter(), "out": Counter()}
        self.zone_current = [Counter() for _ in self.zones]
        self.zone_unique = [dict() for _ in self.zones]
        self.events = deque(maxlen=30)
        self.history = deque(maxlen=120)
        self._last = {}          # track id -> (point, frame number)

    # ------------------------------------------------------------------ main step
    def update(self, detections, frame_size):
        w, h = frame_size
        dets = [d for d in detections if self.classes is None or d.label in self.classes]
        self.frame += 1
        self.current = Counter(d.label for d in dets)
        for label, n in self.current.items():
            self.peak[label] = max(self.peak[label], n)
        self.history.append(sum(self.current.values()))

        line = None
        if self.line:
            x1, y1, x2, y2 = self.line
            line = ((x1 * w, y1 * h), (x2 * w, y2 * h))
            if self.flip:
                line = (line[1], line[0])
        polys = [[(x * w, y * h) for x, y in z] for z in self.zones]
        self.zone_current = [Counter() for _ in self.zones]

        for d in dets:
            pt = d.anchor(self.anchor)
            for zi, poly in enumerate(polys):
                if point_in_polygon(pt, poly):
                    self.zone_current[zi][d.label] += 1
                    if d.track_id is not None:
                        self.zone_unique[zi].setdefault(d.label, set()).add(d.track_id)
            if d.track_id is None:
                continue
            self.unique.setdefault(d.label, set()).add(d.track_id)
            prev = self._last.get(d.track_id)
            if line and prev is not None:
                way = crossing_direction(prev[0], pt, line[0], line[1])
                if way:
                    self.line_counts[way][d.label] += 1
                    self.events.appendleft({"frame": self.frame, "label": d.label, "id": d.track_id, "direction": way})
            self._last[d.track_id] = (pt, self.frame)

        stale = [t for t, (_, f) in self._last.items() if self.frame - f > self.forget_after]
        for t in stale:
            del self._last[t]
        return self.snapshot()

    # ------------------------------------------------------------------ results
    def snapshot(self):
        ins, outs = dict(self.line_counts["in"]), dict(self.line_counts["out"])
        return {
            "frame": self.frame,
            "current": dict(self.current),
            "total_current": sum(self.current.values()),
            "peak": dict(self.peak),
            "unique": {k: len(v) for k, v in self.unique.items()},
            "total_unique": sum(len(v) for v in self.unique.values()),
            "line": {"enabled": self.line is not None, "in": ins, "out": outs,
                     "in_total": sum(ins.values()), "out_total": sum(outs.values()),
                     "net": sum(ins.values()) - sum(outs.values())},
            "zones": [{"current": dict(c), "total_current": sum(c.values()),
                       "unique": {k: len(v) for k, v in u.items()}}
                      for c, u in zip(self.zone_current, self.zone_unique)],
            "events": list(self.events)[:10],
            "history": list(self.history),
        }
