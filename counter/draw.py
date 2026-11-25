"""Drawing: boxes, trails, the counting line, zones and the information panel."""
import colorsys
import cv2
import numpy as np

WHITE, BLACK, LINE_COLOR = (255, 255, 255), (20, 20, 20), (0, 215, 255)


def color_for(cls_id):
    r, g, b = colorsys.hsv_to_rgb((cls_id * 0.137) % 1.0, 0.75, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def _text(img, s, org, scale, color=WHITE, thick=1, bg=None):
    (tw, th), base = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    if bg is not None:
        cv2.rectangle(img, (x - 2, y - th - 3), (x + tw + 2, y + base), bg, -1)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
    return tw


def annotate(frame, detections, engine, snap, fps=0.0, trails=None):
    """Draw everything onto frame (in place) and return it."""
    h, w = frame.shape[:2]
    k = max(0.55, w / 1280.0)
    thick = max(1, int(round(k * 2)))

    for zone in engine.zones:                                     # zones: translucent polygons
        pts = np.array([(int(x * w), int(y * h)) for x, y in zone], np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (255, 160, 40))
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
        cv2.polylines(frame, [pts], True, (255, 160, 40), thick, cv2.LINE_AA)

    if engine.line:                                               # counting line with direction arrow
        x1, y1, x2, y2 = engine.line
        a, b = (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))
        cv2.line(frame, a, b, LINE_COLOR, thick + 1, cv2.LINE_AA)
        mx, my = (a[0] + b[0]) // 2, (a[1] + b[1]) // 2
        dx, dy = b[0] - a[0], b[1] - a[1]
        n = max(1.0, float(np.hypot(dx, dy)))
        nx, ny = -dy / n, dx / n                                   # normal pointing to the 'positive' side
        if engine.flip:
            nx, ny = -nx, -ny
        cv2.arrowedLine(frame, (mx, my), (int(mx + nx * 40 * k), int(my + ny * 40 * k)), LINE_COLOR, thick, cv2.LINE_AA, tipLength=0.35)
        _text(frame, "IN", (int(mx + nx * 50 * k) - 8, int(my + ny * 50 * k) + 5), 0.5 * k, LINE_COLOR, thick)

    if trails:                                                    # short movement trails
        for tid, pts in trails.items():
            if len(pts) > 1:
                cv2.polylines(frame, [np.array(pts, np.int32)], False, (0, 255, 255), max(1, thick - 1), cv2.LINE_AA)

    for d in detections:                                          # boxes + labels
        x1, y1, x2, y2 = (int(v) for v in d.box)
        col = color_for(d.cls_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, thick)
        tag = f"{d.label} #{d.track_id} {d.conf:.2f}" if d.track_id is not None else f"{d.label} {d.conf:.2f}"
        _text(frame, tag, (x1 + 2, max(y1 - 5, 14)), 0.5 * k, BLACK, 1, bg=col)

    # information panel
    lines = [f"FPS {fps:4.1f}", f"In view: {snap['total_current']}"]
    lines += [f"  {lbl}: {n}" for lbl, n in sorted(snap["current"].items(), key=lambda t: -t[1])[:5]]
    if snap["unique"]:
        lines.append("Total seen: " + ", ".join(f"{l} {n}" for l, n in sorted(snap["unique"].items(), key=lambda t: -t[1])[:3]))
    if snap["line"]["enabled"]:
        lines.append(f"Line  IN {snap['line']['in_total']}   OUT {snap['line']['out_total']}")
    for i, z in enumerate(snap["zones"], 1):
        lines.append(f"Zone {i}: {z['total_current']} inside")
    scale, lh = 0.55 * k, int(24 * k)
    pw = int(max(cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] for s in lines) + 20)
    ph = lh * len(lines) + 12
    roi = frame[8:8 + ph, 8:8 + pw]
    if roi.size:
        roi[:] = (roi * 0.35).astype(np.uint8)
    for i, s in enumerate(lines):
        _text(frame, s, (16, 8 + lh * (i + 1) - 2), scale, WHITE if i else (120, 255, 120), 1)
    return frame
