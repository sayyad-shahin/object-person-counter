"""Real-time object and person counter (command line).

Examples
  python count.py                                   # webcam 0, count everything
  python count.py --source people.mp4 --classes person --line h
  python count.py --source 0 --classes person car --zone 0.1,0.1,0.9,0.1,0.9,0.9,0.1,0.9
  python count.py --source video.mp4 --save out.mp4 --csv counts.csv --no-show

Keys in the window:  q / Esc quit   space pause   r reset counts   s save screenshot
"""
import argparse, csv, json, os, sys, time
import cv2

from counter import CountingEngine, CounterPipeline, parse_line, parse_zone
from counter.coco import COCO_NAMES, resolve_classes
from counter.sources import is_file, make_writer, open_capture


def build_parser():
    ap = argparse.ArgumentParser(description="Real-time object & person counter using YOLOv8 + tracking",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--source", default="0", help="webcam number, video file, or stream URL (default 0)")
    ap.add_argument("--model", default="yolov8n.pt", help="yolov8n.pt (fast) ... yolov8x.pt (accurate)")
    ap.add_argument("--conf", type=float, default=0.35, help="minimum detection confidence 0-1")
    ap.add_argument("--imgsz", type=int, default=640, help="inference image size")
    ap.add_argument("--device", default=None, help="'cpu', '0' (first GPU)... default: automatic")
    ap.add_argument("--classes", nargs="*", default=None, help="class names or numbers to count, e.g. person car")
    ap.add_argument("--line", default="none", help="counting line: h, v, none, or x1,y1,x2,y2 (0-1 fractions)")
    ap.add_argument("--flip", action="store_true", help="swap the IN / OUT direction of the line")
    ap.add_argument("--zone", action="append", default=[], help="polygon x1,y1,x2,y2,x3,y3,... (0-1). Repeatable")
    ap.add_argument("--anchor", choices=["center", "bottom"], default="center", help="point of the box used for crossings")
    ap.add_argument("--no-trails", action="store_true", help="do not draw movement trails")
    ap.add_argument("--save", help="save the annotated video, e.g. out.mp4")
    ap.add_argument("--csv", help="write counts to a CSV file once per second")
    ap.add_argument("--no-show", action="store_true", help="do not open a window (headless)")
    ap.add_argument("--loop", action="store_true", help="restart a video file when it ends")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = no limit)")
    ap.add_argument("--list-classes", action="store_true", help="print the available classes and exit")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.list_classes:
        print("\n".join(f"{i:2d}  {n}" for i, n in enumerate(COCO_NAMES)))
        return 0
    try:
        line = parse_line(a.line)
        zones = [parse_zone(z) for z in a.zone]
        class_ids = resolve_classes(a.classes)
    except ValueError as e:
        print(f"Error: {e}")
        return 2

    names = None if class_ids is None else [COCO_NAMES[i] for i in class_ids]
    try:
        cap = open_capture(a.source)                        # fail fast if the camera / file cannot be opened
    except IOError as e:
        print(f"Error: {e}")
        return 1
    from counter.detector import YoloTracker               # imported late so --help and --list-classes are instant
    print(f"Loading {a.model} (the first run downloads the model)...")
    try:
        tracker = YoloTracker(a.model, conf=a.conf, imgsz=a.imgsz, device=a.device, classes=a.classes)
    except Exception as e:                                  # noqa: BLE001
        cap.release()
        print(f"Error: could not load the model '{a.model}': {e}")
        return 1

    engine = CountingEngine(line=line, flip=a.flip, zones=zones, classes=names, anchor=a.anchor)
    pipe = CounterPipeline(tracker, engine, show_trails=not a.no_trails)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = make_writer(a.save, src_fps, size) if a.save else None
    csv_file = csv_writer = None
    if a.csv:
        csv_file = open(a.csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["time_s", "frame", "in_view", "total_seen", "line_in", "line_out", "per_class_in_view"])

    file_mode, paused, frame_no, last_log, snap = is_file(a.source), False, 0, -1, None
    t0 = time.time()
    print("Running. Press q in the window to quit." if not a.no_show else "Running headless. Press Ctrl+C to stop.")
    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    if file_mode and a.loop and frame_no:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                frame_no += 1
                out, snap = pipe.process(frame)
                if writer:
                    writer.write(out)
                stamp = frame_no / src_fps if file_mode else time.time() - t0
                if csv_writer and int(stamp) != last_log:
                    last_log = int(stamp)
                    csv_writer.writerow([round(stamp, 2), frame_no, snap["total_current"], snap["total_unique"],
                                         snap["line"]["in_total"], snap["line"]["out_total"], json.dumps(snap["current"])])
                if a.max_frames and frame_no >= a.max_frames:
                    break
            if not a.no_show:
                cv2.imshow("Object & Person Counter (q = quit)", out)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    paused = not paused
                elif key == ord("r"):
                    pipe.reset()
                elif key == ord("s"):
                    name = f"screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(name, out)
                    print("Saved", name)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer:
            writer.release()
        if csv_file:
            csv_file.close()
        cv2.destroyAllWindows() if not a.no_show else None

    if snap:
        print(f"\nProcessed {frame_no} frames in {time.time() - t0:.1f}s ({pipe.fps:.1f} FPS)")
        print("Seen at the same time (peak):", snap["peak"] or "nothing")
        print("Unique objects tracked      :", snap["unique"] or "nothing")
        if snap["line"]["enabled"]:
            print(f"Line crossings              : IN {snap['line']['in_total']}  OUT {snap['line']['out_total']}  (net {snap['line']['net']})")
        if a.save:
            print("Saved video:", os.path.abspath(a.save))
    return 0


if __name__ == "__main__":
    sys.exit(main())
