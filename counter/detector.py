"""YOLOv8 detection + ByteTrack tracking through the Ultralytics library."""
from .coco import resolve_classes
from .engine import Detection


class YoloTracker:
    """Wraps an Ultralytics YOLO model. Each call to detect() returns Detection objects with persistent track ids."""

    def __init__(self, model="yolov8n.pt", conf=0.35, iou=0.5, imgsz=640, device=None, classes=None,
                 tracker="bytetrack.yaml"):
        try:
            from ultralytics import YOLO          # imported here so the rest of the project works without it
        except ImportError as e:
            raise ImportError("Ultralytics is not installed. Run: pip install -r requirements.txt") from e
        self.model = YOLO(model)                   # downloads the weights automatically on first use
        self.names = self.model.names              # {0: 'person', 1: 'bicycle', ...}
        self.iou, self.imgsz, self.device, self.tracker_cfg = iou, imgsz, device, tracker
        self.configure(conf=conf, classes=classes)

    def configure(self, conf=None, classes=None):
        if conf is not None:
            self.conf = float(conf)
        self.class_ids = resolve_classes(classes, self.names)

    def reset(self):
        """Forget all tracks (use when a new video starts)."""
        self.model.predictor = None

    def detect(self, frame):
        res = self.model.track(frame, persist=True, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                               classes=self.class_ids, tracker=self.tracker_cfg, device=self.device,
                               verbose=False)[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int).tolist() if boxes.id is not None else [None] * len(cls)
        names = res.names
        return [Detection(int(c), str(names[int(c)]), float(p), tuple(float(v) for v in b), t)
                for b, c, p, t in zip(xyxy, cls, conf, ids)]
