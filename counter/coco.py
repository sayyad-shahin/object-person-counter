"""The 80 classes that the standard YOLOv8 models (trained on COCO) can detect."""
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def resolve_classes(items, names=None):
    """Turn class names or numbers (['person', 'car'] or ['0', '2']) into a list of class ids. None = all classes."""
    if not items:
        return None
    names = names or COCO_NAMES
    table = dict(enumerate(names)) if not isinstance(names, dict) else dict(names)
    by_name = {str(v).lower(): int(k) for k, v in table.items()}
    ids = []
    for it in items:
        s = str(it).strip().lower()
        if s in ("all", "*", ""):
            return None
        if s.isdigit() and int(s) in table:
            ids.append(int(s))
        elif s in by_name:
            ids.append(by_name[s])
        else:
            raise ValueError(f"Unknown class '{it}'. Run with --list-classes to see the options.")
    return sorted(set(ids))
