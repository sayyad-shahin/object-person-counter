"""Opening webcams, video files and network streams."""
import sys
import cv2


def parse_source(src):
    s = str(src).strip()
    return int(s) if s.isdigit() else s


def open_capture(src):
    src = parse_source(src)
    cap = None
    if isinstance(src, int) and sys.platform.startswith("win"):
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)       # DirectShow is the most reliable backend for webcams on Windows
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        hint = "Check the camera number (0, 1, 2...) and that no other program is using it." if isinstance(src, int) \
            else "Check that the file path or stream URL is correct."
        raise IOError(f"Could not open video source '{src}'. {hint}")
    return cap


def is_file(src):
    return not isinstance(parse_source(src), int) and "://" not in str(src)


def make_writer(path, fps, size):
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps if fps and fps > 1 else 25.0, size)
    if not writer.isOpened():
        raise IOError(f"Could not create output video '{path}'.")
    return writer
