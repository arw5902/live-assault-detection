import cv2

def iter_video_frames(path: str, frame_stride: int):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % frame_stride == 0:
            yield idx, frame
        idx += 1
    cap.release()