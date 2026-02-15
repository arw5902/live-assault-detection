import numpy as np
from ultralytics import YOLO
from .config import Config

class PoseDetector:
    def __init__(self, cfg: Config = None):
        if cfg is None:
            cfg = Config()
        self.cfg = cfg
        self.model = YOLO(cfg.yolo_pt_path)

    def infer(self, frame):
        r = self.model(frame, conf=self.cfg.yolo_conf, iou=self.cfg.yolo_iou, verbose=False)[0]
        if len(r.boxes) == 0:
            return None

        # pick largest person
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        kps = r.keypoints.xy.cpu().numpy()      # (N,17,2)
        kp_conf = r.keypoints.conf.cpu().numpy()# (N,17)

        i = boxes[:,2]-boxes[:,0]
        j = boxes[:,3]-boxes[:,1]
        idx = (i*j).argmax()

        kp = kps[idx]
        kc = kp_conf[idx]
        kps17 = np.concatenate([kp, kc[:,None]], axis=1)

        return {
            "bbox": boxes[idx].tolist(),
            "det_conf": float(scores[idx]),
            "kps": kps17
        }
