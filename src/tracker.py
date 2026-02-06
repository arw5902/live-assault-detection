from typing import Optional, Tuple

class SingleTargetTracker:
    """
    Minimal tracker: maintain last bbox and a track_age.
    Replace with SORT/ByteTrack if desired.
    """
    def __init__(self):
        self.last_bbox = None
        self.track_age = 0
        self.lost = 0

    def update(self, bbox) -> Tuple[Optional[list], int, int]:
        if bbox is None:
            self.lost += 1
            return self.last_bbox, self.track_age, self.lost
        self.last_bbox = bbox
        self.track_age += 1
        self.lost = 0
        return self.last_bbox, self.track_age, self.lost
