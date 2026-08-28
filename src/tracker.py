from typing import Optional


class SingleTargetTracker:
    """Holds the most recent bounding box so a dropped detection reuses it.

    There is no identity association: the detector already returns a single
    person per frame (largest box).  Replace with SORT or ByteTrack if
    multi-person tracking is added.
    """

    def __init__(self):
        self.last_bbox = None

    def update(self, bbox) -> Optional[list]:
        if bbox is not None:
            self.last_bbox = bbox
        return self.last_bbox
