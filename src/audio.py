"""
Audio threat detection using PANNs (Pre-trained Audio Neural Networks).

Extracts audio from video files via ffmpeg, classifies sounds using a
pre-trained AudioSet model, and produces a threat score based on the
presence of threat-related audio events (screaming, impacts, gunshots, etc.).

Dependencies (optional — audio is gracefully disabled if missing):
    pip install panns-inference librosa soundfile

The visual pipeline works identically with or without audio installed.
Audio only *boosts* a borderline visual score; it never triggers alone.
"""

import subprocess
import numpy as np
import os
from typing import Dict, Optional, Tuple

# ── AudioSet class indices for threat-related sounds ─────────────────────────
# Weights reflect how strongly each sound indicates a physical threat. They are hand-picked heuristics.
# These are multiplied by the model's confidence → higher weight = more influence.
THREAT_CLASSES = {
    # Vocal aggression
    8:   ("Shout",              0.60),
    11:  ("Yell",               0.60),
    12:  ("Battle cry",         0.85),
    14:  ("Screaming",          0.80),
    # Physical impact
    460: ("Thump, thud",        0.50),
    467: ("Slap, smack",        0.70),
    468: ("Whack, thwack",      0.70),
    469: ("Smash, crash",       0.55),
    470: ("Breaking",           0.50),
    443: ("Shatter",            0.55),
    # Weapons
    427: ("Gunshot, gunfire",    0.95),
    426: ("Explosion",          0.80),
    # Exertion (lower weight — common in non-threat scenarios too)
    38:  ("Groan",              0.25),
    39:  ("Grunt",              0.30),
    # Movement (very low weight — only meaningful in combination)
    51:  ("Run",                0.15),
}

PANNS_SAMPLE_RATE = 32000  # PANNs expects 32 kHz mono


class AudioThreatDetector:
    """Pre-trained audio classifier for threat-related sounds.

    Lazy-loads the PANNs model on first use.  If panns_inference is not
    installed, all scoring methods return 0.0 (audio disabled).

    Parameters
    ----------
    device : str
        "cpu" or "cuda".  Pi5 always uses "cpu".
    checkpoint_path : str or None
        Path to a PANNs checkpoint (.pth).  None = auto-download default Cnn14.
    """

    def __init__(self, device: str = "cpu", checkpoint_path: Optional[str] = None):
        self._device = device
        self._checkpoint_path = checkpoint_path
        self._model = None
        self._available = None  # None = not yet checked

    @property
    def available(self) -> bool:
        """True if PANNs model loaded successfully."""
        if self._available is None:
            self._load_model()
        return self._available

    def _load_model(self):
        """Lazy-load PANNs AudioTagging model."""
        try:
            from panns_inference import AudioTagging
            self._model = AudioTagging(
                checkpoint_path=self._checkpoint_path,
                device=self._device,
            )
            self._available = True
            print(f"Audio threat detector: PANNs loaded on {self._device}")
        except ImportError:
            print("Audio threat detector: DISABLED (panns_inference not installed). "
                  "Install with: pip install panns-inference librosa soundfile")
            self._available = False
        except Exception as e:
            print(f"Audio threat detector: DISABLED (model load failed: {e})")
            self._available = False

    # ── audio extraction ─────────────────────────────────────────────────────

    @staticmethod
    def extract_audio(video_path: str) -> Optional[np.ndarray]:
        """Extract audio from a video file as float32 array at 32 kHz mono.

        Uses ffmpeg via subprocess — no Python audio library needed.
        Returns None if the video has no audio track or ffmpeg is unavailable.
        """
        if not os.path.isfile(video_path):
            return None
        try:
            cmd = [
                "ffmpeg",
                "-i", video_path,
                "-f", "f32le",            # raw float32 output
                "-acodec", "pcm_f32le",
                "-ac", "1",               # mono
                "-ar", str(PANNS_SAMPLE_RATE),
                "-v", "quiet",
                "pipe:1",
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=120)
            if result.returncode != 0 or len(result.stdout) == 0:
                return None
            audio = np.frombuffer(result.stdout, dtype=np.float32).copy()
            return audio if len(audio) > 0 else None
        except FileNotFoundError:
            # ffmpeg not installed
            return None
        except subprocess.TimeoutExpired:
            return None

    # ── single-window scoring ────────────────────────────────────────────────

    def score_window(self, audio_window: np.ndarray) -> Tuple[float, str]:
        """Classify a single audio window and return a threat score in [0, 1].

        Parameters
        ----------
        audio_window : np.ndarray
            1-D float32 array at 32 kHz mono (typically 1 second = 32000 samples).

        Returns
        -------
        (float, str)
            Threat score (weighted max of threat-class probabilities) and the
            name of the dominant threat class (empty string when score is 0).
        """
        if not self.available:
            return 0.0, ""

        # PANNs expects [batch, samples]
        inp = audio_window[None, :].astype(np.float32)
        clipwise_output, _ = self._model.inference(inp)
        probs = clipwise_output[0]  # shape [527]

        max_threat = 0.0
        top_class  = ""
        for cls_idx, (name, weight) in THREAT_CLASSES.items():
            score = float(probs[cls_idx]) * weight
            if score > max_threat:
                max_threat = score
                top_class  = name
        return min(max_threat, 1.0), top_class

    # ── batch scoring for a full video ───────────────────────────────────────

    def score_video(self, video_path: str, fps: float, frame_stride: int,
                    window_sec: float = 1.0
                    ) -> Dict[int, Tuple[float, str]]:
        """Pre-compute audio threat scores for every strided frame in a video.

        Extracts the full audio track, then slides a window_sec-long window
        centred on each strided frame's timestamp and classifies it.

        Parameters
        ----------
        video_path   : str   — path to the video file
        fps          : float — video frame rate (used to align audio windows)
        frame_stride : int   — only score every frame_stride-th frame
        window_sec   : float — audio window duration in seconds (default 1.0)

        Returns
        -------
        dict mapping frame_idx (int) → (audio_score, class_name).
        Empty dict if audio extraction fails or PANNs is unavailable.
        """
        if not self.available:
            return {}

        audio = self.extract_audio(video_path)
        if audio is None:
            return {}

        window_samples = int(window_sec * PANNS_SAMPLE_RATE)
        total_duration = len(audio) / PANNS_SAMPLE_RATE
        total_frames   = int(total_duration * fps)

        scores = {}
        for frame_idx in range(0, total_frames, frame_stride):
            t_sec = frame_idx / fps
            centre = int(t_sec * PANNS_SAMPLE_RATE)
            start  = max(0, centre - window_samples // 2)
            end    = min(len(audio), start + window_samples)

            if end - start < PANNS_SAMPLE_RATE // 4:  # < 0.25 s — too short
                scores[frame_idx] = (0.0, "")
                continue

            window = audio[start:end]
            # Zero-pad if shorter than window_sec
            if len(window) < window_samples:
                window = np.pad(window, (0, window_samples - len(window)))

            scores[frame_idx] = self.score_window(window)

        return scores


# ── fusion logic ─────────────────────────────────────────────────────────────

def fuse_scores(visual_score: float, audio_score: float,
                audio_thresh: float = 0.3,
                audio_boost_alpha: float = 0.25) -> float:
    """Option B fusion: audio boosts a borderline visual score.

    Audio alone cannot trigger an alert — it only raises the visual score
    when both modalities agree that something threatening is happening.
    This prevents audio-only false positives (crowd noise, TV audio, etc.)
    while letting audio rescue missed visual detections (out-of-frame attacks,
    close-range scuffles where keypoints fail).

    Parameters
    ----------
    visual_score      : float — hazard_ema from the visual GRU (0–1)
    audio_score       : float — threat score from PANNs (0–1)
    audio_thresh      : float — minimum audio_score to activate boost
    audio_boost_alpha : float — strength of the boost (0 = no effect, 1 = full)

    Returns
    -------
    float — fused score in [0, 1], always >= visual_score.
    """
    if audio_score > audio_thresh:
        return visual_score + (1.0 - visual_score) * audio_score * audio_boost_alpha
    return visual_score
