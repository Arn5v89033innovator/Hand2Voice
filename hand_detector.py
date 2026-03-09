"""
hand_detector.py
----------------
Handles real-time hand detection and landmark extraction using MediaPipe.
Returns normalized 21-landmark coordinates as a flat feature vector.
"""

import cv2
import mediapipe as mp
import numpy as np


class HandDetector:
    """
    Wraps MediaPipe Hands to detect a single hand and extract
    21 3D landmarks, returned as a normalised 63-dimensional vector.
    """

    def __init__(self, max_hands: int = 1, detection_confidence: float = 0.7,
                 tracking_confidence: float = 0.5):
        self.mp_hands = mp.solutions.hands
        self.mp_draw  = mp.solutions.drawing_utils
        self.mp_style = mp.solutions.drawing_styles

        # Initialise MediaPipe Hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray):
        """
        Detect hand landmarks in *frame* (BGR uint8).

        Returns
        -------
        landmarks_norm : np.ndarray | None
            Flat array of shape (63,) – (x, y, z) for each of the 21
            landmarks, normalised so that the wrist sits at the origin
            and the span of the hand equals 1.  Returns None when no
            hand is detected.
        annotated_frame : np.ndarray
            A copy of *frame* with landmarks drawn on it.
        """
        # MediaPipe works on RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True

        annotated = frame.copy()
        landmarks_norm = None

        if results.multi_hand_landmarks:
            hand_lms = results.multi_hand_landmarks[0]          # first hand only

            # Draw skeleton on the annotated frame
            self.mp_draw.draw_landmarks(
                annotated,
                hand_lms,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_style.get_default_hand_landmarks_style(),
                self.mp_style.get_default_hand_connections_style(),
            )

            landmarks_norm = self._extract_features(hand_lms)

        return landmarks_norm, annotated

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_features(self, hand_landmarks) -> np.ndarray:
        """
        Convert raw landmark object → normalised feature vector.

        Normalisation steps
        -------------------
        1. Translate so that landmark 0 (wrist) is the origin.
        2. Scale so that the Euclidean distance from wrist to middle-
           finger MCP (landmark 9) equals 1.  This makes the vector
           invariant to hand size and position in the frame.
        """
        coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
            dtype=np.float32,
        )                                       # shape (21, 3)

        # 1. Translate
        coords -= coords[0]                     # wrist → origin

        # 2. Scale
        scale = np.linalg.norm(coords[9])       # wrist-to-MCP distance
        if scale > 1e-6:
            coords /= scale

        return coords.flatten()                 # shape (63,)

    def release(self):
        """Release MediaPipe resources."""
        self.hands.close()
