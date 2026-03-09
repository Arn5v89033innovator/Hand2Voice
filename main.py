"""
main.py
-------
Hand2Voice – Real-time ASL alphabet recognition and text-to-speech.

Controls (press while the OpenCV window is focused)
----------------------------------------------------
  SPACE  → manually complete the current word and speak it
  B      → backspace (delete last letter)
  C      → clear everything
  Q/ESC  → quit
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.hand_detector      import HandDetector
from modules.gesture_classifier import GestureClassifier
from modules.speech_engine      import SpeechEngine
from modules.word_builder       import WordBuilder

# ── Configuration ─────────────────────────────────────────────────────────────
BACKEND       = "sklearn"   # "sklearn" or "keras"
CONF_THRESHOLD = 0.65       # minimum confidence to show/use a prediction
CAM_INDEX      = 0          # webcam index (0 = default camera)
WINDOW_TITLE   = "Hand2Voice – ASL Recognition"

# ── Colours (BGR) ─────────────────────────────────────────────────────────────
CLR_GREEN  = (50,  210, 80)
CLR_ORANGE = (0,   165, 255)
CLR_WHITE  = (255, 255, 255)
CLR_GRAY   = (180, 180, 180)
CLR_DARK   = (25,  25,  25)
CLR_RED    = (60,   60, 230)


# ==============================================================================
# UI helpers
# ==============================================================================

def draw_hud(frame: np.ndarray, letter: str, confidence: float,
             state: dict, speaking: bool):
    """Overlay HUD elements on *frame* (in-place)."""
    h, w = frame.shape[:2]

    # ── Top bar ──────────────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 80), CLR_DARK, -1)

    if letter:
        conf_pct = int(confidence * 100)
        bar_w    = int((w - 20) * confidence)
        # confidence progress bar
        cv2.rectangle(frame, (10, 60), (10 + bar_w, 75), CLR_GREEN, -1)
        cv2.rectangle(frame, (10, 60), (w - 10, 75), CLR_GRAY, 1)
        # big letter
        cv2.putText(frame, letter, (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, CLR_GREEN, 3)
        cv2.putText(frame, f"{conf_pct}%", (70, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, CLR_ORANGE, 2)
    else:
        cv2.putText(frame, "No hand detected", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, CLR_GRAY, 2)

    # ── Word bar ─────────────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, h - 100), (w, h), CLR_DARK, -1)

    word_text = f"Word : {state['current_word'] or '…'}"
    cv2.putText(frame, word_text, (10, h - 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, CLR_WHITE, 2)

    sent_text = f"Said : {state['sentence'] or '(nothing yet)'}"
    cv2.putText(frame, sent_text, (10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, CLR_GRAY, 1)

    # ── Speaking indicator ───────────────────────────────────────────────────
    if speaking:
        cv2.putText(frame, "♪ Speaking…", (w - 200, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, CLR_ORANGE, 2)

    # ── Key hints ────────────────────────────────────────────────────────────
    hints = "SPACE=word  B=back  C=clear  Q=quit"
    cv2.putText(frame, hints, (10, h - 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, CLR_GRAY, 1)


# ==============================================================================
# Main loop
# ==============================================================================

def main():
    # ── Initialise components ─────────────────────────────────────────────────
    print("=" * 50)
    print("  Hand2Voice – starting up …")
    print("=" * 50)

    detector = HandDetector(max_hands=1,
                            detection_confidence=0.75,
                            tracking_confidence=0.60)

    clf = GestureClassifier(backend=BACKEND)
    if not clf.load():
        print("\n[ERROR] No trained model found.")
        print("  Please run:  python collect_data.py")
        print("  Then:        python train_model.py")
        print("  Then:        python main.py\n")
        detector.release()
        return

    tts = SpeechEngine(rate=145)
    tts.start()

    def on_word(word: str):
        tts.speak(word)

    builder = WordBuilder(on_word_ready=on_word)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        tts.stop(); detector.release(); return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[Main] Running.  Press Q or ESC to exit.\n")

    # ── Video loop ────────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)              # mirror for natural feel

        # 1. Detect
        features, annotated = detector.process_frame(frame)

        # 2. Classify
        pred_letter = None
        pred_conf   = 0.0
        if features is not None:
            pred_letter, pred_conf = clf.predict(features)
            if pred_conf < CONF_THRESHOLD:
                pred_letter = None              # discard low-confidence

        # 3. Accumulate letters → words
        state = builder.update(pred_letter, pred_conf)

        # 4. Speak letter out loud when a new one is confirmed
        if state["just_added"]:
            tts.speak(state["last_added"])

        # 5. Draw HUD
        draw_hud(annotated, pred_letter or "", pred_conf, state, tts.is_busy())
        cv2.imshow(WINDOW_TITLE, annotated)

        # 6. Key handling
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):               # ESC or Q → quit
            break
        elif key == ord(" "):                   # SPACE → force word complete
            word = builder.current_word
            if word:
                tts.speak(word)
                builder.complete_word()
        elif key in (ord("b"), ord("B")):       # B → backspace
            builder.backspace()
        elif key in (ord("c"), ord("C")):       # C → clear
            builder.clear()

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    tts.stop()
    detector.release()
    print("[Main] Exited cleanly.")


if __name__ == "__main__":
    main()