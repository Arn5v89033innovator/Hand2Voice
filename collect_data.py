"""
collect_data.py
---------------
Interactive tool for building your own ASL landmark dataset.

How to use
----------
1. Run:  python collect_data.py
2. Press the letter key you want to record (A-Z).
3. Hold the corresponding ASL gesture in front of the camera.
4. The script captures SAMPLES_PER_LETTER landmark vectors for that letter.
5. Move on to the next letter.
6. When done, the dataset is saved to  data/asl_dataset.npz

Tips
----
- Vary your hand position, distance and lighting for robustness.
- Aim for at least 200 samples per letter (default is 300).
- You can run the script multiple times – existing data is preserved
  and new samples are appended.
"""

import os
import sys
import cv2
import numpy as np

# Allow importing from parent directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.hand_detector import HandDetector
from modules.gesture_classifier import LABELS

# ── Configuration ─────────────────────────────────────────────────────────────
SAMPLES_PER_LETTER = 300
DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
DATA_FILE  = os.path.join(DATA_DIR, "asl_dataset.npz")

# ── Colours (BGR) ─────────────────────────────────────────────────────────────
CLR_GREEN  = (0, 220, 80)
CLR_RED    = (0, 60, 230)
CLR_WHITE  = (255, 255, 255)
CLR_YELLOW = (0, 200, 200)
CLR_BG     = (30, 30, 30)


def load_existing() -> tuple[list, list]:
    if os.path.exists(DATA_FILE):
        data   = np.load(DATA_FILE)
        X_list = list(data["X"])
        y_list = list(data["y"])
        print(f"[Data] Loaded {len(X_list)} existing samples.")
        return X_list, y_list
    return [], []


def save_data(X_list: list, y_list: list):
    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez(DATA_FILE, X=np.array(X_list), y=np.array(y_list))
    print(f"[Data] Saved {len(X_list)} samples → {DATA_FILE}")


def draw_ui(frame, letter: str, collected: int, total: int,
            recording: bool, message: str):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), CLR_BG, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    status = "● REC" if recording else "READY"
    colour = CLR_RED if recording else CLR_GREEN
    cv2.putText(frame, f"{status}  Letter: {letter}  Samples: {collected}/{total}",
                (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)
    cv2.putText(frame, message,
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, CLR_YELLOW, 1)


def main():
    detector = HandDetector()
    cap      = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam."); return

    X_list, y_list = load_existing()

    current_letter = "A"
    recording      = False
    collected_now  = 0
    message        = "Press a letter key (A-Z) to start recording. Press 1 to quit."

    print("\n=== Hand2Voice – Data Collector ===")
    print(f"Target: {SAMPLES_PER_LETTER} samples per letter.\n")
    print("Press 1 to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        features, annotated = detector.process_frame(frame)

        if recording and features is not None:
            label_idx = LABELS.index(current_letter)
            X_list.append(features)
            y_list.append(label_idx)
            collected_now += 1
            message = f"Capturing '{current_letter}'… keep the gesture steady."

            if collected_now >= SAMPLES_PER_LETTER:
                recording     = False
                collected_now = 0
                save_data(X_list, y_list)
                message = f"✓ '{current_letter}' done! Press another letter key."

        draw_ui(annotated, current_letter, collected_now,
                SAMPLES_PER_LETTER, recording, message)
        cv2.imshow("Hand2Voice – Data Collector", annotated)

        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord("1"):        # ESC / 1 → quit
            break
        elif key == ord(" "):                   # SPACE → save & pause
            save_data(X_list, y_list)
            recording = False
        elif 0 <= key <= 255 and chr(key).upper() in LABELS:   # A-Z → start recording
            current_letter = chr(key).upper()
            recording      = True
            collected_now  = 0
            message = f"Recording '{current_letter}'… hold the ASL gesture!"
            print(f"[Collector] Recording letter: {current_letter}")

    save_data(X_list, y_list)
    cap.release()
    cv2.destroyAllWindows()
    detector.release()
    print("\n[Collector] Session ended. Dataset saved.")


if __name__ == "__main__":
    main()
