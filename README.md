# ✋ Hand2Voice
**Real-time ASL alphabet recognition → text → speech**

Hand2Voice uses your webcam to detect American Sign Language (ASL) hand gestures for the full A–Z alphabet, assembles the letters into words, and speaks them aloud using offline text-to-speech — helping mute individuals communicate naturally.

---

## Project Structure

```
Hand2Voice/
├── main.py               ← Run this to start the app
├── collect_data.py       ← Step 1: Record your own training data
├── train_model.py        ← Step 2: Train the classifier
├── requirements.txt
│
├── modules/
│   ├── hand_detector.py       # MediaPipe landmark extraction
│   ├── gesture_classifier.py  # ML model (sklearn or Keras)
│   ├── speech_engine.py       # Async TTS via pyttsx3
│   └── word_builder.py        # Letter → word accumulation
│
├── data/
│   └── asl_dataset.npz        # (created by collect_data.py)
│
└── models/
    └── asl_classifier.pkl     # (created by train_model.py)
```

---

## Quick-Start

### 1 – Install dependencies

```bash
# Python 3.9 – 3.11 recommended
pip install -r requirements.txt
```

> **macOS / Linux TTS note:** `pyttsx3` needs an OS speech engine.
> - macOS: built-in `NSSpeechSynthesizer` – no extra install needed.
> - Ubuntu/Debian: `sudo apt install espeak`
> - Windows: built-in SAPI5 – no extra install needed.

---

### 2 – Collect training data

```bash
python collect_data.py
```

An OpenCV window opens showing your webcam feed.

| Action | Effect |
|--------|--------|
| Press `A`–`Z` | Start recording that letter |
| Hold the ASL gesture steady | 300 samples are captured automatically |
| Press `SPACE` | Pause and save progress |
| Press `Q` / `ESC` | Quit and save |

**Tips for a good dataset**
- Collect **300+ samples per letter**.
- Vary hand position, distance from camera and lighting.
- Run multiple sessions to add diversity.

Samples are saved to `data/asl_dataset.npz` and are additive across sessions.

---

### 3 – Train the model

```bash
# Default: fast RandomForest (sklearn)
python train_model.py

# Alternative: small neural network (requires tensorflow)
python train_model.py --backend keras
```

After training you'll see a per-class accuracy report.  
The model is saved to `models/asl_classifier.pkl` (or `.keras`).

---

### 4 – Run Hand2Voice

```bash
python main.py
```

| Key | Action |
|-----|--------|
| `SPACE` | Force the current word to be spoken now |
| `B` | Backspace – delete the last letter |
| `C` | Clear everything |
| `Q` / `ESC` | Quit |

**How word formation works**
1. Hold a gesture steady for ~20 frames → letter confirmed.
2. Remove your hand for ~30 frames → word boundary → word is spoken.
3. Or press `SPACE` to speak the word immediately.

---

## ASL Alphabet Reference

```
A – fist, thumb beside index    N – fist, thumb over middle+ring
B – flat hand, fingers together O – all fingers form a circle
C – curved open hand            P – K pointing downward
D – index up, others form O     Q – G pointing downward
E – fingers bent, thumb tucked  R – crossed index+middle
F – OK sign variant             S – fist, thumb over fingers
G – index+thumb point sideways  T – thumb between index+middle
H – index+middle point sideways U – index+middle up together
I – pinky up                    V – index+middle spread (peace)
J – I + arc motion              W – index+middle+ring spread
K – index+middle up, thumb out  X – index hooked
L – L-shape (index+thumb)       Y – pinky+thumb out
M – fist, thumb under 3 fingers Z – index traces Z (motion)
```

> **Note on J and Z:** These letters require motion in real ASL.
> For this static-landmark system, use the *starting position* of each.

---

## Switching to Keras

```bash
# Install TensorFlow
pip install tensorflow

# Train with Keras backend
python train_model.py --backend keras --epochs 80

# In main.py, change line:
BACKEND = "keras"

# Then run
python main.py
```

Keras generally achieves higher accuracy with 500+ samples per letter.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Webcam not found | Change `CAM_INDEX = 0` in `main.py` to `1` or `2` |
| Low accuracy | Collect more samples; vary lighting and distance |
| No sound | Check system volume; on Linux install `espeak` |
| MediaPipe import error | `pip install mediapipe --upgrade` |
| Slow on laptop | Reduce frame resolution in `main.py` |

---

## Architecture Overview

```
Webcam frame
    │
    ▼
HandDetector (MediaPipe)
    │  21 landmarks (x,y,z) = 63 features
    ▼
GestureClassifier (sklearn / Keras)
    │  predicted letter + confidence
    ▼
WordBuilder
    │  stable letter → word → sentence
    ▼
SpeechEngine (pyttsx3, async thread)
    │  speak(word)
    ▼
Audio output 🔊
```

---

## License

MIT – free to use, modify and distribute.
