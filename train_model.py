"""
train_model.py
--------------
Load the collected landmark dataset and train the ASL classifier.

Usage
-----
  python train_model.py                  # trains sklearn RandomForest (default)
  python train_model.py --backend keras  # trains a small Dense neural network

The trained model is saved to  models/asl_classifier.pkl  (or .keras).
A quick evaluation report is printed to the console after training.
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.gesture_classifier import GestureClassifier, LABELS

DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "asl_dataset.npz")


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_dataset():
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: Dataset not found at {DATA_FILE}")
        print("Run  python collect_data.py  first to gather training samples.")
        sys.exit(1)

    data = np.load(DATA_FILE)
    X, y = data["X"], data["y"]
    print(f"[Train] Dataset loaded: {X.shape[0]} samples, {len(set(y))} classes.")
    return X, y


def class_report(clf, X_test, y_test):
    """Print per-class accuracy using sklearn's classification_report."""
    from sklearn.metrics import classification_report, accuracy_score

    if clf.backend == "sklearn":
        y_pred = clf.model.predict(X_test)
    else:
        import numpy as _np
        proba  = clf.model.predict(X_test, verbose=0)
        y_pred = _np.argmax(proba, axis=1)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n[Train] Test accuracy: {acc * 100:.2f}%\n")
    print(classification_report(
        y_test, y_pred,
        target_names=LABELS,
        zero_division=0,
    ))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train the Hand2Voice ASL classifier.")
    parser.add_argument(
        "--backend", choices=["sklearn", "keras"], default="sklearn",
        help="ML backend to use (default: sklearn)"
    )
    parser.add_argument(
        "--epochs", type=int, default=60,
        help="Training epochs (Keras only, default: 60)"
    )
    args = parser.parse_args()

    # 1. Load data
    X, y = load_dataset()

    # 2. Shuffle and split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[Train] Train: {len(X_train)}  Test: {len(X_test)}")

    # 3. Optional: scale features for Keras
    if args.backend == "keras":
        from sklearn.preprocessing import StandardScaler
        import pickle, os
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test  = scaler.transform(X_test)
        scaler_path = os.path.join(os.path.dirname(__file__), "models", "scaler.pkl")
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        print(f"[Train] Scaler saved → {scaler_path}")

    # 4. Build and train
    clf = GestureClassifier(backend=args.backend)
    clf.build()
    print(f"[Train] Training {args.backend} model…")
    clf.train(X_train, y_train, epochs=args.epochs)

    # 5. Evaluate
    class_report(clf, X_test, y_test)

    # 6. Save
    clf.save()
    print("\n[Train] Done! You can now run:  python main.py")


if __name__ == "__main__":
    main()
