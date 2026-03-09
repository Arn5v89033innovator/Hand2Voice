"""
gesture_classifier.py
---------------------
Trains and runs an ASL letter classifier.

Two backends are supported:
  • 'sklearn'  – RandomForest / MLPClassifier (lightweight, fast to train)
  • 'keras'    – Small Dense neural network (better accuracy with more data)

The trained model is saved to disk so it only needs to be trained once.
"""

import os
import pickle
import numpy as np
from typing import Optional

# ── Labels ────────────────────────────────────────────────────────────────────
LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")   # 26 ASL letters
N_CLASSES   = len(LABELS)
N_FEATURES  = 63                              # 21 landmarks × 3 (x, y, z)

# ── Paths ─────────────────────────────────────────────────────────────────────
_DIR         = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR    = os.path.join(_DIR, "..", "models")
SKLEARN_PATH = os.path.join(MODEL_DIR, "asl_classifier.pkl")
KERAS_PATH   = os.path.join(MODEL_DIR, "asl_classifier.keras")


# ==============================================================================
# Classifier wrapper
# ==============================================================================

class GestureClassifier:
    """
    Thin wrapper around either a scikit-learn or Keras model.

    Usage
    -----
    clf = GestureClassifier(backend='sklearn')
    clf.train(X_train, y_train)
    clf.save()
    letter, confidence = clf.predict(feature_vector)
    """

    def __init__(self, backend: str = "sklearn"):
        assert backend in ("sklearn", "keras"), \
            "backend must be 'sklearn' or 'keras'"
        self.backend = backend
        self.model   = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self):
        """Instantiate a fresh (untrained) model."""
        if self.backend == "sklearn":
            self.model = self._build_sklearn()
        else:
            self.model = self._build_keras()
        return self

    def _build_sklearn(self):
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )

    def _build_keras(self):
        import tensorflow as tf
        from tensorflow import keras

        model = keras.Sequential([
            keras.layers.Input(shape=(N_FEATURES,)),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64,  activation="relu"),
            keras.layers.Dense(N_CLASSES, activation="softmax"),
        ], name="asl_classifier")

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, X: np.ndarray, y: np.ndarray,
              epochs: int = 60, validation_split: float = 0.15):
        """
        Train the model.

        Parameters
        ----------
        X : np.ndarray, shape (N, 63)
        y : np.ndarray, shape (N,)  – integer labels 0-25
        """
        if self.model is None:
            self.build()

        if self.backend == "sklearn":
            self.model.fit(X, y)
        else:
            self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=32,
                validation_split=validation_split,
                callbacks=[
                    __import__("tensorflow").keras.callbacks.EarlyStopping(
                        monitor="val_accuracy", patience=8, restore_best_weights=True
                    )
                ],
                verbose=1,
            )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        if self.backend == "sklearn":
            with open(SKLEARN_PATH, "wb") as f:
                pickle.dump(self.model, f)
            print(f"[Classifier] Model saved → {SKLEARN_PATH}")
        else:
            self.model.save(KERAS_PATH)
            print(f"[Classifier] Model saved → {KERAS_PATH}")

    def load(self) -> bool:
        """
        Load a previously saved model.

        Returns True on success, False if no saved model exists.
        """
        if self.backend == "sklearn":
            if not os.path.exists(SKLEARN_PATH):
                return False
            with open(SKLEARN_PATH, "rb") as f:
                self.model = pickle.load(f)
            print("[Classifier] sklearn model loaded.")
            return True
        else:
            import os as _os
            if not _os.path.exists(KERAS_PATH):
                return False
            import tensorflow as tf
            self.model = tf.keras.models.load_model(KERAS_PATH)
            print("[Classifier] Keras model loaded.")
            return True

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, features: np.ndarray):
        """
        Predict the ASL letter from a (63,) feature vector.

        Returns
        -------
        letter     : str   – predicted letter ("A"–"Z")
        confidence : float – probability in [0, 1]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.  Call load() or train() first.")

        x = features.reshape(1, -1)

        if self.backend == "sklearn":
            idx   = int(self.model.predict(x)[0])
            proba = self.model.predict_proba(x)[0][idx]
        else:
            proba_all = self.model.predict(x, verbose=0)[0]
            idx       = int(np.argmax(proba_all))
            proba     = float(proba_all[idx])

        return LABELS[idx], proba
