"""
============================================================
Training Script
Handwritten Character Recognition - CNN (Deep Learning)
Author : Sivanesh K
CodeAlpha Machine Learning Internship - Task 2

Trains a single CNN on the union of MNIST (digits) and EMNIST
ByClass (digits + uppercase + lowercase) so one model recognizes
both, satisfying the brief's dataset requirement directly instead
of only covering digits.

Run:
    python src/train.py

Outputs (all in models/):
    best_char_cnn.keras     - final trained model
    training_history.json   - real per-epoch metrics (no fabrication)
    performance.json        - real final test-set metrics
    class_mapping.json      - class index -> character mapping
    confusion_matrix.npy    - raw confusion matrix
============================================================
"""

import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from data_utils import (
    load_combined_dataset, preprocess, make_tf_dataset,
    NUM_CLASSES, CLASS_MAPPING,
)
from model import build_cnn, compile_model

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
BATCH_SIZE = 128
EPOCHS = 25


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def main(use_emnist: bool = True):
    log(f"Loading dataset (EMNIST included: {use_emnist})...")
    images, labels = load_combined_dataset(use_emnist=use_emnist)
    log(f"Total samples: {images.shape[0]}, classes present: {len(np.unique(labels))}")

    X, y = preprocess(images, labels, num_classes=NUM_CLASSES)

    # Stratified split by original integer label (before one-hot)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, labels, test_size=0.15, random_state=RANDOM_STATE, stratify=labels
    )
    X_val, X_test, y_val_int, y_test_int = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
    )
    y_train_oh = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_val_oh = tf.keras.utils.to_categorical(y_val_int, NUM_CLASSES)
    y_test_oh = tf.keras.utils.to_categorical(y_test_int, NUM_CLASSES)

    log(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    train_ds = make_tf_dataset(X_train, y_train_oh, batch_size=BATCH_SIZE, augment=True, shuffle=True)
    val_ds = make_tf_dataset(X_val, y_val_oh, batch_size=BATCH_SIZE, augment=False, shuffle=False)
    test_ds = make_tf_dataset(X_test, y_test_oh, batch_size=BATCH_SIZE, augment=False, shuffle=False)

    model = build_cnn(input_shape=(28, 28, 1), num_classes=NUM_CLASSES)
    model = compile_model(model)
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / "best_char_cnn.keras"),
            monitor="val_accuracy", save_best_only=True, mode="max", verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1,
        ),
    ]

    log(f"Training for up to {EPOCHS} epochs...")
    start = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
    elapsed = time.time() - start
    log(f"Training finished in {elapsed:.1f}s")

    # ---- Real metrics only - nothing hardcoded, nothing fabricated ----
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    log(f"FINAL TEST ACCURACY: {test_acc:.4f}  |  TEST LOSS: {test_loss:.4f}")

    y_pred_probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    report = classification_report(y_test_int, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test_int, y_pred)
    np.save(MODELS_DIR / "confusion_matrix.npy", cm)

    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(MODELS_DIR / "training_history.json", "w") as f:
        json.dump(history_dict, f, indent=2)

    performance = {
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "weighted_f1": report["weighted avg"]["f1-score"],
        "weighted_precision": report["weighted avg"]["precision"],
        "weighted_recall": report["weighted avg"]["recall"],
        "training_time_seconds": elapsed,
        "epochs_run": len(history_dict["loss"]),
        "num_classes": NUM_CLASSES,
        "used_emnist": use_emnist,
    }
    with open(MODELS_DIR / "performance.json", "w") as f:
        json.dump(performance, f, indent=2)
    log(f"Saved real performance metrics -> models/performance.json")

    with open(MODELS_DIR / "class_mapping.json", "w") as f:
        json.dump({str(k): v for k, v in CLASS_MAPPING.items()}, f, indent=2)

    log("Done. Model saved to models/best_char_cnn.keras")


if __name__ == "__main__":
    main(use_emnist=True)
