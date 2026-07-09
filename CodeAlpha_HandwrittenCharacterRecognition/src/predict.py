"""
============================================================
Prediction Script
Handwritten Character Recognition - CNN (Deep Learning)
Author : Sivanesh K
CodeAlpha Machine Learning Internship - Task 2

Preprocessing mirrors MNIST/EMNIST conventions:
  1. Grayscale, threshold to isolate dark strokes on light background
  2. Crop to bounding box of the character
  3. Resize preserving aspect ratio (longest side = 20px)
  4. Center on a blank 28x28 canvas

Run:
    python src/predict.py
============================================================
"""

import sys
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_char_cnn.keras"
CLASS_MAPPING_PATH = BASE_DIR / "models" / "class_mapping.json"


def load_artifacts():
    if not MODEL_PATH.exists():
        print("=" * 70)
        print("MODEL NOT FOUND")
        print("=" * 70)
        print("\nRun src/train.py first (or the notebook) to produce:")
        print("  models/best_char_cnn.keras")
        print("  models/class_mapping.json\n")
        sys.exit(1)

    model = tf.keras.models.load_model(MODEL_PATH)

    if CLASS_MAPPING_PATH.exists():
        with open(CLASS_MAPPING_PATH) as f:
            raw_mapping = json.load(f)
        class_mapping = {int(k): v for k, v in raw_mapping.items()}
    else:
        from data_utils import CLASS_MAPPING as class_mapping

    print("Model and class mapping loaded successfully.\n")
    return model, class_mapping


def preprocess_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image at: {image_path}")

    _, thresh = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(
            "No character contour found in the image. "
            "Make sure the character is dark strokes on a light background."
        )

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    char_crop = thresh[y:y + h, x:x + w]

    size = 20
    height, width = char_crop.shape
    if height > width:
        new_height = size
        new_width = max(1, int(width * size / height))
    else:
        new_width = size
        new_height = max(1, int(height * size / width))

    char_crop = cv2.resize(char_crop, (new_width, new_height))

    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_width) // 2
    y_offset = (28 - new_height) // 2
    canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = char_crop

    return canvas


def predict_character(canvas: np.ndarray, model, class_mapping):
    x = canvas.astype("float32") / 255.0
    x = x.reshape(1, 28, 28, 1)

    probabilities = model.predict(x, verbose=0)[0]
    predicted_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_idx])
    predicted_char = class_mapping.get(predicted_idx, f"class_{predicted_idx}")

    top5_idx = np.argsort(probabilities)[-5:][::-1]
    top5 = [(class_mapping.get(int(i), f"class_{i}"), float(probabilities[i])) for i in top5_idx]

    return predicted_char, confidence, top5


def main():
    model, class_mapping = load_artifacts()

    print("=" * 50)
    print("Handwritten Character Recognition (CNN)")
    print("=" * 50)

    image_path = input("\nEnter image path: ").strip()

    try:
        canvas = preprocess_image(image_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"\nError: {e}")
        sys.exit(1)

    predicted_char, confidence, top5 = predict_character(canvas, model, class_mapping)

    print(f"\nPredicted Character : {predicted_char}")
    print(f"Confidence           : {confidence * 100:.2f}%")
    print("\nTop 5 predictions:")
    for char, prob in top5:
        print(f"  {char}  -  {prob * 100:.2f}%")

    plt.figure(figsize=(4, 4))
    plt.imshow(canvas, cmap="gray")
    plt.title(f"Prediction: {predicted_char}\nConfidence: {confidence * 100:.2f}%")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
