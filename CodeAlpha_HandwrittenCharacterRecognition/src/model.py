"""
============================================================
Model Architecture
Handwritten Character Recognition - CNN (Deep Learning)
Author : Sivanesh K
CodeAlpha Machine Learning Internship - Task 2

A single, well-regularized CNN. Deliberately not using
EfficientNet/ResNet50/ViT-scale backbones here: those are
designed for large, RGB, high-resolution natural images
(224x224+, 3-channel). This task's input is 28x28 grayscale -
applying a backbone built for ImageNet-scale photos wouldn't
add capability, it would only add parameters that can't be
justified by the input's information content, and would slow
down CPU training for no accuracy benefit. The CNN below is
sized appropriately for the actual input while still using
modern practices (BatchNorm, L2 regularization, Dropout,
Global Average Pooling) that were missing or misapplied in
earlier drafts of this project.
============================================================
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def build_cnn(input_shape=(28, 28, 1), num_classes=62) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # Block 1
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Classifier head
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ], name="digit_char_cnn")

    return model


def compile_model(model, learning_rate=1e-3):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model
