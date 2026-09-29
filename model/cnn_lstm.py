"""
cnn_lstm.py — CNN + LSTM model architecture for Human Activity Recognition.

Architecture:
    Input (128, 9)
    → Conv1D(64) + BatchNorm + ReLU + MaxPooling + Dropout
    → Conv1D(128) + BatchNorm + ReLU + MaxPooling + Dropout
    → LSTM(128) + Dropout
    → Dense(64, relu) + Dropout
    → Dense(6, softmax)

Usage:
    from model.cnn_lstm import build_model
    model = build_model()
    model.summary()
"""

import logging
from typing import Optional

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers  # type: ignore
from tensorflow.keras.regularizers import l2  # type: ignore

logger = logging.getLogger(__name__)


def build_model(
    timesteps: int = 128,
    channels: int = 9,
    num_classes: int = 6,
    lstm_units: int = 128,
    dense_units: int = 64,
    dropout_rate: float = 0.4,
    learning_rate: float = 1e-3,
    l2_lambda: float = 1e-4,
) -> tf.keras.Model:
    """
    Build and compile the CNN + LSTM model for HAR.

    The model uses two CNN blocks to extract local temporal features, then
    an LSTM layer to capture long-range sequential patterns, followed by a
    fully-connected classification head.

    Args:
        timesteps:     Number of time steps per window (default 128).
        channels:      Number of sensor channels (default 9).
        num_classes:   Number of output activity classes (default 6).
        lstm_units:    Number of LSTM hidden units.
        dense_units:   Number of units in the dense head.
        dropout_rate:  Dropout probability for regularisation.
        learning_rate: Adam optimizer initial learning rate.
        l2_lambda:     L2 weight-decay coefficient.

    Returns:
        Compiled ``tf.keras.Model``.
    """
    inputs = tf.keras.Input(shape=(timesteps, channels), name="sensor_input")

    # ── CNN Block 1 ──────────────────────────────────────────────────────────
    x = layers.Conv1D(
        filters=64,
        kernel_size=3,
        padding="same",
        kernel_regularizer=l2(l2_lambda),
        name="conv1",
    )(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)
    x = layers.Dropout(dropout_rate, name="drop1")(x)

    # ── CNN Block 2 ──────────────────────────────────────────────────────────
    x = layers.Conv1D(
        filters=128,
        kernel_size=3,
        padding="same",
        kernel_regularizer=l2(l2_lambda),
        name="conv2",
    )(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool2")(x)
    x = layers.Dropout(dropout_rate, name="drop2")(x)

    # ── LSTM Block ───────────────────────────────────────────────────────────
    x = layers.LSTM(
        units=lstm_units,
        return_sequences=False,
        kernel_regularizer=l2(l2_lambda),
        name="lstm",
    )(x)
    x = layers.Dropout(dropout_rate, name="drop3")(x)

    # ── Classification Head ──────────────────────────────────────────────────
    x = layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=l2(l2_lambda),
        name="dense1",
    )(x)
    x = layers.Dropout(dropout_rate / 2, name="drop4")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    # ── Build & Compile ──────────────────────────────────────────────────────
    model = models.Model(inputs=inputs, outputs=outputs, name="HAR_CNN_LSTM")

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    total_params = model.count_params()
    logger.info("Model built — total parameters: %s", f"{total_params:,}")
    return model


def get_model_config(
    timesteps: int = 128,
    channels: int = 9,
    num_classes: int = 6,
    lstm_units: int = 128,
    dense_units: int = 64,
    dropout_rate: float = 0.4,
    learning_rate: float = 1e-3,
    l2_lambda: float = 1e-4,
) -> dict:
    """
    Return the model hyperparameter configuration as a serialisable dictionary.

    Useful for saving ``model_config.json`` alongside the trained weights.
    """
    return {
        "architecture": "CNN + LSTM",
        "input_shape": [timesteps, channels],
        "num_classes": num_classes,
        "cnn_filters": [64, 128],
        "cnn_kernel_size": 3,
        "lstm_units": lstm_units,
        "dense_units": dense_units,
        "dropout_rate": dropout_rate,
        "learning_rate": learning_rate,
        "l2_lambda": l2_lambda,
        "optimizer": "Adam",
        "loss": "categorical_crossentropy",
    }
