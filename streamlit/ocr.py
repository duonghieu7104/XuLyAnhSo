#!/usr/bin/env python3
"""
ocr.py - A simple handwriting OCR script using a trained CNN+BiLSTM model.

Usage:
    py ocr.py --model best_model.h5 --image "C:/path/to/handwritten.png"
"""

import argparse
import cv2
import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import (
    Input, Conv2D, BatchNormalization, Activation,
    MaxPooling2D, Dropout, Reshape, Dense,
    Bidirectional, LSTM
)

# --- 1) Constants ---
ALPHABETS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
NUM_OF_TIMESTAMPS = 64
NUM_CHARS = len(ALPHABETS) + 1  # +1 for CTC blank

# --- 2) Preprocess image to shape (1,256,64,1) ---
def preprocess_image_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image at {path}")
    h, w = img.shape
    canvas = np.ones((64, 256), dtype=np.uint8) * 255
    if w > 256:
        img = img[:, :256]
    if h > 64:
        img = img[:64, :]
    canvas[: img.shape[0], : img.shape[1]] = img
    canvas = cv2.rotate(canvas, cv2.ROTATE_90_CLOCKWISE)
    canvas = canvas.astype(np.float32) / 255.0
    return canvas.reshape(1, 256, 64, 1)

# --- 3) Build inference model (input → softmax output) ---
def build_inference_model():
    inp = Input(shape=(256,64,1), name='input')
    x = Conv2D(32, (3,3), padding='same', kernel_initializer='he_normal')(inp)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = MaxPooling2D((2,2))(x)

    x = Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = MaxPooling2D((1,2))(x)
    x = Dropout(0.3)(x)

    x = Reshape(target_shape=(NUM_OF_TIMESTAMPS, -1))(x)  # (None, 64, features)
    x = Dense(64, activation='relu', kernel_initializer='he_normal')(x)

    x = Bidirectional(LSTM(256, return_sequences=True), name='lstm1')(x)
    x = Bidirectional(LSTM(256, return_sequences=True), name='lstm2')(x)

    x = Dense(NUM_CHARS, kernel_initializer='he_normal')(x)
    y_pred = Activation('softmax', name='softmax')(x)

    return Model(inputs=inp, outputs=y_pred)

# --- 4) Decode CTC output ---
def ctc_decode_predictions(preds):
    input_len = np.ones(preds.shape[0]) * preds.shape[1]
    decoded, _ = K.ctc_decode(preds, input_length=input_len, greedy=True)
    return K.get_value(decoded[0])

# --- 5) Map numeric sequence → text ---
def num_to_text(num_seq):
    text = ""
    for idx in num_seq:
        if idx < 0 or idx >= len(ALPHABETS):
            break
        text += ALPHABETS[idx]
    return text

# --- 6) Main ---
def main(args):
    # 6.1 Build model and load weights
    print("Building inference model...")
    model = build_inference_model()
    print(f"Loading weights from {args.model} ...")
    model.load_weights(args.model)

    # 6.2 Preprocess image
    print(f"Preprocessing image {args.image} ...")
    x = preprocess_image_gray(args.image)

    # 6.3 Predict & decode
    print("Predicting ...")
    preds = model.predict(x)
    seq = ctc_decode_predictions(preds)[0]
    text = num_to_text(seq)

    # 6.4 Output
    print("\n==== Recognized Text ====")
    print(text)
    print("==========================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Handwriting OCR: load image and output recognized text."
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to the .h5 weights file."
    )
    parser.add_argument(
        "--image", required=True,
        help="Path to the input handwriting image."
    )
    args = parser.parse_args()
    main(args)
