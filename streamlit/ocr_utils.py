import cv2
import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import (
    Input, Conv2D, BatchNormalization, Activation,
    MaxPooling2D, Dropout, Reshape, Dense,
    Bidirectional, LSTM
)

# Constants
ALPHABETS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
NUM_OF_TIMESTAMPS = 64
NUM_CHARS = len(ALPHABETS) + 1  # +1 for CTC blank

# Preprocess image to shape (1,256,64,1)
def preprocess_image_gray(path_or_array):
    if isinstance(path_or_array, str):
        img = cv2.imread(path_or_array, cv2.IMREAD_GRAYSCALE)
    else:
        img = path_or_array
    if img is None:
        raise ValueError("Cannot read image for OCR")
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

# Build inference model (input → softmax)
def build_inference_model(weights_path):
    inp = Input(shape=(256,64,1), name='input')
    x = Conv2D(32, (3,3), padding='same', kernel_initializer='he_normal')(inp)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = MaxPooling2D((2,2))(x)

    x = Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = MaxPooling2D((2,2))(x); x = Dropout(0.3)(x)

    x = Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = MaxPooling2D((1,2))(x); x = Dropout(0.3)(x)

    x = Reshape((NUM_OF_TIMESTAMPS, -1))(x)
    x = Dense(64, activation='relu', kernel_initializer='he_normal')(x)

    x = Bidirectional(LSTM(256, return_sequences=True), name='lstm1')(x)
    x = Bidirectional(LSTM(256, return_sequences=True), name='lstm2')(x)

    x = Dense(NUM_CHARS, kernel_initializer='he_normal')(x)
    y_pred = Activation('softmax', name='softmax')(x)

    model = Model(inputs=inp, outputs=y_pred)
    model.load_weights(weights_path)
    return model

# Decode CTC output
def ctc_decode(preds):
    input_len = np.ones(preds.shape[0]) * preds.shape[1]
    decoded, _ = K.ctc_decode(preds, input_length=input_len, greedy=True)
    return K.get_value(decoded[0])

# Map numeric sequence → text
def num_to_text(seq):
    text = ''
    for idx in seq:
        if idx < 0 or idx >= len(ALPHABETS):
            break
        text += ALPHABETS[idx]
    return text
