# tools/create_full_model.py
"""
Builds a MobileNetV2-based binary classifier, attaches a small head,
and saves the full model (architecture + weights) into backend/saved_model/model.h5.

This produces a loadable .h5 that your Flask backend can call with load_model().
Note: the head will be randomly initialized (unless you fine-tune later).
"""

import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

OUT_PATH = os.path.join("backend", "saved_model")
OUT_FILE = os.path.join(OUT_PATH, "model.h5")
os.makedirs(OUT_PATH, exist_ok=True)

def build_full_model(input_shape=(224,224,3), freeze_base=True):
    inp = Input(shape=input_shape, name="input_1")
    base = MobileNetV2(include_top=False, weights="imagenet", input_tensor=inp, alpha=1.0)
    if freeze_base:
        for l in base.layers:
            l.trainable = False

    x = base.output
    x = GlobalAveragePooling2D(name="gap")(x)
    x = Dropout(0.4, name="dropout")(x)
    x = Dense(128, activation="relu", name="dense")(x)
    x = Dropout(0.3, name="dropout_1")(x)
    out = Dense(1, activation="sigmoid", name="dense_1")(x)

    model = Model(inputs=inp, outputs=out, name="mobilenetv2_binary_full")
    return model

if __name__ == "__main__":
    model = build_full_model()
    model.compile(optimizer=Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])
    print("Model summary:")
    model.summary()
    print(f"Saving full model to: {OUT_FILE}")
    model.save(OUT_FILE, include_optimizer=False)
    print("Saved. You can now run your backend and it will load this model.")
