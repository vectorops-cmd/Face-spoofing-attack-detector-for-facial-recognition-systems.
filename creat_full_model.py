# tools/create_full_model.py
import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

OUT_DIR = os.path.join("backend", "saved_model")
OUT_FILE = os.path.join(OUT_DIR, "model.h5")
os.makedirs(OUT_DIR, exist_ok=True)

def build_full_model(input_shape=(224,224,3)):
    inp = Input(shape=input_shape, name="input_1")
    base = MobileNetV2(include_top=False, weights="imagenet", input_tensor=inp, alpha=1.0)
    for l in base.layers:
        l.trainable = False
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu", name="dense")(x)
    x = Dropout(0.3)(x)
    out = Dense(1, activation="sigmoid", name="dense_1")(x)
    model = Model(inputs=inp, outputs=out, name="mobilenetv2_full_for_demo")
    return model

if __name__ == "__main__":
    m = build_full_model()
    m.compile(optimizer=Adam(1e-4), loss="binary_crossentropy")
    print("Saving full model to:", OUT_FILE)
    m.save(OUT_FILE, include_optimizer=False)
    print("Done. Now backend can load the full model file.")
