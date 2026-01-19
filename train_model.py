# backend/train_model.py

import os, json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

BASE_DIR = "backend/dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")

OUT_DIR = "backend/saved_model"
os.makedirs(OUT_DIR, exist_ok=True)

OUTPUT_MODEL = os.path.join(OUT_DIR, "model.h5")
LABEL_MAP_FILE = os.path.join(OUT_DIR, "label_map.json")
THRESH_FILE = os.path.join(OUT_DIR, "detector_threshold.txt")

IMG_SIZE = 224
BATCH = 16
EPOCHS_FROZEN = 6
EPOCHS_FINE = 6


# -------------------------------
# Build Model (3-class classifier)
# -------------------------------
def build_model(input_shape=(224, 224, 3), num_classes=3):
    inp = Input(shape=input_shape)
    base = MobileNetV2(include_top=False, weights="imagenet", input_tensor=inp)
    base.trainable = False  # freeze initial

    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation="softmax")(x)  # 3 classes

    return Model(inputs=inp, outputs=out)


if __name__ == "__main__":

    # Data generators
    train_gen = ImageDataGenerator(
        rescale=1/255.0,
        rotation_range=15,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    val_gen = ImageDataGenerator(rescale=1/255.0)

    train_data = train_gen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH,
        class_mode="categorical",
        shuffle=True
    )

    val_data = val_gen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH,
        class_mode="categorical",
        shuffle=False
    )

    # Create + save label mapping (index -> class name)
    class_map = {v: k for k, v in train_data.class_indices.items()}
    print("Class map:", class_map)

    with open(LABEL_MAP_FILE, "w") as f:
        json.dump({str(k): v for k, v in class_map.items()}, f)

    # Build model
    model = build_model(num_classes=len(class_map))
    model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

    # Callbacks
    ckpt = ModelCheckpoint(os.path.join(OUT_DIR, "best_model.h5"), save_best_only=True,
                           monitor="val_loss", verbose=1)

    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
    early_stop = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1)

    # ------------------------
    # TRAINING STAGE 1: frozen
    ------------------------
    print("\n=== STAGE 1: Training with frozen base ===\n")
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS_FROZEN,
        callbacks=[ckpt, reduce_lr, early_stop]
    )

    # ------------------------
    # TRAINING STAGE 2: Fine tuning
    ------------------------
    print("\n=== STAGE 2: Fine-Tuning MobileNetV2 ===\n")

    base = model.layers[1]  # MobileNetV2
    for layer in base.layers[-50:]:
        layer.trainable = True

    model.compile(optimizer=Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"])

    history2 = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS_FINE,
        callbacks=[ckpt, reduce_lr, early_stop]
    )

    # ------------------------
    # SAVE FINAL MODEL
    ------------------------
    model.save(OUTPUT_MODEL)  # IMPORTANT: saves full Keras model
    print("Saved model:", OUTPUT_MODEL)

    # Default threshold for "REAL vs FAKE"
    with open(THRESH_FILE, "w") as f:
        f.write("0.6")

    print("Label map saved:", LABEL_MAP_FILE)
    print("Threshold saved:", THRESH_FILE)
