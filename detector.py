# backend/detector.py
import os
import traceback
import json
import numpy as np
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model

HERE = os.path.dirname(__file__)
DEFAULT_MODEL_PATH = os.path.join(HERE, "saved_model", "model.h5")
THRESH_FILE = os.path.join(HERE, "saved_model", "detector_threshold.txt")
LABEL_MAP_JSON = os.path.join(HERE, "saved_model", "label_map.json")

def read_threshold(default=0.5):
    try:
        if os.path.exists(THRESH_FILE):
            with open(THRESH_FILE, "r") as f:
                return float(f.read().strip())
    except Exception:
        pass
    return default

def build_mobilenetv2_binary(input_shape=(224, 224, 3)):
    """
    Rebuild MobileNetV2 backbone with a 2-class softmax head:
    class 0 -> fake
    class 1 -> real
    """
    inp = Input(shape=input_shape, name="input_1")
    base = MobileNetV2(include_top=False, weights="imagenet", input_tensor=inp, alpha=1.0)
    base.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu", name="dense")(x)
    x = Dropout(0.3)(x)
    # 2-class softmax head (matches typical training setups)
    out = Dense(2, activation="softmax", name="dense_1")(x)

    model = Model(inputs=inp, outputs=out, name="mobilenetv2_binary_full")
    return model


def load_label_map():
    if os.path.exists(LABEL_MAP_JSON):
        try:
            with open(LABEL_MAP_JSON, "r") as f:
                lm = json.load(f)
                # expect {"0":"fake","1":"real"} or similar
                return {int(k): v for k, v in lm.items()}
        except Exception:
            print("⚠️ Failed to parse label_map.json")
    return None

class LiveGuardDetector:
    def __init__(self, model_path=None, threshold=None, flip_labels=False):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self.threshold = threshold if threshold is not None else read_threshold(0.5)
        self.flip_labels = bool(flip_labels)
        self.label_map = load_label_map()
        self.model = None

        # 1) Try load_model (full model)
        try:
            print(f"🔁 Trying load_model('{self.model_path}')")
            self.model = load_model(self.model_path, compile=False)
            print("✅ load_model succeeded (full model).")
        except Exception as e:
            print("⚠️ load_model() failed:", e)
            traceback.print_exc(limit=1)
            # 2) Try to rebuild MobileNetV2 and load weights
            try:
                print("➡️ Rebuilding MobileNetV2 architecture and trying load_weights(by_name=True)")
                candidate = build_mobilenetv2_binary()
                candidate.compile(optimizer="adam", loss="binary_crossentropy")
                candidate.load_weights(self.model_path, by_name=True)
                self.model = candidate
                print("✅ Loaded weights into MobileNetV2 (by_name=True).")
            except Exception as e2:
                print("⚠️ load_weights(by_name=True) failed:", e2)
                traceback.print_exc(limit=1)
                try:
                    print("➡️ Trying candidate.load_weights(model_path) (direct load)")
                    candidate = build_mobilenetv2_binary()
                    candidate.compile(optimizer="adam", loss="binary_crossentropy")
                    candidate.load_weights(self.model_path)  # may raise if shapes mismatch
                    self.model = candidate
                    print("✅ Loaded weights into MobileNetV2 (direct load).")
                except Exception as e3:
                    print("❌ All attempts to load weights failed. See traces above.")
                    traceback.print_exc(limit=1)
                    raise RuntimeError(f"Failed to load or initialize model from {self.model_path}") from e3

        # Warm-up predict to avoid first-call lag
        try:
            shape = self.model.input_shape
            h = shape[1] or 224
            w = shape[2] or 224
            c = shape[3] or 3
            dummy = np.zeros((1, h, w, c), dtype="float32")
            _ = self.model.predict(dummy, verbose=0)
        except Exception:
            pass

        print(f"📦 Detector initialized. threshold={self.threshold} flip_labels={self.flip_labels} label_map={self.label_map}")

    def preprocess_pil(self, pil_img, target_size=(224,224)):
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        img = pil_img.resize(target_size, Image.BILINEAR)
        arr = np.asarray(img).astype("float32") / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        arr = np.expand_dims(arr, 0)
        return arr

    def _map_index(self, idx):
        if self.label_map:
            return self.label_map.get(int(idx), "real" if idx==1 else "fake")
        # default mapping
        label = "real" if int(idx)==1 else "fake"
        if self.flip_labels:
            label = "real" if label=="fake" else "fake"
        return label

    def predict_pil(self, pil_img):
        x = self.preprocess_pil(pil_img)
        preds = self.model.predict(x, verbose=0)
        p_real = None
        try:
            p = preds[0] if isinstance(preds, (list, tuple)) else preds
            p = np.asarray(p)
            if p.ndim == 2 and p.shape[-1] == 2:
                # two-class softmax — take index for "real"
                real_idx = 1
                if self.label_map:
                    inv = {v:k for k,v in self.label_map.items()}
                    real_idx = inv.get("real", 1)
                p_real = float(p[0, int(real_idx)])
            else:
                p_real = float(p.reshape(-1)[0])
        except Exception:
            p_real = float(np.squeeze(preds))

        p_real = max(0.0, min(1.0, p_real))
        p_fake = 1.0 - p_real

        if p_real >= self.threshold:
            pred_index = 1
            label = self._map_index(pred_index)
            confidence = p_real
        else:
            pred_index = 0
            label = self._map_index(pred_index)
            confidence = p_fake

        print(f"🔍 p_real={p_real:.4f} threshold={self.threshold} => {label} (conf={confidence:.4f})")
        return {
            "prediction": label,
            "confidence": round(float(confidence), 4),
            "attack_type": "unknown",
            "prob_real": round(float(p_real), 4),
            "prob_fake": round(float(p_fake), 4)
        }
