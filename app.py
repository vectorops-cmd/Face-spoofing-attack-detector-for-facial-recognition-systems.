# backend/app.py — backend API with robust real/fake decision logic
import os, time, traceback
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

# local imports
from models import init_db, db, DetectionLog
from detector import LiveGuardDetector

# -------------------------------------------------
# PATH SETUP
# -------------------------------------------------

BASE = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

MODEL_PATH = os.path.join(BASE, "saved_model", "model.h5")

# -------------------------------------------------
# APP INITIALIZATION
# -------------------------------------------------

app = Flask(__name__)
CORS(app)

app.config["DB_FILE"] = os.path.join(BASE, "db.sqlite")
init_db(app)

# -------------------------------------------------
# SAFE MODEL LOADING (CRITICAL FIX)
# -------------------------------------------------

detector = None
MODEL_LOADED = False

if os.path.exists(MODEL_PATH):
    try:
        detector = LiveGuardDetector(MODEL_PATH, threshold=0.5)
        MODEL_LOADED = True
        print("✅ ML model loaded successfully")
    except Exception:
        print("❌ Model found but failed to load")
        traceback.print_exc()
else:
    print("⚠️ WARNING: model.h5 not found — running in mock mode")

# -------------------------------------------------
# HEALTH CHECK
# -------------------------------------------------

@app.route("/")
def health():
    return jsonify({
        "status": "backend running",
        "model_loaded": MODEL_LOADED
    }), 200

# -------------------------------------------------
# LABEL NORMALIZATION
# -------------------------------------------------

def normalize_label(pred):
    if not pred:
        return "unknown"
    p = str(pred).strip().lower()
    if p in ("live", "real", "genuine", "real_face"):
        return "real"
    if p in ("spoof", "fake", "attack", "replay", "print", "mask"):
        return "fake"
    return "unknown"

# -------------------------------------------------
# FACE SPOOF DETECTION API
# -------------------------------------------------

@app.route("/api/detect", methods=["POST"])
def detect():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        f = request.files["image"]
        fname = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}.jpg"
        path = os.path.join(UPLOAD_DIR, fname)
        f.save(path)

        img = Image.open(path).convert("RGB")
        start = time.time()

        # -------------------------------------------------
        # MOCK MODE (NO MODEL)
        # -------------------------------------------------
        if detector is None:
            confidence = 0.50
            pred_label = "fake"
            attack_type = "mock"
        else:
            res = detector.predict_pil(img)
            prob_real = float(res.get("prob_real", 0.0))
            prob_fake = 1.0 - prob_real

            if prob_real >= prob_fake:
                pred_label = "real"
                confidence = prob_real
            else:
                pred_label = "fake"
                confidence = prob_fake

            attack_type = res.get("attack_type", "unknown")

        end = time.time()

        log = DetectionLog(
            timestamp=datetime.utcnow(),
            image_path=path,
            prediction=pred_label,
            confidence=float(confidence),
            attack_type=attack_type,
            processing_time_ms=int((end - start) * 1000),
            model_name=os.path.basename(MODEL_PATH) if MODEL_LOADED else "mock",
        )

        db.session.add(log)
        db.session.commit()

        return jsonify({
            "prediction": pred_label,
            "confidence": round(confidence, 4),
            "attack_type": attack_type,
            "processing_time_ms": log.processing_time_ms,
            "model_loaded": MODEL_LOADED
        }), 200

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500

# -------------------------------------------------
# STATS API
# -------------------------------------------------

@app.route("/api/stats/summary")
def stats_summary():
    total = DetectionLog.query.count()
    real = DetectionLog.query.filter_by(prediction="real").count()
    fake = DetectionLog.query.filter_by(prediction="fake").count()

    return jsonify({
        "total": total,
        "real": real,
        "fake": fake
    }), 200

# -------------------------------------------------
# UPLOAD ACCESS
# -------------------------------------------------

@app.route("/uploads/<path:fname>")
def uploaded_file(fname):
    return send_from_directory(UPLOAD_DIR, fname)

# -------------------------------------------------
# ENTRY POINT
# -------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
