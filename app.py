# backend/app.py — backend API with robust real/fake decision logic
import os, time, traceback
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

# local imports from backend package
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
# APP INITIALIZATION (NO FRONTEND SERVING HERE)
# -------------------------------------------------

app = Flask(__name__)
CORS(app)

# Database (SQLite is fine for Render free tier)
app.config["DB_FILE"] = os.path.join(BASE, "db.sqlite")
init_db(app)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------

detector = LiveGuardDetector(MODEL_PATH, threshold=0.5)

# -------------------------------------------------
# HEALTH CHECK (IMPORTANT FOR RENDER)
# -------------------------------------------------

@app.route("/")
def health():
    return jsonify({"status": "backend running"}), 200

# -------------------------------------------------
# LABEL NORMALIZATION
# -------------------------------------------------

def normalize_label(pred):
    if pred is None:
        return "unknown"

    p = str(pred).strip().lower()

    if p in ("live", "real", "genuine", "real_face", "live_face"):
        return "real"

    if p in ("spoof", "fake", "attack", "replay", "print", "printed", "mask"):
        return "fake"

    return "unknown"

# -------------------------------------------------
# FACE SPOOF DETECTION API
# -------------------------------------------------

@app.route("/api/detect", methods=["POST"])
def detect():
    try:
        # 1) Read image
        if "image" not in request.files:
            return jsonify({
                "error": "No image file sent (expected multipart 'image' field)."
            }), 400

        f = request.files["image"]
        fname = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}.jpg"
        path = os.path.join(UPLOAD_DIR, fname)
        f.save(path)

        img = Image.open(path).convert("RGB")

        # 2) Run model
        start = time.time()
        res = detector.predict_pil(img)
        end = time.time()

        # 3) Probabilities
        prob_real = float(res.get("prob_real", 0.0))
        prob_fake = float(res.get("prob_fake", 1.0 - prob_real))

        prob_real = max(0.0, min(1.0, prob_real))
        prob_fake = max(0.0, min(1.0, prob_fake))

        if prob_real >= prob_fake:
            raw_pred = "real"
            confidence = prob_real
        else:
            raw_pred = "fake"
            confidence = prob_fake

        pred_label = normalize_label(raw_pred)
        attack_type = res.get("attack_type", "unknown")

        # 4) Save result
        log = DetectionLog(
            timestamp=datetime.utcnow(),
            image_path=path,
            prediction=pred_label,
            confidence=float(confidence),
            attack_type=attack_type,
            processing_time_ms=int((end - start) * 1000),
            model_name=os.path.basename(MODEL_PATH),
        )

        db.session.add(log)
        db.session.commit()

        # 5) Response
        return jsonify({
            "prediction": pred_label,
            "confidence": round(float(confidence), 4),
            "attack_type": attack_type,
            "processing_time_ms": log.processing_time_ms,
            "timestamp": log.timestamp.isoformat(),
            "prob_real": round(prob_real, 4),
            "prob_fake": round(prob_fake, 4),
        }), 200

    except Exception as e:
        print("== Exception in /api/detect ==\n", traceback.format_exc())
        return jsonify({
            "error": "Internal server error during detection",
            "detail": str(e),
        }), 500

# -------------------------------------------------
# STATS API
# -------------------------------------------------

@app.route("/api/stats/summary", methods=["GET"])
def stats_summary():
    now = datetime.utcnow()

    total = DetectionLog.query.count()
    real = DetectionLog.query.filter_by(prediction="real").count()
    fake = DetectionLog.query.filter_by(prediction="fake").count()

    timeline = []
    for i in range(7):
        day = (now - timedelta(days=6 - i)).date()
        start = datetime.combine(day, datetime.min.time())
        end = datetime.combine(day, datetime.max.time())
        count = DetectionLog.query.filter(
            DetectionLog.timestamp >= start,
            DetectionLog.timestamp <= end,
        ).count()
        timeline.append({"date": day.isoformat(), "count": count})

    recent = (
        DetectionLog.query.order_by(DetectionLog.timestamp.desc())
        .limit(10)
        .all()
    )

    recent_list = [
        {
            "timestamp": r.timestamp.isoformat(),
            "image_path": r.image_path,
            "prediction": r.prediction,
            "confidence": r.confidence,
            "attack_type": r.attack_type,
        }
        for r in recent
    ]

    return jsonify({
        "total": total,
        "real": real,
        "fake": fake,
        "timeline": timeline,
        "recent": recent_list,
    }), 200

# -------------------------------------------------
# UPLOAD ACCESS (OPTIONAL)
# -------------------------------------------------

@app.route("/uploads/<path:fname>")
def uploaded_file(fname):
    return send_from_directory(UPLOAD_DIR, fname)

# -------------------------------------------------
# ENTRY POINT (RENDER SAFE)
# -------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting backend on port {port}")
    app.run(host="0.0.0.0", port=port)

