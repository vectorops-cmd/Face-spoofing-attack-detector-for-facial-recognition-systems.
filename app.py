# backend/app.py — backend API with robust real/fake decision logic
import os, time, traceback
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

# local imports from backend package
from models import init_db, db, DetectionLog

from detector import LiveGuardDetector


BASE = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Use lowercase 'frontend' unless your folder is actually named 'Frontend'
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="/")
app.config['DB_FILE'] = os.path.join(BASE, "db.sqlite")
CORS(app)
init_db(app)

MODEL_PATH = os.path.join(BASE, "saved_model", "model.h5")

# Lower the detector’s internal threshold a bit (you can tune this later)
detector = LiveGuardDetector(MODEL_PATH, threshold=0.5)


# ----------------- LABEL NORMALISATION -----------------

def normalize_label(pred):
    """
    Map whatever decision we make to canonical labels:
    - 'real'  = genuine / live face
    - 'fake'  = spoof / attack
    - 'unknown' = anything else / unexpected
    """
    if pred is None:
        return "unknown"

    p = str(pred).strip().lower()

    if p in ("live", "real", "genuine", "real_face", "live_face"):
        return "real"

    if p in ("spoof", "fake", "attack", "replay", "print", "printed", "mask"):
        return "fake"

    return "unknown"


# ----------------- ROUTES -----------------

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/detect", methods=["POST"])
def detect():
    try:
        # 1) Read image from multipart/form-data
        if "image" in request.files:
            f = request.files["image"]
            fname = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}.jpg"
            path = os.path.join(UPLOAD_DIR, fname)
            f.save(path)
            img = Image.open(path).convert("RGB")
        else:
            return jsonify({
                "error": "No image file sent (expected multipart 'image' field)."
            }), 400

        # 2) Run detector
        start = time.time()
        res = detector.predict_pil(img)  # dict: prediction, confidence, prob_real, prob_fake, attack_type
        end = time.time()

        # 3) Extract raw probabilities from detector (ignore its label if it's nonsense)
        prob_real = float(res.get("prob_real", 0.0))
        prob_fake = float(res.get("prob_fake", 1.0 - prob_real))

        # Safety clamp
        prob_real = max(0.0, min(1.0, prob_real))
        prob_fake = max(0.0, min(1.0, prob_fake))

        # 4) Make our own final decision here (NOT using res["prediction"])
        #    Simple, symmetric rule: whichever prob is higher wins.
        if prob_real >= prob_fake:
            raw_pred = "real"
            confidence = prob_real
        else:
            raw_pred = "fake"
            confidence = prob_fake

        pred_label = normalize_label(raw_pred)
        attack_type = res.get("attack_type", "unknown")

        # 5) Save to DB
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

        # 6) Build response
        out = {
            "prediction": pred_label,
            "confidence": round(float(confidence), 4),
            "attack_type": attack_type,
            "processing_time_ms": log.processing_time_ms,
            "timestamp": log.timestamp.isoformat(),
            # Optional debug info (helps you see what's going on)
            "prob_real": round(prob_real, 4),
            "prob_fake": round(prob_fake, 4),
        }
        return jsonify(out), 200

    except Exception as e:
        tb = traceback.format_exc()
        print("== Exception in /api/detect ==\n", tb)
        return jsonify({
            "error": "Internal server error during detection",
            "detail": str(e),
        }), 500


@app.route("/api/stats/summary", methods=["GET"])
def stats_summary():
    now = datetime.utcnow()
    total = DetectionLog.query.count()
    real = DetectionLog.query.filter_by(prediction="real").count()
    fake = DetectionLog.query.filter_by(prediction="fake").count()

    # Last 7 days timeline
    timeline = []
    for i in range(7):
        day = (now - timedelta(days=6 - i)).date()
        start = datetime.combine(day, datetime.min.time())
        end = datetime.combine(day, datetime.max.time())
        c = DetectionLog.query.filter(
            DetectionLog.timestamp >= start,
            DetectionLog.timestamp <= end,
        ).count()
        timeline.append({"date": day.isoformat(), "count": c})

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

    return jsonify(
        {
            "total": total,
            "real": real,
            "fake": fake,
            "timeline": timeline,
            "recent": recent_list,
        }
    ), 200


@app.route("/uploads/<path:fname>")
def uploaded_file(fname):
    return send_from_directory(UPLOAD_DIR, fname)


if __name__ == "__main__":
    print("Starting backend app — serving frontend from:", FRONTEND_DIR)
    app.run(host="0.0.0.0", port=5000, debug=True)
