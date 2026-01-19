# backend/models.py
from flask_sqlalchemy import SQLAlchemy
db = SQLAlchemy()

class DetectionLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False)
    image_path = db.Column(db.String, nullable=False)
    prediction = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    attack_type = db.Column(db.String(50), nullable=True)
    processing_time_ms = db.Column(db.Integer, nullable=True)
    model_name = db.Column(db.String(128), nullable=True)

def init_db(app):
    app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{app.config.get('DB_FILE','db.sqlite')}"
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)
    with app.app_context():
        db.create_all()
