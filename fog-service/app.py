"""FOG//SIGHT prototype API: OpenCV image features + weather fusion.
This is an interpretable baseline, not a trained or safety-certified ML model.
"""
import io
import math
import os
from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
from PIL import Image

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": os.getenv("CORS_ORIGIN", "*")}})

def analyze_frame(image_bytes: bytes, humidity: float, temperature: float, wind_speed: float):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image.thumbnail((640, 480))
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

    # Global contrast proxy and dark-channel statistic are common visibility cues.
    contrast = float(gray.std() * 255.0)
    dark_channel = np.min(rgb, axis=2)
    dark_mean = float(np.mean(dark_channel))
    bright_fraction = float(np.mean(gray > 0.82))

    contrast_fog = np.clip((1.0 - contrast / 75.0) * 100.0, 0.0, 100.0)
    veil_fog = np.clip((dark_mean - 0.12) * 105.0 + bright_fraction * 18.0, 0.0, 35.0)
    weather_fog = np.clip((humidity - 55.0) * 1.15 + max(0.0, 15.0 - temperature) * 1.1 - wind_speed * 0.65, 0.0, 100.0)
    score = int(round(np.clip(contrast_fog * 0.58 + veil_fog * 0.14 + weather_fog * 0.28, 0.0, 100.0)))
    visibility_km = round(float(np.clip(10.0 * math.exp(-score / 43.0), 0.1, 10.0)), 2)
    risk = "HIGH" if score >= 70 else "MODERATE" if score >= 40 else "LOW"
    return {
        "fog_score": score,
        "visibility_km": visibility_km,
        "risk_level": risk,
        "contrast": round(contrast, 1),
        "dark_channel_mean": round(dark_mean, 3),
        "humidity": round(humidity, 1),
        "temperature": round(temperature, 1),
        "wind_speed": round(wind_speed, 1),
        "method": "OpenCV-style image statistics + weather heuristic; prototype, not a trained model."
    }

@app.get("/health")
def health():
    return jsonify({"ok": True, "service": "FOG//SIGHT prototype API"})

@app.post("/api/fog/analyze")
def fog_analyze():
    if "image" not in request.files:
        return jsonify({"error": "Upload an image using the 'image' field."}), 400
    try:
        humidity = float(request.form.get("humidity", 80))
        temperature = float(request.form.get("temperature", 18))
        wind_speed = float(request.form.get("wind_speed", 4))
        if not 0 <= humidity <= 100 or not -30 <= temperature <= 60 or not 0 <= wind_speed <= 200:
            raise ValueError("Weather values are outside supported ranges.")
        result = analyze_frame(request.files["image"].read(), humidity, temperature, wind_speed)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5001")))
