"""
app.py
------
Flask REST API for real-time phishing URL detection.

Endpoints:
    GET  /health           -> service liveness + model metadata
    POST /predict           -> single URL prediction
    POST /predict/batch      -> multiple URLs in one request

Example:
    curl -X POST http://localhost:5000/predict \
         -H "Content-Type: application/json" \
         -d '{"url": "http://paypal-secure-login.tk/verify"}'

Response:
    {
      "url": "http://paypal-secure-login.tk/verify",
      "prediction": "phishing",
      "confidence": 0.98,
      "probabilities": {"legitimate": 0.02, "phishing": 0.98},
      "model": "random_forest",
      "latency_ms": 3.2
    }
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import extract_features  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.pkl"

app = Flask(__name__)

# --- Load model bundle once at startup ---
_bundle = None


def load_model():
    global _bundle
    with open(MODEL_PATH, "rb") as f:
        _bundle = pickle.load(f)
    app.logger.info(
        f"Loaded model '{_bundle['model_name']}' trained at "
        f"{_bundle['trained_at']} (test F1={_bundle['metrics']['f1']:.4f})"
    )
    return _bundle


load_model()

MAX_URL_LENGTH = 2048
MAX_BATCH_SIZE = 100


def validate_url_input(url) -> tuple[bool, str]:
    if not isinstance(url, str):
        return False, "url must be a string"
    if not url.strip():
        return False, "url must not be empty"
    if len(url) > MAX_URL_LENGTH:
        return False, f"url exceeds max length of {MAX_URL_LENGTH} characters"
    return True, ""


def score_url(url: str) -> dict:
    feats = extract_features(url)
    x = np.array([feats.to_vector()], dtype=float)

    if _bundle["uses_scaler"] and _bundle["scaler"] is not None:
        x = _bundle["scaler"].transform(x)

    model = _bundle["model"]
    proba = model.predict_proba(x)[0]
    # class order follows model.classes_, typically [0, 1] = [legit, phish]
    classes = list(model.classes_)
    prob_map = {int(c): float(p) for c, p in zip(classes, proba)}
    legit_p = prob_map.get(0, 0.0)
    phish_p = prob_map.get(1, 0.0)

    label = "phishing" if phish_p >= legit_p else "legitimate"
    confidence = max(legit_p, phish_p)

    return {
        "url": url,
        "prediction": label,
        "confidence": round(confidence, 4),
        "probabilities": {
            "legitimate": round(legit_p, 4),
            "phishing": round(phish_p, 4),
        },
    }


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": _bundle["model_name"],
        "trained_at": _bundle["trained_at"],
        "test_metrics": _bundle["metrics"],
    })


@app.route("/predict", methods=["POST"])
def predict():
    t0 = time.perf_counter()
    data = request.get_json(silent=True)
    if data is None or "url" not in data:
        return jsonify({"error": "request body must be JSON with a 'url' field"}), 400

    url = data["url"]
    ok, err = validate_url_input(url)
    if not ok:
        return jsonify({"error": err}), 400

    try:
        result = score_url(url.strip())
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("prediction failed")
        return jsonify({"error": "internal error scoring url", "detail": str(exc)}), 500

    result["model"] = _bundle["model_name"]
    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return jsonify(result)


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    t0 = time.perf_counter()
    data = request.get_json(silent=True)
    if data is None or "urls" not in data or not isinstance(data["urls"], list):
        return jsonify({"error": "request body must be JSON with a 'urls' list"}), 400

    urls = data["urls"]
    if len(urls) == 0:
        return jsonify({"error": "urls list must not be empty"}), 400
    if len(urls) > MAX_BATCH_SIZE:
        return jsonify({"error": f"batch size exceeds max of {MAX_BATCH_SIZE}"}), 400

    results = []
    for url in urls:
        ok, err = validate_url_input(url)
        if not ok:
            results.append({"url": url, "error": err})
            continue
        try:
            results.append(score_url(url.strip()))
        except Exception as exc:  # noqa: BLE001
            results.append({"url": url, "error": str(exc)})

    return jsonify({
        "model": _bundle["model_name"],
        "count": len(results),
        "results": results,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
    })


@app.errorhandler(404)
def not_found(_e):
    return jsonify({"error": "not found", "available_endpoints":
                     ["/health", "/predict", "/predict/batch"]}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
