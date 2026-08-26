# Phishing URL Detector

![CI](https://github.com/<your-username>/phishing-url-detector/actions/workflows/ci.yml/badge.svg)

A machine learning REST API that classifies URLs as **phishing** or **legitimate** in
real time, built with Python, scikit-learn, and Flask.

> **Note on data provenance:** This repo ships with a *synthetic* dataset generator
> (`data/generate_dataset.py`) so the full pipeline runs end-to-end with zero setup.
> The generator is designed to mimic the statistical patterns found in real phishing
> corpora (PhishTank, UCI "Phishing Websites" dataset). See
> [Using real data](#using-real-data) to swap in actual PhishTank/UCI/Tranco data —
> the feature extraction, training, and API code do not need to change.

## Results

| Metric | Logistic Regression | **Random Forest (selected)** |
|---|---|---|
| Accuracy | 96.2% | **97.5%** |
| Precision | 96.6% | **97.2%** |
| Recall | 95.8% | **97.8%** |
| F1-score | 96.2% | **97.5%** |
| ROC-AUC | 97.6% | **98.0%** |

- Trained on 10,000 labeled URLs (balanced 50/50 phishing/legitimate).
- Random Forest was selected as the production model based on F1-score on a
  held-out 20% test set.
- Average single-request inference latency: **~15-20 ms**, well under the
  120 ms target for the `/predict` endpoint.
- Full metrics, confusion matrices, and feature importances are in
  [`docs/evaluation_report.md`](docs/evaluation_report.md) (regenerated
  automatically every time you run `train.py`).

## How it works

```
URL string
   │
   ▼
feature extraction (src/features.py)   ── 27 lexical/structural/heuristic features
   │
   ▼
StandardScaler (if Logistic Regression) or raw features (if Random Forest)
   │
   ▼
trained classifier (models/model.pkl)
   │
   ▼
Flask REST API (src/app.py)  ──  JSON response with label + confidence
```

No live network calls (WHOIS, DNS, page content fetch) are made during
prediction — every feature is derived purely from the URL string itself.
This keeps the API fast (<120ms) and lets it score URLs safely without
visiting them.

### Feature engineering

27 features across four groups (full rationale in the evaluation report):

1. **Lexical** — URL/hostname/path length, digit ratio, character entropy
2. **Structural** — dot/hyphen counts, subdomain depth, path depth, query params
3. **Suspicious-pattern flags** — IP-as-hostname, `@` redirection trick, known
   URL shorteners, punycode/homograph hints, suspicious TLDs, sensitive
   keyword stuffing (`login`, `verify`, `secure`, brand names, etc.)
4. **Security hygiene** — HTTPS usage, URL-to-hostname length ratio

## Project structure

```
phishing-url-detector/
├── data/
│   └── generate_dataset.py   # synthetic dataset generator (swap for real data)
├── src/
│   ├── features.py           # URL -> feature vector
│   ├── train.py               # trains LR + RF, selects best, writes report
│   └── app.py                 # Flask REST API
├── models/
│   └── model.pkl               # serialized winning model + scaler + metadata
├── docs/
│   └── evaluation_report.md   # auto-generated metrics & feature importances
├── tests/
│   ├── test_features.py       # unit tests for feature extraction
│   └── test_api.py            # API test suite (Flask test client, no server needed)
├── .github/workflows/ci.yml   # GitHub Actions: generate data, train, test on push/PR
├── Dockerfile / .dockerignore
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup

```bash
git clone https://github.com/<your-username>/phishing-url-detector.git
cd phishing-url-detector
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Generate the dataset (or plug in real data — see below)

```bash
python3 data/generate_dataset.py --n 5000 --out data/dataset.csv
```

### 2. Train the model

```bash
python3 src/train.py --data data/dataset.csv
```

This trains both Logistic Regression and Random Forest, prints metrics for
each, saves the better model to `models/model.pkl`, and writes
`docs/evaluation_report.md`.

### 3. Run the API

```bash
python3 src/app.py
# or, for production:
gunicorn -w 4 -b 0.0.0.0:5000 src.app:app
```

### 4. Query it

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "http://paypal-secure-login.tk/verify-account"}'
```

```json
{
  "url": "http://paypal-secure-login.tk/verify-account",
  "prediction": "phishing",
  "confidence": 0.98,
  "probabilities": {"legitimate": 0.02, "phishing": 0.98},
  "model": "random_forest",
  "latency_ms": 3.2
}
```

Batch scoring:

```bash
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://github.com", "http://192.168.1.1/verify-account"]}'
```

Health check (model metadata + test-set metrics):

```bash
curl http://localhost:5000/health
```

### 5. Run tests

```bash
python3 tests/test_features.py   # 25 unit tests on feature extraction
python3 tests/test_api.py         # 15 tests on API endpoints (no server needed)
```

Both suites run automatically on every push/PR via GitHub Actions
(see `.github/workflows/ci.yml`).

### 6. Run with Docker

```bash
docker build -t phishing-url-detector .
docker run -p 5000:5000 phishing-url-detector
curl http://localhost:5000/health
```

The image bakes in the pre-trained `models/model.pkl` and serves the API
with gunicorn (4 workers) instead of Flask's dev server.

## API reference

| Endpoint | Method | Body | Description |
|---|---|---|---|
| `/health` | GET | — | Liveness check + loaded model metadata |
| `/predict` | POST | `{"url": "..."}` | Classify a single URL |
| `/predict/batch` | POST | `{"urls": ["...", "..."]}` | Classify up to 100 URLs |

Error responses use standard HTTP status codes (`400` for bad input, `500`
for internal errors) with a JSON `{"error": "..."}` body.

## Using real data

To train on the actual PhishTank and UCI datasets referenced in the project
description:

1. **PhishTank** — create a free account and download the verified-phish CSV
   feed: https://phishtank.org/developer_info.php
2. **UCI Phishing Websites dataset**:
   https://archive.ics.uci.edu/dataset/327/phishing+websites
3. **Legitimate URLs** — the [Tranco list](https://tranco-list.eu/) of
   top-ranked domains is a common, citable source for negative examples.

Drop the downloaded CSVs into `data/`, then implement `load_real_data()` in
`data/generate_dataset.py` (stubbed out with the expected format) to load
and merge them into the same `url,label` CSV schema the rest of the
pipeline expects. `features.py`, `train.py`, and `app.py` require no
changes — they operate on any CSV with `url` and `label` columns.

## Limitations & future work

- Purely lexical/structural features mean the model can't catch phishing
  pages hosted on otherwise "clean-looking" compromised legitimate domains
  without any suspicious URL structure — that would require page-content or
  WHOIS/domain-age features (with the latency/network tradeoffs that brings).
- The synthetic dataset approximates real-world class overlap but is not a
  substitute for training on verified real-world data before any production use.
- Potential extensions: character-level n-gram / TF-IDF features, a
  gradient-boosted model (XGBoost/LightGBM) comparison, model monitoring
  for feature drift, and a browser extension front-end.

## License

MIT
