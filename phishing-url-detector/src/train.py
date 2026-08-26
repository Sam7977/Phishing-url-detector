"""
train.py
--------
Trains and evaluates two candidate models on engineered URL features:
    1. Logistic Regression (baseline, interpretable)
    2. Random Forest (higher capacity, non-linear feature interactions)

Selects the best model by F1-score on a held-out test set, then
serializes the winning model + fitted scaler + feature names to
models/model.pkl for use by the Flask API.

Also writes docs/evaluation_report.md with metrics, a confusion
matrix, and feature importances so results are reproducible and
documented (matching the "full documentation" claim in the project
description).

Usage:
    python3 src/train.py --data data/dataset.csv
"""

import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import URLFeatures, extract_feature_matrix  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def load_dataset(path: str):
    urls, labels = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            urls.append(row["url"])
            labels.append(int(row["label"]))
    return urls, labels


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    cm = confusion_matrix(y_test, preds).tolist()
    print(f"\n--- {name} ---")
    for k, v in metrics.items():
        print(f"{k:10s}: {v:.4f}")
    print(f"confusion matrix [[TN, FP], [FN, TP]]: {cm}")
    return metrics, cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=str(ROOT / "data" / "dataset.csv"))
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=str(ROOT / "models" / "model.pkl"))
    ap.add_argument("--report", type=str,
                     default=str(ROOT / "docs" / "evaluation_report.md"))
    args = ap.parse_args()

    print(f"Loading dataset from {args.data} ...")
    urls, labels = load_dataset(args.data)
    print(f"Loaded {len(urls)} URLs ({sum(labels)} phishing / "
          f"{len(labels) - sum(labels)} legitimate)")

    print("Extracting features ...")
    feature_names, matrix = extract_feature_matrix(urls)
    X = np.array(matrix, dtype=float)
    y = np.array(labels, dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}

    # --- Logistic Regression baseline ---
    lr = LogisticRegression(max_iter=1000, random_state=args.seed)
    lr.fit(X_train_s, y_train)
    lr_metrics, lr_cm = evaluate("Logistic Regression", lr, X_test_s, y_test)
    results["logistic_regression"] = {"metrics": lr_metrics, "confusion_matrix": lr_cm}

    # --- Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=args.seed,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)  # tree ensembles don't need scaling
    rf_metrics, rf_cm = evaluate("Random Forest", rf, X_test, y_test)
    results["random_forest"] = {"metrics": rf_metrics, "confusion_matrix": rf_cm}

    # --- Select winner by F1 ---
    winner_name = max(results, key=lambda k: results[k]["metrics"]["f1"])
    winner_model = rf if winner_name == "random_forest" else lr
    uses_scaler = winner_name == "logistic_regression"
    print(f"\nSelected model: {winner_name} "
          f"(F1={results[winner_name]['metrics']['f1']:.4f})")

    # --- Measure inference latency (single prediction, warmed up) ---
    single_X = X_test_s[:1] if uses_scaler else X_test[:1]
    for _ in range(10):
        winner_model.predict_proba(single_X)
    n_timing = 200
    t0 = time.perf_counter()
    for _ in range(n_timing):
        winner_model.predict_proba(single_X)
    latency_ms = (time.perf_counter() - t0) / n_timing * 1000
    print(f"Avg single-prediction inference latency: {latency_ms:.3f} ms")

    # --- Feature importances (Random Forest) or coefficients (LR) ---
    if winner_name == "random_forest":
        importances = sorted(
            zip(feature_names, winner_model.feature_importances_),
            key=lambda t: -t[1],
        )
    else:
        importances = sorted(
            zip(feature_names, np.abs(winner_model.coef_[0])),
            key=lambda t: -t[1],
        )

    # --- Persist model bundle ---
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": winner_model,
        "scaler": scaler if uses_scaler else None,
        "uses_scaler": uses_scaler,
        "feature_names": feature_names,
        "model_name": winner_name,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metrics": results[winner_name]["metrics"],
    }
    with open(args.out, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Saved model bundle to {args.out}")

    # --- Write evaluation report ---
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as f:
        f.write("# Model Evaluation Report\n\n")
        f.write(f"Generated: {bundle['trained_at']}\n\n")
        f.write(f"Dataset size: {len(urls)} URLs "
                f"({sum(labels)} phishing / {len(labels) - sum(labels)} legitimate)\n")
        f.write(f"Train/test split: {int((1-args.test_size)*100)}/"
                f"{int(args.test_size*100)}, stratified, seed={args.seed}\n\n")

        f.write("## Results\n\n")
        f.write("| Metric | Logistic Regression | Random Forest |\n")
        f.write("|---|---|---|\n")
        for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            f.write(f"| {m} | "
                    f"{results['logistic_regression']['metrics'][m]:.4f} | "
                    f"{results['random_forest']['metrics'][m]:.4f} |\n")

        f.write(f"\n**Selected model: `{winner_name}`** "
                f"(highest F1-score on held-out test set)\n\n")

        f.write("## Confusion Matrices\n\n")
        f.write(f"Logistic Regression `[[TN, FP], [FN, TP]]`: "
                f"{results['logistic_regression']['confusion_matrix']}\n\n")
        f.write(f"Random Forest `[[TN, FP], [FN, TP]]`: "
                f"{results['random_forest']['confusion_matrix']}\n\n")

        f.write("## Inference Latency\n\n")
        f.write(f"Average single-URL prediction latency (model only, "
                f"excluding HTTP overhead): **{latency_ms:.3f} ms**\n\n")

        f.write("## Top Feature Importances\n\n")
        f.write("| Rank | Feature | Importance |\n|---|---|---|\n")
        for i, (name, imp) in enumerate(importances[:15], 1):
            f.write(f"| {i} | {name} | {imp:.4f} |\n")

        f.write("\n## Feature Engineering Rationale\n\n")
        f.write(
            "Features fall into four groups, chosen because they reflect "
            "well-documented phishing URL construction patterns "
            "(see references in README):\n\n"
            "- **Lexical/character-level** (length, digit ratio, entropy): "
            "phishing URLs tend to be longer and noisier than legitimate "
            "ones because attackers pack brand names, keywords, and random "
            "tokens into a single hostname/path.\n"
            "- **Structural** (dot/hyphen counts, subdomain depth, query "
            "param count): excessive subdomains and hyphens are a classic "
            "way to squat on a brand name (e.g. `paypal-secure-login.evil.tk`).\n"
            "- **Suspicious-pattern flags** (IP-as-host, `@` symbol, known "
            "shorteners, punycode, suspicious TLDs, sensitive keyword "
            "stuffing): each corresponds to a specific, well-known phishing "
            "technique for hiding or spoofing the real destination.\n"
            "- **Security hygiene** (HTTPS usage, url-to-hostname ratio): "
            "legitimate high-traffic sites overwhelmingly use HTTPS and "
            "have a stable domain relative to path length; phishing "
            "kits are frequently deployed without valid TLS or on "
            "long, disposable hostnames.\n\n"
            "No WHOIS/DNS/content-fetch features are used, by design: "
            "this keeps feature extraction fast (<1ms) and lets the API "
            "score URLs it has never made a network request to, which "
            "matters for a real-time endpoint.\n"
        )

    print(f"Wrote evaluation report to {args.report}")


if __name__ == "__main__":
    main()
