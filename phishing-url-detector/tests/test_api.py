"""
test_api.py
-----------
Basic tests for the Flask API using Flask's built-in test client
(no live server / network socket required).

Run:
    python3 tests/test_api.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from app import app  # noqa: E402


def pretty(resp):
    return resp.status_code, resp.get_json()


def main():
    client = app.test_client()
    failures = []

    def check(label, cond):
        status = "PASS" if cond else "FAIL"
        print(f"[{status}] {label}")
        if not cond:
            failures.append(label)

    # --- health ---
    status, body = pretty(client.get("/health"))
    check("GET /health returns 200", status == 200)
    check("health payload has model name", "model" in body)

    # --- predict: obvious phishing ---
    status, body = pretty(client.post(
        "/predict", json={"url": "http://192.168.4.55/paypal/secure-login/verify.php"}
    ))
    print("  ->", body)
    check("predict obvious phishing returns 200", status == 200)
    check("obvious phishing classified as phishing", body.get("prediction") == "phishing")
    check("latency reported under 120ms", body.get("latency_ms", 999) < 120)

    # --- predict: shortener ---
    status, body = pretty(client.post("/predict", json={"url": "http://bit.ly/3xAbCde"}))
    print("  ->", body)
    check("predict shortener returns 200", status == 200)
    check("shortener classified as phishing", body.get("prediction") == "phishing")

    # --- predict: legit-looking ---
    status, body = pretty(client.post(
        "/predict", json={"url": "https://www.github.com/anthropics/claude-code"}
    ))
    print("  ->", body)
    check("predict legit url returns 200", status == 200)
    check("legit url classified as legitimate", body.get("prediction") == "legitimate")

    # --- predict: missing url field ---
    status, body = pretty(client.post("/predict", json={}))
    check("missing url field returns 400", status == 400)

    # --- predict: empty url ---
    status, body = pretty(client.post("/predict", json={"url": "   "}))
    check("empty url returns 400", status == 400)

    # --- predict: non-string url ---
    status, body = pretty(client.post("/predict", json={"url": 12345}))
    check("non-string url returns 400", status == 400)

    # --- batch predict ---
    status, body = pretty(client.post("/predict/batch", json={"urls": [
        "https://www.wikipedia.org/wiki/Python",
        "http://secure-apple-verify-9x.tk/login.php",
    ]}))
    print("  ->", body)
    check("batch predict returns 200", status == 200)
    check("batch predict returns 2 results", body.get("count") == 2)

    # --- batch predict: empty list ---
    status, body = pretty(client.post("/predict/batch", json={"urls": []}))
    check("empty batch returns 400", status == 400)

    # --- 404 ---
    status, body = pretty(client.get("/does-not-exist"))
    check("unknown route returns 404", status == 404)

    print("\n" + "=" * 40)
    if failures:
        print(f"{len(failures)} test(s) FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All tests passed.")


if __name__ == "__main__":
    main()
