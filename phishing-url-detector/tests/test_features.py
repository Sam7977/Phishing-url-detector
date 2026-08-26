"""
test_features.py
-----------------
Unit tests for src/features.py — verifies each engineered feature fires
correctly on known-pattern URLs (IP host, '@' trick, shorteners, etc.)
and that the feature vector shape/order is stable.

Run:
    python3 tests/test_features.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from features import URLFeatures, extract_feature_matrix, extract_features  # noqa: E402


def main():
    failures = []

    def check(label, cond):
        status = "PASS" if cond else "FAIL"
        print(f"[{status}] {label}")
        if not cond:
            failures.append(label)

    # --- basic shape / determinism ---
    f1 = extract_features("https://example.com/")
    f2 = extract_features("https://example.com/")
    check("feature extraction is deterministic", f1 == f2)
    check("feature vector length matches feature_names length",
          len(f1.to_vector()) == len(URLFeatures.feature_names()))

    # --- IP-as-hostname ---
    f = extract_features("http://192.168.1.1/login")
    check("detects IPv4 hostname", f.has_ip_host == 1)
    f = extract_features("https://example.com/login")
    check("does not flag normal hostname as IP", f.has_ip_host == 0)

    # --- '@' redirection trick ---
    f = extract_features("http://paypal.com@evil.tk/verify")
    check("detects '@' symbol trick", f.has_at_symbol == 1)
    f = extract_features("https://example.com/contact")
    check("no false positive on '@' trick", f.has_at_symbol == 0)

    # --- known URL shorteners ---
    f = extract_features("http://bit.ly/3xAbCde")
    check("detects known shortener (bit.ly)", f.is_known_shortener == 1)
    f = extract_features("http://tinyurl.com/abc123")
    check("detects known shortener (tinyurl.com)", f.is_known_shortener == 1)
    f = extract_features("https://github.com/foo/bar")
    check("does not flag github.com as shortener", f.is_known_shortener == 0)

    # --- punycode ---
    f = extract_features("https://xn--pple-43d.com/verify")
    check("detects punycode hostname", f.has_punycode == 1)
    f = extract_features("https://apple.com/verify")
    check("no false positive on punycode", f.has_punycode == 0)

    # --- suspicious TLD ---
    f = extract_features("http://secure-login.tk/verify")
    check("detects suspicious TLD (.tk)", f.has_suspicious_tld == 1)
    f = extract_features("https://secure-login.com/verify")
    check("does not flag .com as suspicious TLD", f.has_suspicious_tld == 0)

    # --- HTTPS usage ---
    f = extract_features("https://example.com/")
    check("detects https scheme", f.uses_https == 1)
    f = extract_features("http://example.com/")
    check("detects non-https scheme", f.uses_https == 0)

    # --- sensitive keyword counting ---
    f = extract_features("http://secure-paypal-login-verify.tk/account")
    check("counts multiple sensitive keywords",
          f.sensitive_keyword_count >= 3)
    f = extract_features("https://example.com/about")
    check("no sensitive keywords on benign path", f.sensitive_keyword_count == 0)

    # --- subdomain counting ---
    f = extract_features("https://a.b.c.example.com/")
    check("counts multiple subdomains", f.num_subdomains == 3)
    f = extract_features("https://example.com/")
    check("zero subdomains on bare domain", f.num_subdomains == 0)

    # --- hex-encoded IP ---
    f = extract_features("http://0x1a2b3c4d/login")
    check("detects hex-encoded IP host", f.has_hex_ip == 1)

    # --- port detection ---
    f = extract_features("http://example.com:8080/login")
    check("detects explicit port", f.has_port == 1)
    f = extract_features("http://example.com/login")
    check("no port flagged when absent", f.has_port == 0)

    # --- empty-ish / degenerate input doesn't crash ---
    try:
        extract_features("http://")
        check("handles minimal/degenerate URL without raising", True)
    except Exception as e:  # noqa: BLE001
        check(f"handles minimal/degenerate URL without raising ({e})", False)

    # --- batch extraction matches single extraction ---
    urls = ["https://example.com/", "http://192.168.1.1/login", "http://bit.ly/x"]
    names, matrix = extract_feature_matrix(urls)
    check("feature_matrix names match URLFeatures.feature_names()",
          names == URLFeatures.feature_names())
    check("feature_matrix row count matches input URL count",
          len(matrix) == len(urls))
    check("feature_matrix row matches single extraction",
          matrix[1] == extract_features(urls[1]).to_vector())

    print("\n" + "=" * 40)
    if failures:
        print(f"{len(failures)} test(s) FAILED:")
        for fl in failures:
            print(f"  - {fl}")
        sys.exit(1)
    else:
        print("All tests passed.")


if __name__ == "__main__":
    main()
