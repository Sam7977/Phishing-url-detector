"""
features.py
------------
Lexical / host-based feature extraction for phishing URL detection.

This module turns a raw URL string into a fixed-length numeric feature
vector. No live network calls are made (no WHOIS, no DNS, no page
content fetch) so that feature extraction is fast (<1ms) and safe to
run on arbitrary/untrusted URLs inside the Flask API.

Feature groups:
    1. Lexical / character-level statistics (length, digit ratio, etc.)
    2. Structural URL components (subdomains, path depth, query params)
    3. Suspicious-pattern flags (IP-as-host, '@' symbol, many hyphens,
       URL shorteners, punycode/homograph hints, brand-keyword stuffing)
    4. Security-hygiene signals (HTTPS usage, port oddities)

Every function is pure and side-effect free so it can be unit tested
in isolation and reused both for offline training and the live API.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, fields
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Reference data used by a few heuristic features
# ---------------------------------------------------------------------------

KNOWN_SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "shorte.st", "bl.ink", "rebrand.ly", "cutt.ly", "tiny.cc",
}

# Brand / financial keywords frequently stuffed into phishing hostnames
# or paths to build false trust (e.g. "secure-paypal-login.example.com")
SENSITIVE_KEYWORDS = [
    "secure", "account", "update", "login", "signin", "bank", "verify",
    "confirm", "webscr", "paypal", "ebay", "amazon", "apple", "microsoft",
    "password", "billing", "suspend", "unlock", "wallet",
]

SUSPICIOUS_TLDS = {
    "zip", "review", "country", "kim", "cricket", "science", "work",
    "party", "gq", "link", "xyz", "top", "click", "tk", "ml", "ga", "cf",
}

IPV4_RE = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
HEX_IP_RE = re.compile(r"^0x[0-9a-fA-F]+$")


def _shannon_entropy(s: str) -> float:
    """Character-level Shannon entropy of a string (0 for empty string)."""
    if not s:
        return 0.0
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def _safe_urlparse(url: str):
    """urlparse that tolerates URLs missing a scheme (assumes http)."""
    url = url.strip()
    if "://" not in url:
        url = "http://" + url
    return urlparse(url)


@dataclass
class URLFeatures:
    """One row of engineered features for a single URL."""

    # --- lexical / character-level ---
    url_length: int
    hostname_length: int
    path_length: int
    digit_ratio: float
    letter_ratio: float
    special_char_count: int
    url_entropy: float

    # --- structural ---
    num_dots: int
    num_hyphens: int
    num_underscores: int
    num_slashes: int
    num_query_params: int
    num_subdomains: int
    path_depth: int
    tld_length: int

    # --- suspicious pattern flags (0/1) ---
    has_ip_host: int
    has_at_symbol: int
    has_double_slash_redirect: int
    has_hex_ip: int
    is_known_shortener: int
    has_suspicious_tld: int
    sensitive_keyword_count: int
    has_port: int
    has_punycode: int
    hostname_has_digits: int

    # --- security hygiene ---
    uses_https: int
    ratio_url_to_hostname: float

    @staticmethod
    def feature_names() -> list[str]:
        return [f.name for f in fields(URLFeatures)]

    def to_vector(self) -> list[float]:
        return [getattr(self, name) for name in self.feature_names()]


def extract_features(url: str) -> URLFeatures:
    """Extract the full engineered feature set from a raw URL string."""
    parsed = _safe_urlparse(url)
    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""
    full = url.strip()

    length = len(full)
    digits = sum(ch.isdigit() for ch in full)
    letters = sum(ch.isalpha() for ch in full)
    specials = sum(not ch.isalnum() for ch in full)

    labels = hostname.split(".") if hostname else []
    tld = labels[-1] if len(labels) > 1 else ""
    # subdomain count = total labels - (domain + tld), floor at 0
    num_subdomains = max(len(labels) - 2, 0)

    host_no_port = hostname.split(":")[0] if hostname else ""

    return URLFeatures(
        url_length=length,
        hostname_length=len(hostname),
        path_length=len(path),
        digit_ratio=round(digits / length, 4) if length else 0.0,
        letter_ratio=round(letters / length, 4) if length else 0.0,
        special_char_count=specials,
        url_entropy=round(_shannon_entropy(full), 4),
        num_dots=full.count("."),
        num_hyphens=full.count("-"),
        num_underscores=full.count("_"),
        num_slashes=full.count("/"),
        num_query_params=(len(query.split("&")) if query else 0),
        num_subdomains=num_subdomains,
        path_depth=len([p for p in path.split("/") if p]),
        tld_length=len(tld),
        has_ip_host=int(bool(IPV4_RE.match(host_no_port))),
        has_at_symbol=int("@" in full),
        has_double_slash_redirect=int(full.rfind("//") > 7),
        has_hex_ip=int(bool(HEX_IP_RE.match(host_no_port))),
        is_known_shortener=int(host_no_port.lower() in KNOWN_SHORTENERS),
        has_suspicious_tld=int(tld.lower() in SUSPICIOUS_TLDS),
        sensitive_keyword_count=sum(
            kw in full.lower() for kw in SENSITIVE_KEYWORDS
        ),
        has_port=int(parsed.port is not None),
        has_punycode=int("xn--" in hostname.lower()),
        hostname_has_digits=int(any(ch.isdigit() for ch in host_no_port)),
        uses_https=int(parsed.scheme == "https"),
        ratio_url_to_hostname=(
            round(length / len(hostname), 4) if hostname else 0.0
        ),
    )


def extract_feature_matrix(urls: list[str]):
    """Vectorized helper: list[str] -> (feature_names, list[list[float]])."""
    rows = [extract_features(u).to_vector() for u in urls]
    return URLFeatures.feature_names(), rows


if __name__ == "__main__":
    samples = [
        "https://www.google.com/search?q=test",
        "http://192.168.1.1/secure-login/paypal/update-account",
        "http://bit.ly/3xF9kLm",
        "https://xn--pple-43d.com/verify-account-now",
    ]
    for s in samples:
        print(s)
        print(extract_features(s))
        print()
