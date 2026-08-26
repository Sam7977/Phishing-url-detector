"""
generate_dataset.py
--------------------
Builds a labeled URL dataset for training.

IMPORTANT — READ THIS:
This script generates a SYNTHETIC dataset that mimics the statistical
patterns of real phishing/legitimate URLs (the kind found in PhishTank's
verified feed and the UCI "Phishing Websites" repository). It exists so
the whole pipeline (features -> training -> API) runs end-to-end without
needing network access to external data sources.

For your real GitHub project you should REPLACE this with actual data:
    1. PhishTank verified feed (CSV/JSON export, free account required):
       https://phishtank.org/developer_info.php
    2. UCI ML Repository "Phishing Websites" dataset:
       https://archive.ics.uci.edu/dataset/327/phishing+websites
    3. Legitimate URLs: Tranco or Majestic Million top-sites lists
       https://tranco-list.eu/

Once downloaded, drop the CSVs in this data/ folder and adapt
`load_real_data()` below (stubbed out) to point at them — the rest
of the pipeline (features.py, train.py, app.py) does not need to change.

Usage:
    python3 generate_dataset.py --n 10000 --out dataset.csv
"""

import argparse
import csv
import random
import string

random.seed(42)

LEGIT_BRANDS = [
    "github.com", "google.com", "wikipedia.org", "amazon.com", "nytimes.com",
    "stackoverflow.com", "reddit.com", "bbc.com", "linkedin.com", "medium.com",
    "microsoft.com", "apple.com", "cloudflare.com", "python.org", "spotify.com",
    "netflix.com", "dropbox.com", "salesforce.com", "adobe.com", "shopify.com",
    "coursera.org", "khanacademy.org", "who.int", "un.org", "nasa.gov",
    "harvard.edu", "mit.edu", "nature.com", "sciencedirect.com", "ieee.org",
]

LEGIT_PATH_WORDS = [
    "articles", "blog", "docs", "about", "contact", "products", "help",
    "search", "user", "settings", "news", "learn", "courses", "api",
    "download", "support", "careers", "pricing", "features", "index",
]

PHISH_BRAND_TARGETS = [
    "paypal", "apple", "amazon", "microsoft", "netflix", "bankofamerica",
    "wellsfargo", "chase", "ebay", "google", "facebook", "instagram",
    "dhl", "usps", "irs", "coinbase", "binance", "outlook", "office365",
]

PHISH_KEYWORDS = [
    "secure", "login", "signin", "verify", "update", "account", "confirm",
    "billing", "suspend", "unlock", "wallet", "password-reset", "webscr",
    "support", "alert", "security-check",
]

SUSPICIOUS_TLDS = ["xyz", "top", "click", "tk", "ml", "ga", "cf", "review",
                    "work", "zip", "kim", "country"]
NORMAL_TLDS = ["com", "org", "net", "edu", "gov", "io", "co"]

SHORTENERS = ["bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd"]


def _rand_str(n, alphabet=string.ascii_lowercase + string.digits):
    return "".join(random.choice(alphabet) for _ in range(n))


def _rand_ip():
    return ".".join(str(random.randint(1, 254)) for _ in range(4))


def make_legit_url(hard=False):
    brand = random.choice(LEGIT_BRANDS)
    scheme = "https"
    r = random.random()
    if not hard:
        if r < 0.35:
            return f"{scheme}://{brand}/"
        elif r < 0.7:
            sub = random.choice(["www", "www", "docs", "blog", "help", "shop"])
            path = "/".join(random.choice(LEGIT_PATH_WORDS)
                             for _ in range(random.randint(1, 2)))
            return f"{scheme}://{sub}.{brand}/{path}"
        else:
            path = "/".join(random.choice(LEGIT_PATH_WORDS)
                             for _ in range(random.randint(1, 3)))
            qs = f"?id={random.randint(1, 99999)}" if random.random() < 0.4 else ""
            return f"{scheme}://{brand}/{path}{qs}"
    # "hard" legitimate examples: real sites do sometimes look noisy —
    # marketing landing pages, password reset flows, long tracking query
    # strings, multiple subdomains. These overlap with phishing signals
    # and are what keeps accuracy realistic instead of a trivial 100%.
    r2 = random.random()
    if r2 < 0.3:
        # legit password/account pages contain the same "sensitive" words
        kw = random.choice(["login", "signin", "account", "password-reset",
                             "verify", "billing", "support"])
        sub = random.choice(["accounts", "id", "auth", "my", "secure"])
        return f"{scheme}://{sub}.{brand}/{kw}"
    elif r2 < 0.55:
        # deep multi-level subdomains (CDNs, regional sites) + hyphens
        sub = f"{_rand_str(4)}-{random.choice(['cdn','static','api','eu','us'])}"
        path = "/".join(random.choice(LEGIT_PATH_WORDS)
                         for _ in range(random.randint(1, 3)))
        return f"{scheme}://{sub}.{brand}/{path}"
    elif r2 < 0.8:
        # long tracking query strings
        path = random.choice(LEGIT_PATH_WORDS)
        qs = "&".join(f"utm_{p}={_rand_str(5)}" for p in
                       random.sample(["source", "medium", "campaign", "term"], 3))
        return f"{scheme}://{brand}/{path}?{qs}"
    else:
        # occasional http (not https) legitimate mirror/redirect page
        path = "/".join(random.choice(LEGIT_PATH_WORDS)
                         for _ in range(random.randint(1, 2)))
        return f"http://{brand}/{path}"


def make_phish_url(hard=False):
    style = random.random()
    target = random.choice(PHISH_BRAND_TARGETS)
    kw = random.choice(PHISH_KEYWORDS)
    tld = random.choice(SUSPICIOUS_TLDS + NORMAL_TLDS)

    if not hard:
        if style < 0.18:
            # raw IP host, brand + keyword stuffed in path
            return f"http://{_rand_ip()}/{target}/{kw}/{_rand_str(6)}"
        elif style < 0.32:
            # known shortener
            return f"http://{random.choice(SHORTENERS)}/{_rand_str(7)}"
        elif style < 0.55:
            # brand as subdomain of unrelated/suspicious domain
            filler = _rand_str(random.randint(4, 9))
            return f"http://{target}-{kw}.{filler}.{tld}/{_rand_str(5)}"
        elif style < 0.75:
            # long hyphenated hostname mimicking brand
            parts = [target, kw, _rand_str(4), "secure"]
            random.shuffle(parts)
            return f"http://{'-'.join(parts)}.{tld}/login.php"
        elif style < 0.9:
            # '@' trick
            return f"http://{target}.com@{_rand_str(8)}.{tld}/{kw}"
        else:
            # punycode / homograph-style hint
            return f"https://xn--{_rand_str(5)}-{_rand_str(3)}.{tld}/{target}-{kw}"
    # "hard" phishing examples: more sophisticated attacks that evade
    # naive heuristics — valid HTTPS (many phishing kits now use free
    # certs), common TLD, short clean-looking hostname, single keyword.
    r2 = random.random()
    if r2 < 0.35:
        # short, clean-looking compromised legitimate domain used as host
        filler = _rand_str(random.randint(5, 8))
        return f"https://{filler}.{random.choice(NORMAL_TLDS)}/{kw}"
    elif r2 < 0.6:
        # single subtle hyphen, valid https, common tld, no raw IP
        return f"https://{target}-{kw}.{random.choice(NORMAL_TLDS)}/"
    elif r2 < 0.8:
        # brand name as a path segment only, clean host, https
        filler = _rand_str(random.randint(4, 7))
        return f"https://{filler}.{random.choice(NORMAL_TLDS)}/{target}/{kw}"
    else:
        # minimal-length phishing link (common in SMS/social phishing)
        return f"https://{_rand_str(6)}.{random.choice(NORMAL_TLDS)}/{_rand_str(3)}"


def generate(n_per_class: int, hard_ratio: float = 0.25, noise_ratio: float = 0.02):
    """
    hard_ratio: fraction of each class drawn from the harder, overlapping
        generators above (mirrors real-world label noise / evasive
        phishing / noisy legitimate URLs).
    noise_ratio: fraction of labels randomly flipped to simulate
        annotation noise / edge cases present in real crowd-sourced
        datasets like PhishTank.
    """
    rows = []
    for _ in range(n_per_class):
        hard = random.random() < hard_ratio
        rows.append([make_legit_url(hard=hard), 0])
    for _ in range(n_per_class):
        hard = random.random() < hard_ratio
        rows.append([make_phish_url(hard=hard), 1])

    n_flip = int(len(rows) * noise_ratio)
    for idx in random.sample(range(len(rows)), n_flip):
        rows[idx][1] = 1 - rows[idx][1]

    random.shuffle(rows)
    return [tuple(r) for r in rows]


def load_real_data(phishtank_csv: str | None, legit_csv: str | None):
    """
    STUB for plugging in real data. Not called by default.

    Expected minimal format:
      - PhishTank export: a column named 'url' for confirmed phishing URLs.
      - Legit list (e.g. Tranco top-1m.csv): a column of domains; prefix
        with 'https://' to form a URL.

    Example:
        rows = []
        with open(phishtank_csv) as f:
            for row in csv.DictReader(f):
                rows.append((row['url'], 1))
        with open(legit_csv) as f:
            for row in csv.DictReader(f):
                rows.append((f"https://{row['domain']}", 0))
        return rows
    """
    raise NotImplementedError(
        "Plug in real PhishTank/UCI/Tranco CSVs here for the production "
        "version of this dataset. See module docstring."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000,
                     help="number of URLs PER CLASS (total = 2n)")
    ap.add_argument("--out", type=str, default="dataset.csv")
    args = ap.parse_args()

    rows = generate(args.n)
    out_path = args.out
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "label"])
        writer.writerows(rows)

    n_phish = sum(r[1] for r in rows)
    print(f"Wrote {len(rows)} rows to {out_path} "
          f"({n_phish} phishing / {len(rows) - n_phish} legitimate)")


if __name__ == "__main__":
    main()
