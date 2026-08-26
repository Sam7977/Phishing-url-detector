# Model Evaluation Report

Generated: 2026-08-26T08:14:41Z

Dataset size: 10000 URLs (4978 phishing / 5022 legitimate)
Train/test split: 80/20, stratified, seed=42

## Results

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| accuracy | 0.9620 | 0.9750 |
| precision | 0.9656 | 0.9721 |
| recall | 0.9578 | 0.9779 |
| f1 | 0.9617 | 0.9750 |
| roc_auc | 0.9761 | 0.9796 |

**Selected model: `random_forest`** (highest F1-score on held-out test set)

## Confusion Matrices

Logistic Regression `[[TN, FP], [FN, TP]]`: [[970, 34], [42, 954]]

Random Forest `[[TN, FP], [FN, TP]]`: [[976, 28], [22, 974]]

## Inference Latency

Average single-URL prediction latency (model only, excluding HTTP overhead): **15.210 ms**

## Top Feature Importances

| Rank | Feature | Importance |
|---|---|---|
| 1 | hostname_has_digits | 0.1917 |
| 2 | digit_ratio | 0.1170 |
| 3 | uses_https | 0.1033 |
| 4 | sensitive_keyword_count | 0.0840 |
| 5 | tld_length | 0.0650 |
| 6 | ratio_url_to_hostname | 0.0598 |
| 7 | num_hyphens | 0.0591 |
| 8 | url_entropy | 0.0412 |
| 9 | is_known_shortener | 0.0364 |
| 10 | hostname_length | 0.0335 |
| 11 | num_query_params | 0.0283 |
| 12 | letter_ratio | 0.0280 |
| 13 | special_char_count | 0.0249 |
| 14 | url_length | 0.0245 |
| 15 | num_subdomains | 0.0216 |

## Feature Engineering Rationale

Features fall into four groups, chosen because they reflect well-documented phishing URL construction patterns (see references in README):

- **Lexical/character-level** (length, digit ratio, entropy): phishing URLs tend to be longer and noisier than legitimate ones because attackers pack brand names, keywords, and random tokens into a single hostname/path.
- **Structural** (dot/hyphen counts, subdomain depth, query param count): excessive subdomains and hyphens are a classic way to squat on a brand name (e.g. `paypal-secure-login.evil.tk`).
- **Suspicious-pattern flags** (IP-as-host, `@` symbol, known shorteners, punycode, suspicious TLDs, sensitive keyword stuffing): each corresponds to a specific, well-known phishing technique for hiding or spoofing the real destination.
- **Security hygiene** (HTTPS usage, url-to-hostname ratio): legitimate high-traffic sites overwhelmingly use HTTPS and have a stable domain relative to path length; phishing kits are frequently deployed without valid TLS or on long, disposable hostnames.

No WHOIS/DNS/content-fetch features are used, by design: this keeps feature extraction fast (<1ms) and lets the API score URLs it has never made a network request to, which matters for a real-time endpoint.
