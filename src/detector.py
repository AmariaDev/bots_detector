#!/usr/bin/env python3
"""
Bot or Not - Detector
Team: AR Force

Usage:
    python detector.py --dataset path/to/dataset.json --team TEAM_NAME [--lang en|fr]

Outputs:
    TEAM_NAME.detections.en.txt  or  TEAM_NAME.detections.fr.txt
"""

import json
import datetime
import statistics
import re
import numpy as np
import argparse
import os
import sys
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier

# ─────────────────────────────────────────────
# TRAINING DATA PATHS (practice datasets)
# ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DATASETS = [
    (os.path.join(SCRIPT_DIR, "data", f"dataset.posts&users.{i}.json"),
     os.path.join(SCRIPT_DIR, "data", f"dataset.bots.{i}.txt"))
    for i in range(1, 7)
]


# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────
def extract_features(uid, users, user_posts):
    u = users[uid]
    plist = user_posts[uid]
    desc = u.get("description") or ""

    texts = [p["text"] for p in plist]

    # ─────────────────────────
    # TIMING
    # ─────────────────────────
    if len(plist) >= 2:
        times = sorted(
            datetime.datetime.fromisoformat(p["created_at"].replace("Z", "+00:00"))
            for p in plist
        )
        intervals = [(times[i + 1] - times[i]).total_seconds() for i in range(len(times) - 1)]

        t_mean = statistics.mean(intervals)
        t_var = statistics.variance(intervals) if len(intervals) > 1 else 0
        t_cv = (t_var ** 0.5 / t_mean) if t_mean > 0 else 0

        burst_ratio = sum(1 for iv in intervals if iv < 30) / len(intervals)
        ultra_fast_ratio = sum(1 for iv in intervals if iv < 10) / len(intervals)
        t_min = min(intervals)
    else:
        t_mean = t_var = t_cv = burst_ratio = ultra_fast_ratio = t_min = 0

    # ─────────────────────────
    # TEXT BASIQUE
    # ─────────────────────────
    avg_len = statistics.mean(len(t) for t in texts) if texts else 0
    link_ratio = sum(1 for t in texts if "http" in t) / len(texts) if texts else 0
    hashtag_ratio = sum(1 for t in texts if "#" in t) / len(texts) if texts else 0
    rt_ratio = sum(1 for t in texts if t.startswith("RT ")) / len(texts) if texts else 0

    # ─────────────────────────
    # 💣 FEATURES KILLER
    # ─────────────────────────
    if texts:
        unique_text_ratio = len(set(texts)) / len(texts)
        duplicate_ratio = 1 - unique_text_ratio

        same_consecutive = sum(
            1 for i in range(len(texts)-1) if texts[i] == texts[i+1]
        ) / len(texts) if len(texts) > 1 else 0

        avg_hashtags = statistics.mean(len(re.findall(r"#\w+", t)) for t in texts)

        short_ratio = sum(1 for t in texts if len(t) < 40) / len(texts)

        caps_ratio = statistics.mean(
            sum(1 for w in t.split() if w.isupper() and len(w) > 1) / max(len(t.split()), 1)
            for t in texts
        )

        emoji_ratio = sum(
            1 for t in texts if any(c in t for c in "🚀🔥💰😂")
        ) / len(texts)

    else:
        unique_text_ratio = duplicate_ratio = same_consecutive = 0
        avg_hashtags = short_ratio = caps_ratio = emoji_ratio = 0

    # ─────────────────────────
    # DIVERSITÉ TEMPORELLE
    # ─────────────────────────
    if plist:
        hours = [
            datetime.datetime.fromisoformat(p["created_at"].replace("Z", "+00:00")).hour
            for p in plist
        ]
        hour_unique_r = len(set(hours)) / len(hours)
    else:
        hour_unique_r = 0

    return [
        u["z_score"],
        u["tweet_count"],

        # timing
        t_mean,
        t_var,
        t_cv,
        burst_ratio,
        ultra_fast_ratio,
        t_min,

        # text
        avg_len,
        link_ratio,
        hashtag_ratio,
        rt_ratio,
        avg_hashtags,
        short_ratio,
        caps_ratio,

        # killer
        unique_text_ratio,
        duplicate_ratio,
        same_consecutive,
        emoji_ratio,

        # meta
        1 if desc.strip() else 0,
        len(desc),
        hour_unique_r
    ]


def load_dataset(json_path, bots_path=None):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    users = {u["id"]: u for u in data["users"]}
    user_posts = defaultdict(list)
    for p in data["posts"]:
        user_posts[p["author_id"]].append(p)

    bots = set()
    if bots_path and os.path.exists(bots_path):
        with open(bots_path) as f:
            bots = set(line.strip() for line in f if line.strip())

    return users, user_posts, bots, data.get("lang", "en")


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
def build_training_data(exclude_lang=None):
    all_X, all_y = [], []
    loaded = 0
    for json_path, bots_path in TRAIN_DATASETS:
        if not os.path.exists(json_path) or not os.path.exists(bots_path):
            continue
        users, user_posts, bots, lang = load_dataset(json_path, bots_path)
        if exclude_lang and lang == exclude_lang:
            continue
        for uid in users:
            all_X.append(extract_features(uid, users, user_posts))
            all_y.append(1 if uid in bots else 0)
        loaded += 1

    if loaded == 0:
        print("[ERROR] No training datasets found in data/ folder.", file=sys.stderr)
        sys.exit(1)

    return np.array(all_X), np.array(all_y)


def train_model(X, y):
    clf = GradientBoostingClassifier(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    clf.fit(X, y)
    return clf


# ─────────────────────────────────────────────
# THRESHOLD SELECTION
# based on scoring: +2 TP, -2 FN, -6 FP
# we optimise on training data for a safe threshold
# ─────────────────────────────────────────────
def find_best_threshold(clf, X, y):
    proba = clf.predict_proba(X)[:, 1]
    best_score, best_thresh = -9999, 0.5
    for thresh in np.arange(0.2, 0.98, 0.01):
        pred = (proba >= thresh).astype(int)
        tp = np.sum((pred == 1) & (y == 1))
        fn = np.sum((pred == 0) & (y == 1))
        fp = np.sum((pred == 1) & (y == 0))
        score = 2 * tp - 2 * fn - 6 * fp
        if score > best_score:
            best_score, best_thresh = score, thresh
    return best_thresh


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Bot or Not Detector")
    parser.add_argument("--dataset", required=True, help="Path to the evaluation dataset JSON")
    parser.add_argument("--team", required=True, help="Your team name")
    parser.add_argument("--lang", choices=["en", "fr"], default=None,
                        help="Override language (auto-detected from dataset if omitted)")
    args = parser.parse_args()

    # Load evaluation dataset
    print(f"[*] Loading evaluation dataset: {args.dataset}")
    users, user_posts, _, detected_lang = load_dataset(args.dataset)
    lang = args.lang or detected_lang
    print(f"[*] Language: {lang} | Users: {len(users)}")

    # Train (exclude same-language datasets to avoid leakage if desired — here we use all)
    print("[*] Training model on practice datasets...")
    X_train, y_train = build_training_data()
    clf = train_model(X_train, y_train)

    # Find best threshold on training data
    thresh = find_best_threshold(clf, X_train, y_train)
    print(f"[*] Optimal threshold: {thresh:.2f}")

    # Predict on evaluation dataset
    uids = list(users.keys())
    X_eval = np.array([extract_features(uid, users, user_posts) for uid in uids])
    proba = clf.predict_proba(X_eval)[:, 1]

    detected_bots = [uid for uid, p in zip(uids, proba) if p >= thresh]
    print(f"[*] Detected {len(detected_bots)} bot accounts out of {len(uids)}")

    # Write output file
    out_filename = f"{args.team}.detections.{lang}.txt"
    with open(out_filename, "w") as f:
        for uid in detected_bots:
            f.write(uid + "\n")

    print(f"[*] Output written to: {out_filename}")


if __name__ == "__main__":
    main()
