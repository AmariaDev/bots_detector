#!/usr/bin/env python3
"""
Bot or Not 2026 - Détecteur de comptes bots
Équipe : AR Force

Usage:
    python src/detector.py --dataset path/to/dataset.json --team AR_Force [--lang en|fr]

Sortie:
    AR_Force.detections.en.txt  ou  AR_Force.detections.fr.txt
"""

import json
import re
import os
import sys
import argparse
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DATASETS = [
    (os.path.join(SCRIPT_DIR, "data", f"dataset.posts&users.{i}.json"),
     os.path.join(SCRIPT_DIR, "data", f"dataset.bots.{i}.txt"))
    for i in range(1, 7)
]

# ── Vocabulaire suspects ───────────────────────────────────────────────────────
SUSPICIOUS = list(set([
    "buy","free","click","win","offer","deal","discount","promo","cash","prize",
    "earn","money","fast","cheap","limited","exclusive","guaranteed","profit",
    "crypto","bitcoin","investment","urgent","subscribe","follow","retweet",
    "share","viral","trending","income","rich","giveaway","gift","bonus",
    "reward","claim","selected","congratulations","lucky","special","instant",
    "double","million","forex","trading","passive","signal","wallet","nft",
    "acheter","achetez","gratuit","cliquez","gagner","gagnez","offre","solde",
    "rabais","argent","vite","rapide","limité","exclusif","garanti","investissement",
    "abonnez","suivez","partagez","tendance","revenus","riche","concours","cadeau",
    "récompense","réclamez","sélectionné","félicitations","chanceux","spécial",
    "doublez","arnaque","opportunité","portefeuille",
]))

LLM_PHRASES = [
    "here is", "here's a", "certainly", "as an ai", "modified version",
    "slightly modified", "feel the fear", "in conclusion", "furthermore",
    "moreover", "it is worth noting", "i'd like to", "i would like to",
    "as a language", "i am an ai", "je suis une ia", "en conclusion",
    "il convient de noter", "il est important de",
    "here are some", "here are a few", "sure, here",
    # Nouveaux : variantes françaises de réponses LLM
    "voici une légère modification", "voici quelques-uns de mes",
    "voici une version légèrement", "légère modification",
    "voici mes derniers tweets", "voici quelques tweets",
    "version modifiée", "slightly different version",
    "here is a slightly", "here is a modified",
]

SPORTS_WORDS = {'nhl','nba','hockey','basketball','game','player','team','score',
                'goal','puck','court','dunk','playoff','league','draft'}
MUSIC_WORDS  = {'pop','music','song','album','artist','concert','lyrics','singer',
                'track','banger','playlist','stream','spotify','release','hit'}
MOVIE_WORDS  = {'movie','film','cinema','watch','director','actor','scene','plot',
                'sequel','oscar','streaming','series','episode','show','cast'}

URL_PAT         = re.compile(r'https?://\S+|www\.\S+')
HASHTAG_PAT     = re.compile(r'#\w+')
MENTION_PAT     = re.compile(r'@\w+')
DIGIT_NOISE_PAT = re.compile(r'[a-zA-Z]\d{1,2}[a-zA-Z]')
RANDOM_USER_PAT = re.compile(r'[A-Z][a-z]+[A-Z][a-z]+\d{2,}|[a-z]+[A-Z][a-z]+\d{2,}')
DASH_DIAG_PAT   = re.compile(r'^-\s+[^\-\s]')
LIST_NOT_PAT    = re.compile(r"^\[[\'\"]")
FOLLOW_CMD_PAT  = re.compile(r'\(DOIT ME SUIVRE\)|\(MUST FOLLOW\)', re.I)
OVERRATED_PAT   = re.compile(r'is .+ overrated', re.I)

TWITTER_CONFUSED_PAT = re.compile(
    r"\b(hashtag|retweet|dm|direct message|followers?|following|timeline|notification|tweet|twitter|"
    r"abr[eé]viations?|bouton pour|fil d'actualit[eé]|lien.{0,15}aller|cliqu[ée]r|"
    r"qui est ce|qu'est.ce que cela|qu'est.ce que c'est)\b",
    re.I
)
SOCIAL_MECHANICS_PAT = re.compile(
    r"(how do (i|you|one)|can (i|one|you)|do (i|you)|what (is|are)|does one|explain|understand|figure out|"
    r"pourquoi|qu'est.ce|comment|quelqu'un pourrait|quelqu'un peut|pouvez.vous m'expliquer).{0,40}"
    r"(hashtag|tweet|dm|follow|retweet|mention|notification|profile|timeline|"
    r"lien|bouton|abr[eé]v|cliqu|fil|populaire\?|signifie)",
    re.I
)

BOT_TEMPLATES = [
    re.compile(r'voici (quelques|mes|une).*(tweets|messages|modification)|here are (some|my) (recent )?tweets', re.I),
    re.compile(r'voici une (l[eé]g[eè]re|l[eé]g[eè]rement).*(modification|version)', re.I),
    re.compile(r'trouvé.{0,20}(note|mot).{0,20}(porte|door)|found.{0,20}note.{0,20}(door|porte)', re.I),
    re.compile(r'(brulé|brûlé|burned|burnt).{0,20}(langue|tongue)|tongue.{0,20}(coffee|café)', re.I),
    re.compile(r'(chanté|sang|belted).{0,30}(mauvais|wrong|faux).{0,30}(paroles|lyrics)|wrong lyrics', re.I),
    re.compile(r'(appuyé|pressed?).{0,20}snooze|snooze.{0,20}(fois|times)', re.I),
    re.compile(r'(souhaité|envoyé|texted|sent).{0,30}(anniversaire|birthday).{0,30}(mauvais|wrong|pas le bon)', re.I),
]


# ── Parsing date ───────────────────────────────────────────────────────────────
def _parse_dt(s):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


# ── Extraction de features ─────────────────────────────────────────────────────
def extract_features(uid, users, user_posts):
    user  = users[uid]
    posts = user_posts[uid]
    texts = [str(p.get("text", "")) for p in posts]
    times = [p.get("created_at", "") for p in posts]
    n     = max(len(texts), 1)
    desc  = str(user.get("description", "") or "")

    def tweet_stats(t):
        low   = t.lower()
        words = t.split()
        L     = max(len(t), 1)
        s     = t.strip()
        return {
            "has_url":          int(bool(URL_PAT.search(low))),
            "n_hashtags":       len(HASHTAG_PAT.findall(t)),
            "n_mentions":       len(MENTION_PAT.findall(t)),
            "n_exclaim":        t.count("!"),
            "uppercase_ratio":  sum(1 for c in t if c.isupper()) / L,
            "word_count":       len(words),
            "unique_ratio":     len(set(w.lower() for w in words)) / max(len(words), 1),
            "suspicious_count": sum(1 for w in SUSPICIOUS if w in low),
            "llm_score":        sum(1 for p in LLM_PHRASES if p in low),
            "tweet_len":        len(t),
            "n_digits":         sum(1 for c in t if c.isdigit()),
            "question_tmpl":    int(bool(OVERRATED_PAT.search(low))),
            "sports_topic":     int(any(w in low for w in SPORTS_WORDS)),
            "music_topic":      int(any(w in low for w in MUSIC_WORDS)),
            "movie_topic":      int(any(w in low for w in MOVIE_WORDS)),
            "digit_noise":      int(bool(DIGIT_NOISE_PAT.search(t))),
            "dash_dialogue":    int(bool(DASH_DIAG_PAT.match(s))),
            "list_notation":    int(bool(LIST_NOT_PAT.match(s))),
            "follow_cmd":       int(bool(FOLLOW_CMD_PAT.search(t))),
            "bot_template":     int(any(p.search(t) for p in BOT_TEMPLATES)),
            "is_rt":            int(t.startswith("RT ")),
            "is_short":         int(len(t) < 50),
        }

    stats = [tweet_stats(t) for t in texts]
    keys  = list(stats[0].keys())
    agg   = {}
    for k in keys:
        vals = [s[k] for s in stats]
        agg[f"{k}_mean"] = np.mean(vals)
        agg[f"{k}_max"]  = np.max(vals)
        agg[f"{k}_std"]  = np.std(vals)

    # Intervalles temporels
    dts = sorted([dt for dt in (_parse_dt(t) for t in times if t) if dt is not None])
    if len(dts) >= 2:
        gaps            = [(dts[i+1] - dts[i]).total_seconds() for i in range(len(dts)-1)]
        gap_mean        = float(np.mean(gaps))
        gap_std         = float(np.std(gaps))
        gap_regularity  = gap_std / max(gap_mean, 1)
        posting_regular = 1.0 / (1.0 + gap_regularity)
        gap_min         = float(np.min(gaps))
        gap_max         = float(np.max(gaps))
        burst_ratio     = sum(1 for g in gaps if g < 60) / len(gaps)
    else:
        gap_mean = gap_std = gap_regularity = gap_min = gap_max = burst_ratio = 0.0
        posting_regular = 0.5

    # Heures de publication
    if dts:
        hours       = [dt.hour for dt in dts]
        night_posts = sum(1 for h in hours if h < 6) / len(hours)
        hour_diversity = len(set(hours)) / len(hours)
    else:
        night_posts = hour_diversity = 0.0

    # Diversité langue des tweets
    langs          = [p.get("lang", "") for p in posts]
    lang_diversity = len(set(langs)) / max(len(langs), 1)

    # Contenu
    normalized        = [re.sub(r'https?://\S+', 'URL', t.lower().strip()) for t in texts]
    content_diversity = len(set(normalized)) / n
    all_words         = " ".join(texts).lower().split()
    vocab_richness    = len(set(all_words)) / max(len(all_words), 1)

    exact_counts   = Counter(t.strip().lower() for t in texts)
    max_exact_rep  = exact_counts.most_common(1)[0][1] if exact_counts else 0
    exact_rep_rate = sum(1 for t in texts if exact_counts[t.strip().lower()] > 1) / n
    high_rep_rate  = sum(1 for t in texts if exact_counts[t.strip().lower()] >= 3) / n

    tfidf_avg_sim = tfidf_max_sim = 0.0
    if len(texts) >= 2:
        try:
            vec  = TfidfVectorizer(max_features=300, min_df=1)
            mat  = vec.fit_transform(texts)
            sims = cosine_similarity(mat)
            np.fill_diagonal(sims, 0)
            tfidf_avg_sim = float(sims.mean())
            tfidf_max_sim = float(sims.max())
        except Exception:
            pass

    has_sports      = sum(s["sports_topic"] for s in stats)
    has_music       = sum(s["music_topic"]  for s in stats)
    has_movies      = sum(s["movie_topic"]  for s in stats)
    topic_mix_score = (min(has_sports, has_music) + min(has_sports, has_movies)
                       + min(has_music, has_movies)) / n

    bot_template_rate  = sum(s["bot_template"]  for s in stats) / n
    dash_dialogue_rate = sum(s["dash_dialogue"] for s in stats) / n
    llm_rate           = sum(s["llm_score"]     for s in stats) / n

    # Jaccard similarité consécutive
    def jaccard(a, b):
        sa, sb = set(a.lower().split()), set(b.lower().split())
        return len(sa & sb) / len(sa | sb) if sa | sb else 0.0

    consec_sim = float(np.mean([jaccard(texts[i], texts[i+1])
                                for i in range(len(texts)-1)])) if len(texts) >= 2 else 0.0

    # Profil
    z_score         = float(user.get("z_score", 0) or 0)
    tweet_count     = float(user.get("tweet_count", n) or n)
    desc_len        = len(desc)
    desc_susp       = sum(1 for w in SUSPICIOUS if w in desc.lower())
    has_location    = int(bool(user.get("location", "")))
    has_desc        = int(bool(desc.strip()))
    username        = str(user.get("username", "") or "")
    random_username = int(bool(RANDOM_USER_PAT.search(username)))

    all_ht = []
    for t in texts:
        all_ht.extend(HASHTAG_PAT.findall(t.lower()))
    top_ht_freq       = (Counter(all_ht).most_common(1)[0][1] / n) if all_ht else 0.0
    hashtag_diversity = len(set(all_ht)) / max(len(all_ht), 1)

    # Persona "grandpa apprenant internet" : beaucoup de questions, peu d'argot
    SLANG_PAT2 = re.compile(r'\b(lol|omg|wtf|lmao|tbh|ngl|idk|bruh|mdr|ptdr|jsuis|ptdrrr|mdrrr)\b', re.I)
    q_rate     = sum(1 for t in texts if t.count('?') >= 1) / n
    slang_rate = sum(1 for t in texts if SLANG_PAT2.search(t)) / n
    grandpa_score = q_rate * (1.0 - slang_rate)  # élevé = bot grandpa persona

    # Spam de hashtag promotionnel répété (même hashtag dans >50% des tweets)
    all_ht2 = []
    for t in texts:
        all_ht2.extend(HASHTAG_PAT.findall(t.lower()))
    promo_ht_rate = 0.0
    if all_ht2:
        top_ht_count = Counter(all_ht2).most_common(1)[0][1]
        promo_ht_rate = top_ht_count / n  # si >0.5 : même hashtag dans 50%+ des tweets

    # ── Nouveau : répétition de trigrammes ───────────────────────────────────────
    def get_trigrams(text):
        words = re.sub(r'[^a-zA-ZÀ-ÿ ]', '', text.lower()).split()
        return [' '.join(words[i:i+3]) for i in range(len(words) - 2)]

    all_trigrams = []
    for t in texts:
        all_trigrams.extend(get_trigrams(t))
    if all_trigrams:
        top_tri_count = Counter(all_trigrams).most_common(1)[0][1]
        top_trigram_rate = top_tri_count / n          # même 3-gramme dans X% des tweets
        trigram_diversity = len(set(all_trigrams)) / len(all_trigrams)
    else:
        top_trigram_rate = trigram_diversity = 0.0

    # ── Nouveau : persona confus sur Twitter (SirReginald style) ─────────────────
    twitter_conf_rate = sum(1 for t in texts if TWITTER_CONFUSED_PAT.search(t)) / n
    social_mech_rate  = sum(1 for t in texts if SOCIAL_MECHANICS_PAT.search(t)) / n
    all_question_rate = sum(1 for t in texts if t.strip().endswith('?') or
                            (t.count('?') >= 1 and t.count('.') == 0 and len(t) < 200)) / n
    # Score composite "grandpa confus sur Twitter"
    twitter_confusion_score = (twitter_conf_rate * 0.5 + social_mech_rate * 0.3
                               + all_question_rate * 0.2)

    agg_vals = np.array([agg[k] for k in sorted(agg.keys())], dtype=np.float32)

    extra = np.array([
        posting_regular, gap_mean, gap_std, gap_regularity, gap_min, gap_max,
        burst_ratio, night_posts, hour_diversity, lang_diversity,
        content_diversity, vocab_richness,
        tfidf_avg_sim, tfidf_max_sim,
        topic_mix_score,
        llm_rate, dash_dialogue_rate, bot_template_rate,
        int(bot_template_rate > 0),
        exact_rep_rate, float(max_exact_rep), high_rep_rate,
        consec_sim,
        z_score, tweet_count,
        desc_len, desc_susp, has_location, has_desc,
        top_ht_freq, hashtag_diversity, random_username,
        float(n),
        grandpa_score,
        promo_ht_rate,
        top_trigram_rate, trigram_diversity,
        twitter_confusion_score, twitter_conf_rate, social_mech_rate, all_question_rate,
    ], dtype=np.float32)

    return np.concatenate([agg_vals, extra])


# ── Chargement dataset ─────────────────────────────────────────────────────────
def load_dataset(json_path, bots_path=None):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    users      = {str(u["id"]): u for u in data["users"]}
    user_posts = defaultdict(list)
    for p in data["posts"]:
        user_posts[str(p["author_id"])].append(p)
    bots = set()
    if bots_path and os.path.exists(bots_path):
        with open(bots_path, encoding="utf-8") as f:
            bots = {line.strip() for line in f if line.strip()}
    return users, user_posts, bots, data.get("lang", "en")


# ── Modèle ─────────────────────────────────────────────────────────────────────
def build_model(scale_pos_weight=1.0):
    lr  = LogisticRegression(max_iter=5000, C=1.0, solver="saga", class_weight="balanced")
    svm = CalibratedClassifierCV(LinearSVC(max_iter=3000, C=1.0, class_weight="balanced"), cv=3)
    rf  = RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=42, n_jobs=-1)
    xgb = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.03,
                        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                        random_state=42, n_jobs=-1, scale_pos_weight=scale_pos_weight)
    return VotingClassifier(
        estimators=[("lr", lr), ("svm", svm), ("rf", rf), ("xgb", xgb)],
        voting="soft", weights=[1, 1, 2, 3],
    )


def competition_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return 2*tp - 2*fn - 6*fp, int(tp), int(fp), int(fn)


def find_best_threshold(probas_cv, y_true):
    best_score, best_t = -9999, 0.5
    for t in np.linspace(0.30, 0.85, 111):
        preds = (probas_cv >= t).astype(int)
        s, _, _, _ = competition_score(y_true, preds)
        if s > best_score:
            best_score, best_t = s, float(t)
    return best_t


# ── Pipeline principal ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Bot or Not - AR Force")
    parser.add_argument("--dataset", required=True, help="Chemin vers le dataset JSON final")
    parser.add_argument("--team",    required=True, help="Nom de l'équipe")
    parser.add_argument("--lang",    choices=["en", "fr"], default=None)
    args = parser.parse_args()

    # Chargement des données d'entraînement
    print("[*] Chargement des datasets de pratique...")
    all_X, all_y = [], []
    for json_path, bots_path in TRAIN_DATASETS:
        if not os.path.exists(json_path) or not os.path.exists(bots_path):
            continue
        users, user_posts, bots, lang = load_dataset(json_path, bots_path)
        for uid in users:
            feats = extract_features(uid, users, user_posts)
            all_X.append(feats)
            all_y.append(1 if uid in bots else 0)
        print(f"  [OK] {os.path.basename(json_path)} [{lang}] — {len(users)} comptes, {len(bots)} bots")

    if not all_X:
        print("[ERREUR] Aucun dataset trouvé dans data/", file=sys.stderr)
        sys.exit(1)

    X_all = np.nan_to_num(np.array(all_X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y_all = np.array(all_y)
    print(f"\n[*] Total : {len(y_all)} comptes | {int(y_all.sum())} bots ({y_all.mean()*100:.1f}%)")

    # Normalisation
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_all)

    # Modèle
    n_bots   = int(y_all.sum())
    n_humans = len(y_all) - n_bots
    model    = build_model(scale_pos_weight=n_humans / max(n_bots, 1))

    # Validation croisée pour trouver le seuil optimal
    print("[*] Validation croisée (5-fold) pour optimiser le seuil...")
    skf       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probas_cv = cross_val_predict(model, X_sc, y_all, cv=skf, method="predict_proba")[:, 1]
    threshold = find_best_threshold(probas_cv, y_all)

    cv_preds = (probas_cv >= threshold).astype(int)
    sc, tp, fp, fn = competition_score(y_all, cv_preds)
    print(f"[*] Seuil optimal : {threshold:.2f} | TP={tp} FP={fp} FN={fn} | Score : {sc:+d}")

    # Entraînement final sur 100% des données
    print("[*] Entraînement final sur toutes les données...")
    model.fit(X_sc, y_all)

    # Prédiction sur le dataset final
    print(f"\n[*] Chargement du dataset final : {args.dataset}")
    users_eval, user_posts_eval, _, detected_lang = load_dataset(args.dataset)
    lang = args.lang or detected_lang
    print(f"[*] Langue : {lang} | Comptes : {len(users_eval)}")

    uids   = list(users_eval.keys())
    X_eval = np.nan_to_num(
        np.array([extract_features(uid, users_eval, user_posts_eval) for uid in uids], dtype=np.float32),
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    X_eval_sc  = scaler.transform(X_eval)
    bot_probas = model.predict_proba(X_eval_sc)[:, 1]
    detected   = [uid for uid, p in zip(uids, bot_probas) if p >= threshold]

    print(f"[*] Bots détectés : {len(detected)} / {len(uids)}")

    out_file = f"{args.team}.detections.{lang}.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        for uid in detected:
            f.write(uid + "\n")
    print(f"[*] Résultats écrits dans : {out_file}")


if __name__ == "__main__":
    main()