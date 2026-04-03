# Bot or Not - AR Force

Détecteur de bots pour la compétition **Bot or Not** (McGill Network Dynamics Lab, 2026).

## Approche

Le détecteur utilise un **Gradient Boosting Classifier** entraîné sur les 6 datasets de pratique fournis. Il repose sur des **features comportementales et textuelles** extraites des posts de chaque compte, sans analyse NLP lourde, ce qui le rend **langage-agnostique** (fonctionne en EN et en FR avec le même modèle).

## Features utilisées

### Timing
| Feature | Description |
|---|---|
| `t_cv` | Coefficient de variation des intervalles entre posts: bots postent à rythme régulier |
| `t_var` | Variance des intervalles entre posts |
| `t_min` | Intervalle minimum entre deux posts |
| `burst_ratio` | Proportion d'intervalles < 60 secondes |
| `ultra_fast_ratio` | Proportion d'intervalles < 10 secondes |
| `night_posts` | Proportion de posts entre 0h et 6h |

### Texte
| Feature | Description |
|---|---|
| `avg_hashtags` | Nombre moyen de hashtags par tweet |
| `hashtag_ratio` | Proportion de tweets contenant un hashtag |
| `link_ratio` | Proportion de tweets contenant un lien |
| `rt_ratio` | Proportion de retweets |
| `avg_len` | Longueur moyenne des tweets |
| `short_ratio` | Proportion de tweets < 50 caractères |
| `caps_ratio` | Proportion de mots en MAJUSCULES |
| `unique_text_ratio` | Proportion de tweets uniques (détecte le contenu copié-collé) |
| `same_consecutive` | Proportion de tweets identiques consécutifs |
| `avg_consec_sim` | Similarité Jaccard moyenne entre tweets consécutifs |
| `emoji_ratio` | Proportion de tweets contenant des emojis fréquents |

### Métadonnées
| Feature | Description |
|---|---|
| `z_score` | Z-score du tweet_count fourni dans les métadonnées |
| `tweet_count` | Nombre total de tweets dans le dataset |
| `has_desc` | Le compte a-t-il une bio ? |
| `desc_len` | Longueur de la bio |
| `hour_unique_r` | Diversité des heures de publication |
| `lang_diversity` | Diversité des langues détectées dans les tweets |

## Scoring et seuil

La règle de score de la compétition (**+2 TP, -2 FN, -6 FP**) pénalise fortement les faux positifs. Le seuil de décision est fixé à **0.76**, calibré par leave-one-out cross-validation sur les 6 datasets de pratique pour maximiser le score tout en minimisant les faux positifs.

## Structure du projet

```
bots_detector/
├── src/
│   └── detector.py
├── data/
│   ├── dataset.posts&users.1.json
│   ├── dataset.bots.1.txt
│   └── ... (jusqu'à 6)
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

```bash
python src/detector.py --dataset data/dataset.posts&users.1.json --team AR_Force
```

Options :
- `--dataset` : chemin vers le JSON du dataset à analyser
- `--team` : nom de l'équipe (utilisé pour nommer le fichier de sortie)
- `--lang` : `en` ou `fr` - optionnel, auto-détecté depuis le dataset

## Sortie

Le script génère un fichier `AR_Force.detections.en.txt` (ou `.fr.txt`) contenant un user ID par ligne - format identique aux fichiers `dataset.bots.X.txt` fournis.

## Différences EN / FR

Le même modèle est utilisé pour les deux langues. Les features étant comportementales et non linguistiques, aucun traitement spécifique par langue n'est appliqué. Le modèle est entraîné sur tous les datasets disponibles (EN + FR combinés).