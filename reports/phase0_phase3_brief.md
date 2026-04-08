# Phase 0 & Phase 3 — Project Brief

---

## Phase 0: Data Preparation & Feature Engineering

### Objective
- Clean and filter raw Steam data into a modelling-ready dataset
- Construct a leakage-safe temporal train/test split
- Pre-compute all features consumed by Phases 1–3: LOO aggregates, multi-hot encodings, BERT embeddings, user genre vectors, and domain-informed synthetic negatives

---

### Data Overview

**Raw input:**
- 239,664 applications, 1,048,148 reviews across 9 CSV tables

**After filtering pipeline:**

| Step | Reviews | Users | Games |
|---|---|---|---|
| Raw (type=game) | 865,782 | 600,542 | 85,529 |
| Users ≥ 3 reviews | 250,379 | 36,996 | 58,367 |
| Games ≥ 2 reviews | 224,895 | 36,840 | 32,883 |
| Iterative filtering (converged at iter 6) | **216,361** | **32,792** | **31,697** |

- Sparsity: **99.98%** — users have reviewed ~7 games out of 31,697
- Label distribution: 75.1% positive (`voted_up=True`), 24.9% negative
- Average reviews per user: 6.6 | per game: 6.8

**Temporal train/test split (per-user 80/20 on timestamp):**
- Train: 169,439 reviews | Test: 46,922 reviews
- All 32,792 users appear in both sets — every user has at least 1 held-out test review

---

### Key Steps

**Data Cleaning (`final_phase0_data_sampling.ipynb`)**
- Iterative filtering: after standard threshold filtering, some users drop below 3 reviews — the iterative pass re-applies both constraints in a loop until convergence, preventing silent data quality issues
- Canonicalised genres/categories to English (ASCII filter + minimum-count filter): 44 → 28 genres, 120 → 58 categories
- Capped `votes_funny` overflow at 99.9th percentile (214 values); capped price outliers at 99.9th percentile (299 values)
- Log-transformed `author_playtime_forever`, `mat_final_price` to reduce right skew
- Dropped `steam_purchase` (zero variance)
- Added a second canonicalisation pass to catch non-English entries missed in pass 1

**Feature Engineering (`final_phase0_feature_engineering.ipynb`)**

*Static game features (94-dim per game):*
- Genre multi-hot (28-dim), category multi-hot (58-dim), platform multi-hot (3-dim)
- Scalar features: `is_free`, `required_age`, `metacritic_score` (imputed with genre-median), `mat_final_price_log`, `publisher_portfolio_size`

*Leakage-safe LOO aggregate features (9 features):*
- For each (user, game) pair, all aggregates exclude that pair's own label to prevent leakage
- Test aggregates additionally only use training interactions before the test timestamp
- Features: `user_positive_ratio`, `user_avg_playtime`, `user_review_count_loo`, `user_price_preference`, `game_positive_ratio`, `game_avg_playtime`, `game_review_count_loo`, `playtime_vs_user_avg`, `developer_avg_rating`

*Domain-informed negative sampling:*
- 4 synthetic negatives per positive interaction → 673,048 negatives total
- Genre-proportional sampling: negatives sampled proportional to each user's genre history (harder than random — forces model to discriminate within genres, not across them)
- Platform-compatible: only samples games available on platforms the user has reviewed on
- Two-stage rejection sampling with pre-computed cumulative weights: O(1) per sample
- `is_synthetic_negative` flag stored for downstream analysis

*BERT description embeddings (128-dim):*
- Encoded `short_description` of all games using `all-MiniLM-L6-v2` sentence transformer
- 1,014,105 rows × 128 dimensions (26.7 MB)

*User genre preference vectors (28-dim):*
- Computed from positive reviews in training data only, weighted by playtime, normalised
- 30,415 users × 28 genres (users with no positive reviews excluded)

---

### Output Files

| File | Shape | Consumed by |
|---|---|---|
| `train_with_features.parquet` | 169,439 × 36 | Phase 3 |
| `test_with_features.parquet` | 46,922 × 36 | Phase 3 |
| `negatives_with_features.parquet` | 673,048 × 14 | Phase 1, 3 |
| `game_features_static.parquet` | 31,697 × 94 | Phase 3 |
| `game_description_embeddings.parquet` | 1,014,105 × 128 | Phase 2, 3 |
| `user_genre_vectors.parquet` | 30,415 × 28 | Phase 3 |

---

### Questions for Phase 0

1. We ran Phase 3 twice — with and without iterative filtering + two-pass canonicalisation. Wide & Deep NDCG@10 dropped from 0.9667 → 0.4489 and best validation AUC-ROC collapsed to 0.6514 at epoch 1 (model barely trained) without the extra filtering. We attribute this to noisy LOO aggregates from borderline users (too few interactions for stable aggregation) and inconsistent multi-hot encodings from uncleaned genre labels. Is this the right explanation, or could something else account for such a drastic drop specifically in the deep model?

2. XGBoost was nearly unaffected by the absence of filtering (NDCG@10: 0.9588 → 0.9553), while Wide & Deep collapsed. Does this confirm that the extra filtering steps are specifically necessary for neural models relying on dense feature representations — and that tree-based models are inherently more robust to this kind of data noise?

3. Iterative filtering removed ~8,000 reviews and ~4,000 users beyond the initial threshold pass. Given the demonstrated impact on model quality, is this reduction justified — or should we be concerned about further constraining an already sparse dataset (99.98%)?

4. User genre preference vectors are computed from positive reviews only, weighted by playtime. Is playtime-weighting a justifiable proxy for preference strength — or could high playtime on a disliked game (sunk cost) introduce noise?

---

## Phase 3: Wide & Deep Classification Model

### Objective
- Predict `voted_up` for (user, game) pairs using all available signals: collaborative embeddings (Phase 1), content embeddings (Phase 2), and behavioural aggregates (Phase 0)
- Compare against Logistic Regression and XGBoost baselines
- Produce predicted probability scores for Phase 4 hybrid stacking

---

### Data Overview

**Training set:**
- 842,487 samples = `train_with_features.parquet` (real reviews) + `negatives_with_features.parquet` (synthetic negatives)
- 15% positive, 85% negative
- The 15/85 split ≠ the 1:4 sampling ratio because real training reviews also contain natural negatives (~24% of real reviews are `voted_up=False`), pushing the overall negative proportion higher than 80%

**Validation set:**
- 10% of real training reviews (natural distribution, ~75% positive) — used for early stopping and threshold tuning only

**Test set:**
- 46,922 real reviews (76.2% positive, natural Steam distribution) — used for final evaluation only

---

### Feature Setup

**Feature matrix (~326 columns per row):**

| Feature group | Dims | Source |
|---|---|---|
| NCF MLP user embeddings | 32 | Phase 1 |
| NCF MLP item embeddings | 32 | Phase 1 |
| BERT description embeddings | 128 | Phase 0 |
| User genre preference vector | 28 | Phase 0 |
| Game genre multi-hot | 28 | Phase 0 |
| Game category multi-hot | 58 | Phase 0 |
| Platform multi-hot | 3 | Phase 0 |
| Game scalar features | 5 | Phase 0 |
| LOO aggregates | 9 | Phase 0 |
| Cross features | 3 | Phase 3 (derived) |

**Wide pathway (12 features):** 9 LOO aggregates + 3 cross features
- `genre_match_score` = dot product of user genre history × game genre vector
- `price_gap` = |user_price_preference − game price|
- `free_match` = is_free × user price sensitivity

**Deep pathway (323 features):** all of the above including embeddings

---

### Model Architecture

```
WIDE:     12 features → Linear(1)
DEEP:     323 features → Dense(256) → BN+ReLU+Dropout(0.3)
                       → Dense(128) → BN+ReLU+Dropout(0.3)
                       → Dense(64)  → BN+ReLU
COMBINED: [wide(1) || deep(64)] → Dense(32) → ReLU → Dropout(0.2) → Dense(1) → Sigmoid
```

- Total trainable parameters: **127,150**
- Wide = memorisation (direct associations, e.g. genre match → positive)
- Deep = generalisation (complex patterns across embedding space)

**Training setup:**
- Loss: weighted BCE, positive class weight = 5.65 (compensates for 85% negative training distribution)
- Optimiser: Adam (lr=0.001, weight_decay=0.0001)
- Scheduler: CosineAnnealingLR
- Early stopping: patience=5 on validation AUC-ROC
- Best checkpoint: Epoch 39 (Val AUC-ROC=0.7980, Val AUC-PR=0.8982)

---

### Performance Results

**Classification** (46,922 real test reviews):

| Model | AUC-ROC | AUC-PR | F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.6904 | 0.8482 | 0.8697 | — | — | — |
| XGBoost | 0.7385 | **0.9022** | 0.8653 | — | — | — |
| **Wide & Deep** | **0.7577** | 0.8824 | 0.8634 | 0.8447 | 0.8829 | 0.79 |

**Ranking** (1 positive + 99 random negatives per user, 25,793 users):

| Model | HR@10 | NDCG@10 |
|---|---|---|
| Logistic Regression | 0.9458 | 0.7592 |
| XGBoost | 0.9929 | 0.9588 |
| **Wide & Deep** | **0.9940** | **0.9667** |

**Per-class report (Wide & Deep, threshold=0.8730):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Negative (0) | 0.56 | 0.48 | 0.52 | 11,144 |
| Positive (1) | 0.84 | 0.88 | 0.86 | 35,778 |

---

### Key Insights

- **Ranking > classification for this task.** AUC-ROC of 0.76 is moderate, but HR@10 of 0.994 means the liked game lands in the top 10 out of 100 candidates 99.4% of the time. A recommender needs to rank correctly, not just classify correctly.
- **LR → XGBoost gap is large** (NDCG: 0.76 → 0.96) because LR cannot model non-linear interactions between high-dimensional embeddings. XGBoost discovers these automatically via tree splits.
- **W&D marginal gain over XGBoost** (NDCG: 0.96 → 0.97) shows embeddings add genuine signal, but tabular LOO features are already very powerful.
- **XGBoost wins AUC-PR** (0.9022 vs 0.8824) — likely because it discovers feature interactions automatically, while W&D's wide component relies on only 3 manually chosen cross features.
- **Negative class F1=0.52 is expected and acceptable.** Both the positive class weight (5.65×) and the embedding space (learned from 76% positive reviews) bias the model toward positive predictions. But HR@10=0.994 shows the positive signal is strong enough to dominate the ranking even with a noisy candidate pool — rejecting negatives perfectly is not necessary for a recommender to work.

---

### Questions for Phase 3

1. Without extra filtering, Wide & Deep's best validation AUC-ROC was 0.6514 at epoch 1 — meaning early stopping triggered almost immediately and the model learned almost nothing. With filtering it reached 0.7980 at epoch 39. What does it mean mechanistically when a deep model early-stops at epoch 1 — is it a data quality issue, a learning rate issue, or something else?

2. The results show XGBoost is robust to the filtering difference (NDCG@10 barely changes) but Wide & Deep is not. Is this expected behaviour given that gradient-based deep models are more sensitive to feature noise than tree-based models — and does it validate the filtering as a necessary preprocessing step specifically for neural architectures?

3. Without filtering, Wide & Deep's recall jumps to 0.9725 (vs 0.8829 with filtering) while precision drops to 0.7710. The model predicts almost everything as positive. Is this symptomatic of a degenerate solution where the model learned to always predict the majority class rather than learning real patterns?

4. The ranking evaluation uses 99 randomly sampled negatives per user. Our training negatives are genre-informed (harder), while evaluation negatives are random (easier). Does this mismatch mean HR@10 overestimates real-world ranking quality — and is it more pronounced for the with-filtering model which was trained on harder negatives?
