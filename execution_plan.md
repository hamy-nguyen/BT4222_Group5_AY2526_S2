# Steam Recommender System — Execution Plan

---

## Overview

This plan implements a 5-phase recommender system pipeline: data preparation, collaborative filtering (NCF), content-based filtering (BERT), supervised classification (Wide & Deep), and hybrid combination. All phases incorporate the professor's data leakage feedback and address user-item sparsity via informed negative sampling.

**Course techniques used**: Word Embedding, Neural Networks & Deep Learning, BERT, Ensemble Learning, NLP Feature Engineering, Model Evaluation.

---

## Phase 0: Data Preparation & Leakage-Safe Feature Engineering

> **Goal**: Build the cleaned dataset, temporal split, negative samples, and all aggregate features — computed in a leakage-safe manner.

### 0.1 Data Cleaning & Filtering

1. Filter `applications.csv` to `type = "game"` only (~150K games)
2. Filter users with >= 3 reviews (~47K users, ~329K reviews)
3. Filter games with >= 2 reviews from qualifying users
4. Canonicalise genre/category names to English (34 genres, 65 categories)
5. Clean `votes_funny` overflow values (cap at 99.9th percentile)
6. Log-transform `author_playtime_forever`, `mat_final_price`
7. Cap `mat_final_price` at 99th percentile for outlier removal
8. Drop `steam_purchase` (100% True — zero variance)

### 0.2 Temporal Train/Test Split

Split on `timestamp_created`:

- **Train**: oldest 80% of reviews per user
- **Test**: most recent 20% per user (at least 1 held-out review per user)
- Store `test_timestamp_per_user` — the timestamp of each user's earliest test interaction

> **Why per-user split?** A global 80/20 timestamp cut would leave some users entirely in train or entirely in test, breaking evaluation. Per-user ensures every user has both history and a held-out target.

### 0.3 Leakage-Safe Aggregate Features

**Critical rule**: For any aggregate involving `voted_up` or the specific (user, game) interaction being predicted, **exclude that interaction** before computing the aggregate.

#### Implementation Pattern: Leave-One-Out (LOO) Aggregates

For each interaction (u, g) in the dataset:

```
Feature                  | How to compute (leakage-safe)
-------------------------|----------------------------------------------
user_positive_ratio      | (sum of voted_up for user u EXCLUDING game g) / (count of reviews for u EXCLUDING game g)
user_avg_playtime        | mean playtime of user u EXCLUDING game g
game_positive_ratio      | (sum of voted_up for game g EXCLUDING user u) / (count of reviews for g EXCLUDING user u)
game_avg_playtime        | mean playtime of game g EXCLUDING user u
developer_avg_rating     | mean positive ratio across developer's games EXCLUDING game g
review_recency           | timestamp of THIS interaction relative to user's history — no leakage concern
playtime_vs_user_avg     | playtime(u,g) / user_avg_playtime(u, EXCLUDING g)
```

**Additional temporal constraint for test set**: When computing aggregates for a test interaction at time `t`, only use training interactions with `timestamp < t`. This means aggregate features vary per-user-per-timestamp.

#### Implementation (Efficient Pandas Pattern)

```python
# Example: leakage-safe user_positive_ratio
user_totals = train_df.groupby('author_steamid')['voted_up'].agg(['sum', 'count'])
# For each row, subtract the current interaction's contribution
df['user_positive_ratio'] = df.apply(
    lambda r: (user_totals.loc[r.author_steamid, 'sum'] - r.voted_up) /
              (user_totals.loc[r.author_steamid, 'count'] - 1),
    axis=1
)
# Vectorised version (much faster):
df = df.merge(user_totals, on='author_steamid')
df['user_positive_ratio'] = (df['sum'] - df['voted_up']) / (df['count'] - 1)
```

### 0.4 Negative Sampling Strategy (Addressing Sparsity)

The dataset only contains reviewed (user, game) pairs. To train models that distinguish "liked" from "not liked", we need negative samples.

#### Domain-Informed Negative Sampling

Instead of random negatives, construct *harder*, more realistic negatives using game-specific domain knowledge:

1. **Genre-proportional negatives**: Binary genre overlap is insufficient — top genres (Indie, Action, Adventure, Casual) cover ~92% of games, making the filter nearly useless. Instead, sample genres *proportional to the user's play history*. A user with 5 Action games and 1 Casual game gets ~5× more Action negatives than Casual. This produces harder negatives concentrated in the user's actual taste profile.
2. **Platform-compatible negatives**: Only sample games available on platforms the user has played on (inferred from their review history). A Mac-only user cannot play Windows-exclusive games — including those as negatives adds noise rather than signal.
3. **Popularity-weighted negatives**: Within each genre, sample games weighted by review count. Popular games that a user skipped are stronger negative signals than obscure titles.

#### Sampling Method: Two-Stage Rejection Sampling

For each positive training interaction, generate `NEG_RATIO` negatives via:
1. **Stage 1**: Sample a genre from the user's normalised genre distribution
2. **Stage 2**: Sample a game from that genre, weighted by popularity (O(1) via pre-computed cumulative weights + `searchsorted`)
3. **Reject** if game was already reviewed by user or platform-incompatible; retry (up to 5 attempts)

This runs in ~10–20s vs minutes for the naïve per-user `np.random.choice(replace=False)` approach, because each sample is O(1) instead of O(n_games).

#### Sampling Ratio & Label

- Sample **4 negatives per positive** (standard ratio for implicit feedback NCF)
- Label: `voted_up = 0` for synthetic negatives, `voted_up = 1` for positive reviews, `voted_up = 0` for actual negative reviews
- Store a `is_synthetic_negative` flag to allow analysis
- Deduplicate (user, game) pairs after sampling to avoid repeat negatives

#### Why This Matters

With 99.98% sparsity, the model only sees ~7 interactions per user out of ~33K games. Without negatives, the model has no signal for "user u would NOT like game g." Random negatives are easy to classify (a hardcore FPS player won't like a farming simulator — trivial). Genre-proportional negatives force the model to learn *finer-grained* preferences within genres the user actually engages with, while platform filtering ensures negatives represent genuinely reachable games.

### 0.5 Review Text Processing (Leakage-Safe)

**Professor's clean rule applied**:


| Text source                                                                  | Can use to predict rating?         | Rationale                                                              |
| ---------------------------------------------------------------------------- | ---------------------------------- | ---------------------------------------------------------------------- |
| `short_description`                                                          | YES — as item feature              | Official editorial content, independent of user experience             |
| `review_text` of (u, g) pair                                                 | NO                                 | Downstream expression of the same experience that generates `voted_up` |
| Aggregated review text of OTHER users for game g, written BEFORE timestamp t | YES — as time-varying item feature | Independent of user u's experience                                     |
| Review text of user u for OTHER games (taste profiling)                      | YES — as user feature              | Captures general preferences, not specific to target item              |


#### Legitimate uses of review text:

1. **User taste profile** (user feature): Aggregate BERT embeddings of user u's reviews for *other* games (excluding target game g). This captures what themes/genres/mechanics the user gravitates toward.
2. **Game community sentiment** (time-varying item feature): For game g at time t, aggregate sentiment/embeddings from *other users'* reviews written before t. This captures "what do people generally say about this game."

---

## Phase 1: Neural Collaborative Filtering (NCF)

> **Goal**: Learn latent user and item embeddings from interaction patterns.
> **Techniques**: Neural Networks & Deep Learning, Word Embedding (learned embeddings)

### 1.1 Architecture

Use the NCF framework (He et al., 2017) combining two pathways:

```
UserID ─→[Embedding 32-dim] ──┬─→ GMF (element-wise product) ──┐
                              │                                ├─→ Concat ─→ Dense (64) ─→ Dense(1) ─→ Sigmoid
GameID ─→[Embedding 32-dim] ──┤                                │
                              └─→ MLP (concat → 128 → 64) ─────┘
```

- **GMF pathway**: Generalised Matrix Factorisation — element-wise product of user/item embeddings, captures linear interactions
- **MLP pathway**: Multi-layer perceptron on concatenated embeddings, captures non-linear interactions
- **Output**: Predicted probability that user u would recommend game g

### 1.2 Training Details

- **Input**: (user_id, game_id, label) triples from train set + negative samples (Phase 0.4)
- **Loss**: Binary cross-entropy
- **Negatives**: 4 per positive, genre-aware sampling
- **Optimiser**: Adam, lr=1e-3, weight decay=1e-5
- **Regularisation**: Dropout (0.2) on MLP layers, L2 on embeddings
- **Epochs**: Early stopping on validation HR@10 (patience=5)
- **Batch size**: 1024

### 1.3 Output

- Trained user embeddings (32-dim) and item embeddings (32-dim) — these become **features for Phase 3**
- NCF predicted scores for all (user, candidate_game) pairs — **features for Phase 4**

### 1.4 Evaluation

- Hit Rate @ 10 (HR@10): proportion of test items appearing in top-10 predictions
- NDCG @ 10: ranking quality metric

### 1.5 Expectations & Risk

Sparsity (~99.98%) means NCF will likely underperform compared to Phase 3. The primary value of Phase 1 is producing **learned embeddings** that encode collaborative signals, to be consumed downstream as features.

---

## Phase 2: Content-Based Filtering with BERT Embeddings

> **Goal**: Compute item representations from game metadata + description text, and user taste profiles from past review text.
> **Techniques**: BERT, Word Embedding, NLP Feature Engineering

### 2.1 Game Representation (Item Feature Vector)

For each game, construct a feature vector combining:


| Component                             | Dimensions      | Source                                              |
| ------------------------------------- | --------------- | --------------------------------------------------- |
| BERT embedding of `short_description` | 768 → 128 (PCA) | `applications.csv` — **safe**: editorial content    |
| Genre multi-hot                       | 34              | `application_genres` junction table                 |
| Category multi-hot                    | 65              | `application_categories` junction table             |
| Normalised price                      | 1               | `mat_final_price` (log-transformed, min-max scaled) |
| Metacritic score                      | 1               | Impute missing with genre-median, then scale        |
| Platform support                      | 3               | windows/mac/linux binary                            |
| `is_free`                             | 1               | Binary                                              |
| Developer avg rating                  | 1               | Leakage-safe aggregate (Phase 0.3)                  |
| Publisher portfolio size              | 1               | Count from junction table                           |


**Total item vector**: ~234 dimensions

### 2.2 BERT Encoding of `short_description`

- Use `sentence-transformers/all-MiniLM-L6-v2` (384-dim, fast, English-optimised)
  - Lighter than full BERT-base (768-dim), suitable for short texts (median 202 chars)
- Encode all ~150K game descriptions once (batch inference)
- Reduce to 128 dimensions via PCA (retain ~95% variance)

### 2.3 User Taste Profile from Review Text (Leakage-Safe)

For user u predicting game g:

1. Collect all review texts written by user u for games **other than g**, with `timestamp < t` (where t is the prediction timestamp)
2. Encode each review with the same sentence-transformer
3. Average the embeddings, weighted by recency (exponential decay)
4. This 128-dim vector represents "what does user u generally talk about in reviews" — their taste signal

**This is safe** because it's a *user feature* derived from *other* games, not the target item.

### 2.4 Game Community Sentiment (Leakage-Safe, Time-Varying)

For game g at time t:

1. Collect review texts from **other users** for game g, with `timestamp < t`
2. Compute average sentiment score (simple positive/negative ratio) and average BERT embedding
3. This captures "community reception of this game up to this point"

**This is safe** because it excludes user u's own review and only uses reviews before the prediction time.

### 2.5 Content-Based Similarity Scoring

- Compute cosine similarity between user taste profile vector and candidate game vectors
- Produce top-K ranked candidates per user
- Content similarity scores become **features for Phase 4**

### 2.6 Evaluation

- Same as Phase 1: HR@10, NDCG@10
- Additionally: coverage (% of games that appear in at least one user's top-10) — content-based should have better coverage than CF

---

## Phase 3: Supervised Classification with Wide & Deep Network

> **Goal**: Predict `voted_up` for (user, game) pairs using all available features. This is the **primary model** — expected to produce the best results.
> **Techniques**: Neural Networks & Deep Learning, Ensemble Learning (wide + deep), Word Embedding

### 3.1 Why Wide & Deep?

The dataset has both:

- **Sparse categorical features** (genre × user interactions, developer identity) — benefit from **memorisation** (Wide component)
- **Dense continuous features** (embeddings, playtime, price) — benefit from **generalisation** (Deep component)

Wide & Deep (Cheng et al., 2016) is designed exactly for this combination and is the standard production architecture at Google for recommender systems.

### 3.2 Architecture

```
WIDE component (memorisation):
  Cross-product features:
    - user_top_genre × game_genre (does this game match user's favourite genre?)
    - user_price_tier × game_price_tier (price bracket match)
    - is_free × user_free_game_ratio
  → Linear layer → concat with deep output

DEEP component (generalisation):
  Dense features:
    - NCF user embedding (32-dim, from Phase 1)
    - NCF item embedding (32-dim, from Phase 1)
    - BERT description embedding (128-dim, from Phase 2)
    - User taste profile embedding (128-dim, from Phase 2)
    - user_positive_ratio (LOO, leakage-safe)
    - game_positive_ratio (LOO, leakage-safe)
    - user_avg_playtime (LOO)
    - game_avg_playtime (LOO)
    - developer_avg_rating (LOO)
    - playtime_vs_user_avg (LOO)
    - normalised price, metacritic, review_count, etc.
  → Dense(256) → ReLU → Dropout(0.3)
  → Dense(128) → ReLU → Dropout(0.3)
  → Dense(64) → ReLU → Dropout(0.2)
  → concat with wide output

COMBINED:
  → Dense(32) → ReLU → Dense(1) → Sigmoid
```

### 3.3 Feature Summary (All Leakage-Safe)


| Feature                    | Dims      | Source                  | Leakage status                         |
| -------------------------- | --------- | ----------------------- | -------------------------------------- |
| NCF user embedding         | 32        | Phase 1                 | Safe (learned from other interactions) |
| NCF item embedding         | 32        | Phase 1                 | Safe (learned from other interactions) |
| BERT description embedding | 128       | Phase 2                 | Safe (editorial content)               |
| User taste profile (BERT)  | 128       | Phase 2                 | Safe (other games, past timestamps)    |
| Game community sentiment   | 128       | Phase 2                 | Safe (other users, past timestamps)    |
| Genre multi-hot            | 34        | Junction table          | Safe (static metadata)                 |
| Category multi-hot         | 65        | Junction table          | Safe (static metadata)                 |
| user_positive_ratio        | 1         | LOO aggregate           | Safe (excludes target pair)            |
| game_positive_ratio        | 1         | LOO aggregate           | Safe (excludes target pair)            |
| user_avg_playtime          | 1         | LOO aggregate           | Safe (excludes target pair)            |
| game_avg_playtime          | 1         | LOO aggregate           | Safe (excludes target pair)            |
| developer_avg_rating       | 1         | LOO aggregate           | Safe (excludes target game)            |
| playtime_vs_user_avg       | 1         | LOO aggregate           | Safe (excludes target pair)            |
| Normalised price           | 1         | Static metadata         | Safe                                   |
| Metacritic score           | 1         | Static metadata         | Safe                                   |
| is_free                    | 1         | Static metadata         | Safe                                   |
| Platform support           | 3         | Static metadata         | Safe                                   |
| user_review_count          | 1         | Count (minus 1 for LOO) | Safe                                   |
| game_review_count          | 1         | Count (minus 1 for LOO) | Safe                                   |
| Wide cross features        | ~10       | Derived                 | Safe                                   |
| **Total**                  | **~570+** |                         |                                        |


### 3.4 Training Details

- **Input**: (user, game) pairs — all positive reviews + informed negative samples (Phase 0.4)
- **Label**: `voted_up` (1 for positive, 0 for negative/synthetic negative)
- **Loss**: Binary cross-entropy with class weights (76% positive → weight negative class ~3.2x) OR focal loss
- **Optimiser**: Adam, lr=1e-3 with cosine annealing
- **Regularisation**: Dropout (0.2-0.3), L2 weight decay (1e-5), batch normalisation
- **Batch size**: 2048
- **Epochs**: Early stopping on validation AUC-ROC (patience=5)

### 3.5 Evaluation (Classification Metrics)


| Metric                | Why                                                                         |
| --------------------- | --------------------------------------------------------------------------- |
| AUC-ROC               | Overall ranking quality, threshold-independent                              |
| AUC-PR                | Better for imbalanced classes (76/24 split)                                 |
| F1-score              | Balance of precision and recall at chosen threshold                         |
| Precision@K, Recall@K | Recommendation-relevant: of top-K predictions, how many are truly positive? |
| Calibration plot      | Are predicted probabilities well-calibrated?                                |


### 3.6 Baselines for Comparison

To demonstrate the value of the deep model, compare against:

1. **Logistic Regression** (wide-only, no deep component) — baseline
2. **Random Forest / XGBoost** (ensemble learning) — strong tabular baseline
3. **Wide & Deep** — the proposed model

This progression demonstrates: basic ML → ensemble → deep learning, using course techniques.

---

## Phase 4: Hybrid Combination

> **Goal**: Combine signals from all phases into a final ranking.
> **Techniques**: Ensemble Learning

### 4.1 Approach

A lightweight meta-learner (logistic regression or small MLP) that takes as input:


| Signal                                           | Source  |
| ------------------------------------------------ | ------- |
| NCF predicted score                              | Phase 1 |
| Content-based similarity score                   | Phase 2 |
| Wide & Deep predicted probability                | Phase 3 |
| User cold-start flag (1 if user has < 5 reviews) | Data    |
| Game cold-start flag (1 if game has < 3 reviews) | Data    |


The meta-learner learns to weight Phase 1 vs Phase 2 vs Phase 3 depending on data availability. For cold-start users, it should lean toward content-based; for data-rich users, toward collaborative + supervised signals.

### 4.2 Training

- Train on validation set predictions (NOT train set — avoid overfitting to train-set scores)
- Simple logistic regression is sufficient given only 5 input features

### 4.3 Final Evaluation

- HR@10, NDCG@10 (ranking metrics)
- AUC-ROC, F1 (classification metrics)
- Compare all 4 phases side-by-side + baselines

---

## Implementation Timeline & Task Breakdown

### Task 1: Data Pipeline (Phase 0) — Foundation

```
[ ] 0.1  Clean & filter dataset (games only, user >= 3 reviews, game >= 2 reviews)
[ ] 0.2  Canonicalise genres/categories to English
[ ] 0.3  Temporal train/test split (per-user 80/20)
[ ] 0.4  Implement LOO aggregate feature computation (vectorised pandas)
[ ] 0.5  Implement informed negative sampling (genre-aware, popularity-weighted)
[ ] 0.6  BERT-encode all short_descriptions (batch, save to disk)
[ ] 0.7  Compute user taste profiles from review text (LOO, temporal)
[ ] 0.8  Save processed feature matrices to parquet files
```

### Task 2: NCF Model (Phase 1)

```
[ ] 1.1  Build NCF architecture (GMF + MLP) in PyTorch
[ ] 1.2  Implement data loader with negative sampling
[ ] 1.3  Train NCF, tune embedding dimension & MLP depth
[ ] 1.4  Evaluate HR@10, NDCG@10
[ ] 1.5  Extract & save learned user/item embeddings
```

### Task 3: Content-Based Model (Phase 2)

```
[ ] 2.1  Build game feature vectors (BERT + metadata)
[ ] 2.2  Compute PCA-reduced BERT embeddings
[ ] 2.3  Build user taste profiles (leakage-safe BERT aggregation)
[ ] 2.4  Compute content-based similarity scores
[ ] 2.5  Evaluate HR@10, NDCG@10, coverage
```

### Task 4: Wide & Deep Model (Phase 3)

```
[ ] 3.1  Assemble full feature matrix (all LOO features + embeddings)
[ ] 3.2  Implement Wide & Deep in PyTorch
[ ] 3.3  Train baselines: Logistic Regression, XGBoost
[ ] 3.4  Train Wide & Deep with class weighting
[ ] 3.5  Evaluate AUC-ROC, AUC-PR, F1, Precision@K
[ ] 3.6  Feature importance analysis (SHAP or permutation importance)
```

### Task 5: Hybrid & Final Evaluation (Phase 4)

```
[ ] 4.1  Collect Phase 1-3 predicted scores on validation set
[ ] 4.2  Train meta-learner
[ ] 4.3  Final evaluation on test set (all metrics, all phases compared)
[ ] 4.4  Ablation study: effect of LOO features vs naive features
```

---

## Key Design Decisions Summary


| Decision                  | Choice                                                      | Rationale                                                       |
| ------------------------- | ----------------------------------------------------------- | --------------------------------------------------------------- |
| Aggregate features        | Leave-one-out per (u, g) pair                               | Professor's feedback: prevent label leakage                     |
| Review text usage         | User taste profiles + community sentiment (not per-pair)    | Professor's clean rule: review text is downstream of rating     |
| `short_description`       | Used as item feature                                        | Safe: editorial content, independent of user experience         |
| Negative sampling         | Genre-aware + popularity-weighted (4:1 ratio)               | Addresses 99.98% sparsity with informative negatives            |
| Primary model             | Wide & Deep (Phase 3)                                       | Best fit for mixed sparse/dense feature space                   |
| Deep learning requirement | NCF (Phase 1) + Wide & Deep (Phase 3)                       | Two distinct neural architectures satisfying course requirement |
| BERT usage                | Sentence-transformer for descriptions + user taste profiles | Course technique; lightweight model for short texts             |
| Baseline comparison       | LR → XGBoost → Wide & Deep                                  | Shows progression through course techniques                     |
| Temporal split            | Per-user 80/20 on timestamp                                 | Prevents future-leaking and ensures all users have test data    |


