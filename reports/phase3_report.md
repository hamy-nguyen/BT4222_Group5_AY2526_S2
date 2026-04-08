# Phase 3: Wide & Deep — Contributions, Walkthrough & Results

---

## 1. My Contributions

### Phase 0: Data Pipeline Foundation (`final_phase0_data_sampling.ipynb`, `final_phase0_feature_engineering.ipynb`)

I built the data foundation shared by all phases.

**Step 1b — Iterative Filtering**
- Added an extra iterative filtering pass to address a caveat in the original sampling: after filtering games to those with ≥2 reviews from qualifying users, some users could drop below the 3-review threshold. The iterative pass re-filters users and games in a loop until both constraints are simultaneously satisfied, preventing silent data quality degradation.

**Step 3 Pass 2 — English Genre/Category Canonicalisation**
- Added a second pass over the genre/category tables to catch non-English entries missed in pass 1. This ensures the final multi-hot encoding covers all 34 genres and 65 categories without silent gaps.

**Domain-Informed Negative Sampling**
- Replaced random negative sampling with a genre-proportional, popularity-weighted, platform-compatible strategy.
- Sampling is proportional to each user's actual genre history (harder negatives in the user's taste space) rather than uniform random (which produces trivially easy negatives).
- A model trained on random negatives learns "action player dislikes farming simulator" — trivial. Ours learns "action player dislikes *this specific action game*" — harder and more useful.
- Platform filtering (Windows/Mac/Linux) ensures negatives represent games the user could actually play.
- Two-stage rejection sampling with pre-computed cumulative weights reduces runtime from O(n_games) per sample to O(1).

**Leakage-Safe LOO Aggregate Features**
- All aggregates involving `voted_up` (user/game positive ratios, average playtimes, developer ratings, playtime-vs-user-avg) are computed with leave-one-out exclusion of the target (user, game) pair.
- Test set aggregates additionally respect temporal ordering — only training interactions before the test timestamp are used.
- Unlike naive implementations that compute aggregates on the full dataset (leaking the target label into the feature), our stricter LOO + temporal computation means the model's good performance cannot be attributed to leakage — the signal is real.

---

### Phase 1: NCF Embedding Fix (`phase_1_ncf.ipynb`)

**Root cause**: The NCF model has two embedding pathways — GMF and MLP. The MLP pathway dominated during training, causing the GMF pathway to collapse (values decayed from 0.01 init to ~1e-38, effectively zero). Phase 1 was exporting the GMF embeddings (dead) instead of the MLP embeddings (trained), so Phase 3 received 64 features of zeros.

**Fix**: Updated the embedding export to use the MLP pathway:

```python
# Before (dead GMF embeddings — ~[-7e-38, 7e-38])
gmf_user_emb_np = model.gmf_user_emb.weight.detach().cpu().numpy()
gmf_item_emb_np = model.gmf_item_emb.weight.detach().cpu().numpy()

# After (trained MLP embeddings)
mlp_user_emb_np = model.mlp_user_emb.weight.detach().cpu().numpy()
mlp_item_emb_np = model.mlp_item_emb.weight.detach().cpu().numpy()
```

This fix was identified by inspecting actual embedding values, not just training loss. It demonstrates that understanding what the model actually learned — not just what it was supposed to learn — is critical when chaining models across phases.

---

### Phase 3: Wide & Deep Classification Model (`phase_3_wide_and_deep.ipynb`)

**Architecture** (127,150 trainable parameters):

```
WIDE (memorisation):
  12 features → Linear(1) → scalar

DEEP (generalisation):
  323 features → Dense(256) → BN → ReLU → Dropout(0.3)
               → Dense(128) → BN → ReLU → Dropout(0.3)
               → Dense(64)  → BN → ReLU

COMBINED:
  [wide_out || deep_out] (65-dim) → Dense(32) → ReLU → Dropout(0.2) → Dense(1) → Sigmoid
```

**Wide features (12)**:
- 9 LOO aggregates: `user_positive_ratio`, `user_avg_playtime`, `user_review_count_loo`, `user_price_preference`, `game_positive_ratio`, `game_avg_playtime`, `game_review_count_loo`, `playtime_vs_user_avg`, `developer_avg_rating`
- 3 cross features: `genre_match_score` (dot product of user genre history × game genre vector), `price_gap`, `free_match`

**Deep features (323)**:
- NCF MLP user embeddings (32-dim, Phase 1)
- NCF MLP item embeddings (32-dim, Phase 1)
- BERT `short_description` embeddings (128-dim)
- User genre preference vector (28-dim)
- Game genre multi-hot (28-dim)
- Game category multi-hot (58-dim)
- Platform multi-hot (3-dim)
- Game scalar features: price, release date, etc. (5-dim)
- LOO aggregates (9-dim, shared with wide)

**Training**:
- Dataset: 842,487 samples (15% positive, 85% negative)
  - Built by concatenating `train_with_features.parquet` (real training reviews) + `negatives_with_features.parquet` (synthetic negatives). Test data is never used in training.
  - The 1:4 ratio governs synthetic negatives per positive real review only. But `train_feat` also contains real negative reviews (voted_up=False) — Steam's natural distribution is ~76% positive, so ~24% of real reviews are already negative.
  - Total negatives = real negatives + synthetic negatives, which together push the ratio to 15/85 rather than the 20/80 that a pure 1:4 strategy would imply.
- Loss: weighted BCE (positive class weight = 5.65)
- Optimizer: Adam (lr=0.001, weight_decay=0.0001)
- Scheduler: CosineAnnealingLR (T_max=50)
- Early stopping: patience=5 on validation AUC-ROC
- Best checkpoint: Epoch 39 (Val AUC-ROC=0.7980, Val AUC-PR=0.8982)

---

## 2. Phase 3 Pipeline Walkthrough

### Loading data
- Phase 3 consumes outputs from Phase 0 and Phase 1 — it computes no features from scratch.
- Loads 8 parquet files: train/test reviews with LOO aggregates, synthetic negatives, game static features (genres, categories, platform, price, metacritic), BERT description embeddings (128-dim), user genre vectors (28-dim), NCF user and item embeddings (32-dim each).

### Assembling the feature matrix
- `assemble_features()` left-joins all feature sources onto each (user, game) row:
  - Game static features, BERT embeddings, NCF item embeddings → joined on `appid`
  - User genre vectors, NCF user embeddings → joined on `author_steamid`
- Missing embeddings filled with zeros.
- Result: one flat row per (user, game) interaction with all features merged. A single row looks like:

```
user=76561198012345  game=730  voted_up=1
  user_positive_ratio = 0.82      # this user likes 82% of games they review
  game_positive_ratio = 0.91      # this game is liked by 91% of reviewers
  playtime_vs_user_avg = 2.3      # played this game 2.3x their usual playtime
  genre_match_score = 3.4         # strong genre overlap with user history
  price_gap = 0.1                 # game price close to user's usual spend
  ncf_user_emb_0..31 = [0.12, -0.34, 0.07, ...]   # who this user is
  ncf_item_emb_0..31 = [0.45, 0.21, -0.18, ...]   # what this game is
  desc_emb_0..127    = [0.03, -0.11, ...]          # game description semantics
  user_genre_0..27   = [0.4, 0.0, 0.3, ...]        # user's genre preferences
  genre_Action=1, genre_Indie=0, ...               # game's genres (multi-hot)
  is_free=0, price=2.3, metacritic=0.83            # game scalar features
```

Total: ~326 columns per row. Wide pathway sees 12 columns (LOO aggregates + cross features); deep pathway sees all 323. The 9 LOO aggregates are shared between both.

### Cross features for the wide component
- 3 hand-crafted interaction features encoding domain hypotheses about user-game fit:
  - `genre_match_score` — dot product of user genre history × game genre vector
  - `price_gap` — |user price preference − game price|
  - `free_match` — is_free × user price sensitivity

### Routing features into wide vs deep
- **Wide (12 features)**: 9 LOO aggregates + 3 cross features → single linear layer → memorises direct associations (e.g. "user who likes Action games + Action game → positive")
- **Deep (323 features)**: NCF embeddings + BERT embeddings + genre/category multi-hots + game scalars + LOO aggregates → 3-layer MLP → generalises over complex patterns in embedding space

### Building the training set
- Combine real training reviews + Phase 0 synthetic negatives → 842,487 samples (15% positive, 85% negative).
- Validation: 10% of real training reviews only (natural distribution) — used for early stopping and threshold tuning.
- Ranking evaluation set: for each test positive, sample 99 random negatives → 1-vs-99 ranking problem per user.

### Evaluation

**Classification** — can the model correctly label each (user, game) pair as liked or not?
- Model outputs raw probabilities → **AUC-ROC** and **AUC-PR** computed directly from these, no threshold needed.
- Sweep all thresholds on validation set, pick the one maximising F1 → optimal threshold = 0.8730.
- Apply 0.8730 to test set → binary predictions → F1 (0.8634), precision (0.8447), recall (0.8829).

**Ranking** — can the model surface liked games above disliked ones in a candidate list?
- Simulates the actual recommendation scenario: the model sees a pool of 100 candidates (1 liked + 99 random unseen games) and must rank them.
- **HR@10** = 0.9940 — 99.4% of users had their liked game in the top 10.
- **NDCG@10** = 0.9667 — the liked game tends to appear near rank 1, not buried at rank 9 (penalises lower ranks via log₂(rank+1) discount).
- Why 99 negatives? From He et al. (2017) NCF protocol — with 100 candidates, a random model hits HR@10 only 10% of the time, so anything well above 10% demonstrates real signal.

**Key insight**: classification tells us how clean the predictions are; ranking tells us whether the system actually works as a recommender. A model can be a moderate classifier (AUC-ROC 0.76) but an excellent ranker (HR@10 0.994) — the positive signal is strong enough to consistently push liked games to the top even if the raw probability calibration isn't perfect.

### Baselines
- LR and XGBoost trained on identical data, evaluated with identical protocols → results directly comparable.
- Progression: LR (linear, no interactions) → XGBoost (non-linear, tabular) → Wide & Deep (non-linear + embeddings).

### Outputs for Phase 4
- Best model scores every (user, game) row in train/test/negatives → saved as parquet.
- Phase 4 meta-learner stacks these scores alongside NCF and content-based signals for final ranking.

---

## 3. Phase 3 Results

### 3a. With Extra Filtering Steps (Final Approach)

Extra filtering steps: iterative filtering (users ≥3 reviews AND games ≥2 reviews enforced simultaneously) + two-pass genre/category canonicalisation.

**Test set**: 46,922 real reviews (76.2% positive), 25,793 users. Training: 842,487 samples. Best val AUC-ROC: 0.7980 (epoch 39).

#### Classification Metrics

| Model               | AUC-ROC    | AUC-PR     | F1     | Precision | Recall | Accuracy |
| ------------------- | ---------- | ---------- | ------ | --------- | ------ | -------- |
| Logistic Regression | 0.6904     | 0.8482     | 0.8697 | —         | —      | —        |
| XGBoost             | 0.7385     | **0.9022** | 0.8653 | —         | —      | —        |
| **Wide & Deep**     | **0.7577** | 0.8824     | 0.8634 | 0.8447    | 0.8829 | 0.79     |

#### Ranking Metrics

| Model               | HR@10      | NDCG@10    |
| ------------------- | ---------- | ---------- |
| Logistic Regression | 0.9458     | 0.7592     |
| XGBoost             | 0.9929     | 0.9588     |
| **Wide & Deep**     | **0.9940** | **0.9667** |

#### Per-Class Report (Wide & Deep, threshold=0.8730)

| Class        | Precision | Recall | F1   | Support |
| ------------ | --------- | ------ | ---- | ------- |
| Negative (0) | 0.56      | 0.48   | 0.52 | 11,144  |
| Positive (1) | 0.84      | 0.88   | 0.86 | 35,778  |
| Weighted avg | 0.78      | 0.79   | 0.78 | 46,922  |

---

### 3b. Without Extra Filtering Steps (Ablation)

No iterative filtering, no second canonicalisation pass — single-pass threshold filtering only.

**Test set**: 51,063 real reviews (76.1% positive), 28,308 users. Training: 864,369 samples. Best val AUC-ROC: 0.6514 (epoch 1 — model barely trained).

#### Classification Metrics

| Model               | AUC-ROC    | AUC-PR     | F1     | Precision | Recall | Accuracy |
| ------------------- | ---------- | ---------- | ------ | --------- | ------ | -------- |
| Logistic Regression | 0.6237     | 0.8249     | 0.8646 | —         | —      | —        |
| XGBoost             | **0.7234** | **0.8962** | 0.8644 | —         | —      | —        |
| Wide & Deep         | 0.6604     | 0.8457     | 0.8601 | 0.7710    | 0.9725 | 0.76     |

#### Ranking Metrics

| Model               | HR@10      | NDCG@10    |
| ------------------- | ---------- | ---------- |
| Logistic Regression | 0.8290     | 0.4744     |
| XGBoost             | **0.9947** | **0.9553** |
| Wide & Deep         | 0.8085     | 0.4489     |

---

### 3c. Comparison: With vs Without Extra Filtering

| Model | AUC-ROC (with) | AUC-ROC (without) | HR@10 (with) | HR@10 (without) | NDCG@10 (with) | NDCG@10 (without) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.6904 | 0.6237 | 0.9458 | 0.8290 | 0.7592 | 0.4744 |
| XGBoost | 0.7385 | 0.7234 | 0.9929 | 0.9947 | 0.9588 | 0.9553 |
| **Wide & Deep** | **0.7577** | 0.6604 | **0.9940** | 0.8085 | **0.9667** | 0.4489 |

#### Key observations

- **Wide & Deep degrades the most** without extra filtering: AUC-ROC drops from 0.7577 → 0.6604 and NDCG@10 collapses from 0.9667 → 0.4489. The best validation AUC-ROC hit 0.6514 at epoch 1, indicating the model failed to learn meaningful patterns and early-stopped almost immediately.
- **XGBoost is the most robust** to the absence of filtering — ranking metrics are nearly identical (NDCG@10: 0.9588 → 0.9553), likely because tree-based models are inherently more tolerant of noisy, inconsistent feature encodings.
- **LR suffers moderately** — NDCG@10 drops from 0.7592 → 0.4744, showing the linear model is sensitive to data quality but not as severely as the deep model.

#### Justification for extra filtering steps

**Iterative filtering** ensures every user has ≥3 reviews AND every game has ≥2 reviews simultaneously. Without it, borderline users and games remain — their LOO aggregates are computed from very few interactions (e.g. a user with 3 reviews where 1 is the target has only 2 interactions left for aggregation). These near-zero-sample aggregates are noisy and destabilise the Wide & Deep model's dense layers, which are more sensitive to feature noise than tree-based methods. The collapse to val AUC-ROC 0.6514 at epoch 1 without iterative filtering supports this — the model converged immediately to a degenerate solution.

**Two-pass genre/category canonicalisation** ensures all genre/category multi-hot columns represent clean, consistent English taxonomy labels. Without pass 2, non-English ASCII names (e.g. "Strategie", "Azione") remain as spurious columns — a game might have both "Strategy" and "Strategie" set to 1, or a user's genre preference vector may partially encode a non-English duplicate. This introduces inconsistency in the 28+58 multi-hot dimensions that the deep component relies on, degrading the genre-related signal for both the wide cross features (`genre_match_score`) and the deep pathway.

---

## 4. Insights

### Novelty and value added

**Embedding fusion across three signal types**
Wide & Deep is not a new architecture, but using it as a fusion layer combining collaborative (NCF), semantic (BERT), and behavioural (LOO) signals in one unified model is non-trivial. Each signal captures something the others cannot:
- NCF captures who-likes-what patterns from interaction history
- BERT captures what-the-game-is-about from content
- LOO aggregates capture how-this-user-behaves relative to their history

No single signal is sufficient; the model learns to weight them jointly.

**Domain-informed negatives force harder learning**
Genre-proportional, platform-compatible negatives concentrate the training signal within the user's taste space. Random negatives produce trivially easy discrimination (action player vs farming simulator). Our negatives force the model to learn fine-grained preferences within genres — which is exactly what a recommender needs in practice.

**Leakage-safe design validates real signal**
LOO exclusion per (user, game) pair with temporal constraints on test is stricter than standard practice. The model still performs well, confirming the features carry genuine predictive signal rather than leaked label information.

**NCF bug fix demonstrates cross-phase awareness**
Identifying the GMF embedding collapse (64 zero features silently fed into Phase 3) required inspecting actual learned values, not just training metrics. This kind of cross-phase debugging is non-obvious and directly affected downstream model quality.

---

### Results interpretation

**Ranking >> Classification for this task**
AUC-ROC of 0.76 is moderate, but HR@10 of 0.994 is exceptional. The gap between these two metrics is itself informative — it tells you the model is a strong ranker even if its probability calibration isn't perfect. For a recommender, this is the right tradeoff: what matters is whether the liked game surfaces at the top of the list, not whether every individual prediction is correct.

**LR → XGBoost → Wide & Deep progression is meaningful**
NDCG@10 jumps from 0.76 (LR) to 0.96 (XGBoost) to 0.97 (W&D). LR's inability to model non-linear interactions between embeddings explains the large gap to XGBoost. W&D's marginal gain over XGBoost shows the deep component adds genuine value, but the tabular features (LOO + cross features) are already very powerful — embeddings are complementary, not dominant.

**XGBoost wins AUC-PR**
XGBoost leads on AUC-PR (0.9022 vs 0.8824 for W&D). This suggests XGBoost has a better precision-recall tradeoff on the positive class, likely because it natively handles feature interactions through tree splits without needing manually chosen cross features. The three cross features in W&D's wide component are domain-driven but limited — XGBoost discovers interactions automatically.

**Validation → test AUC-ROC drop is expected**
Val AUC-ROC = 0.7980, test AUC-ROC = 0.7577 (~0.04 drop). The validation set is a small subset of real reviews with natural distribution; the test set is larger and drawn from a slightly different temporal window. The gap is consistent with normal generalisation, not overfitting.

---

### Understanding the Negative Class Gap (F1: 0.52 vs 0.86)

The model predicts positives well (F1=0.86) but struggles on negatives (F1=0.52, precision=0.56, recall=0.48). This is expected and explainable from three angles.

**1. Training distribution mismatch**
The training set is 85% synthetic negatives, so a high positive class weight (5.65×) was applied to compensate. This pushes the model to be aggressive about predicting positives — it would rather over-predict positive than miss a real one. The effect carries over to test: when a game is truly disliked, the model often still predicts positive, directly hurting negative recall (0.48).

**2. Architecture bias from the deep component**
The deep component ingests NCF MLP embeddings and BERT embeddings — both learned primarily from positive interactions (real reviews are 76% positive). These embeddings encode "similarity to things users like," which is a positive-leaning signal by construction. The wide component's cross features (`genre_match_score`, `price_gap`, `free_match`) are also framed as match signals — they fire when a game fits the user's taste, not when it doesn't. So both components provide weak signal for the negative class.

**3. This gap is acceptable for a recommender**
- A **false negative on a positive** (model misses a game the user would like) is a lost opportunity — the system fails the user.
- A **false positive on a negative** (model recommends a game the user wouldn't like) is a tolerable error — the user simply ignores or skips it.

The model achieves negative recall of only 0.48, meaning it lets many disliked games through. But HR@10=0.9940 shows the truly liked game still appears in the top 10 99.4% of the time, even when the candidate pool contains many false positives. The positive signal is strong enough to push liked games above the noise — which is exactly what a recommender needs.

Optimising for negative F1 (e.g. lowering the threshold, increasing negative class weight) would improve rejection of disliked games but risks suppressing liked ones — trading HR@10 for negative precision, which is the wrong tradeoff for this task.
