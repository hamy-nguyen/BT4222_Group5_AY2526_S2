# Phase 3 Contributions & Results

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
- Platform filtering (Windows/Mac/Linux) ensures negatives represent games the user could actually play.
- Two-stage rejection sampling with pre-computed cumulative weights reduces runtime from O(n_games) per sample to O(1).

**Leakage-Safe LOO Aggregate Features**

- All aggregates involving `voted_up` (user/game positive ratios, average playtimes, developer ratings, playtime-vs-user-avg) are computed with leave-one-out exclusion of the target (user, game) pair.
- Test set aggregates additionally respect temporal ordering — only training interactions before the test timestamp are used.

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

- Dataset: 842,487 samples (15% positive, 85% synthetic negatives)
- Loss: weighted BCE (positive class weight = 5.65)
- Optimizer: Adam (lr=0.001, weight_decay=0.0001)
- Scheduler: CosineAnnealingLR (T_max=50)
- Early stopping: patience=5 on validation AUC-ROC
- Best checkpoint: Epoch 39 (Val AUC-ROC=0.7980, Val AUC-PR=0.8982)

**macOS Kernel Stability Fixes** (from session 2026-04-03):

- Switched LR solver from `saga` → `lbfgs` (saga uses OpenMP internally, unsafe with macOS fork-based Jupyter)
- Added thread environment variables before all imports to prevent OpenMP fork-safety crashes:
  ```python
  os.environ["OMP_NUM_THREADS"] = "1"
  os.environ["OPENBLAS_NUM_THREADS"] = "1"
  os.environ["MKL_NUM_THREADS"] = "1"
  os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
  os.environ["NUMEXPR_NUM_THREADS"] = "1"
  ```

---

## 2. Phase 3 Results

**Test set**: 46,922 real reviews (76.2% positive, natural distribution).
**Ranking evaluation**: 1 positive + 99 negatives per user (25,793 users).

### Classification Metrics

| Model               | AUC-ROC    | AUC-PR     | F1     | Precision | Recall | Accuracy |
| ------------------- | ---------- | ---------- | ------ | --------- | ------ | -------- |
| Logistic Regression | 0.6904     | 0.8482     | 0.8697 | —         | —      | —        |
| XGBoost             | 0.7385     | **0.9022** | 0.8653 | —         | —      | —        |
| **Wide & Deep**     | **0.7577** | 0.8824     | 0.8634 | 0.8447    | 0.8829 | 0.79     |

### Ranking Metrics

| Model               | HR@10      | NDCG@10    |
| ------------------- | ---------- | ---------- |
| Logistic Regression | 0.9458     | 0.7592     |
| XGBoost             | 0.9929     | 0.9588     |
| **Wide & Deep**     | **0.9940** | **0.9667** |

### Per-Class Report (Wide & Deep, threshold=0.8730)

| Class        | Precision | Recall | F1   | Support |
| ------------ | --------- | ------ | ---- | ------- |
| Negative (0) | 0.56      | 0.48   | 0.52 | 11,144  |
| Positive (1) | 0.84      | 0.88   | 0.86 | 35,778  |
| Weighted avg | 0.78      | 0.79   | 0.78 | 46,922  |

### Takeaways

- **Wide & Deep** achieves the best AUC-ROC (0.7577) and ranking metrics (HR@10: 0.9940, NDCG@10: 0.9667), confirming the deep component's generalisation benefit over LR and tree-based baselines.
- **XGBoost** leads on AUC-PR (0.9022), suggesting stronger precision-recall balance on the positive class — likely due to its native ability to handle feature interactions without manual cross features.
- The ~0.04 drop in AUC-ROC from validation (0.7980) to test (0.7577) is consistent with the validation set being drawn from a different distribution (smaller, real reviews only vs. mixed train set).

---

### Understanding the Negative Class Gap (F1: 0.52 vs 0.86)

The model predicts positives well (F1=0.86) but struggles on negatives (F1=0.52, precision=0.56, recall=0.48). This is expected and explainable from three angles.

**1. Training distribution mismatch**

The training set is 85% synthetic negatives, so a high positive class weight (5.65×) was applied to compensate. This pushes the model to be aggressive about predicting positives — it would rather over-predict positive than miss a real one. The effect carries over to test: when a game is truly disliked, the model often still predicts positive, directly hurting negative recall (0.48).

**2. Architecture bias from the deep component**

The deep component ingests NCF MLP embeddings and BERT embeddings — both learned primarily from positive interactions (real reviews are 76% positive). These embeddings encode "similarity to things users like," which is a positive-leaning signal by construction. The wide component's cross features (`genre_match_score`, `price_gap`, `free_match`) are also framed as match signals — they fire when a game _fits_ the user's taste, not when it doesn't. So both components provide weak signal for the negative class.

**3. This gap is acceptable for a recommender**

The end goal is to surface games a user would like in the top-K. From a recommendation standpoint:

- A **false negative on a positive** (model misses a game the user would like) is a lost opportunity — the system fails the user.
- A **false positive on a negative** (model recommends a game the user wouldn't like) is a tolerable error — the user simply ignores or skips it.

The model achieves negative recall of only 0.48, meaning it lets many disliked games through. But the ranking metrics show HR@10=0.9940 — the truly liked game still appears in the top 10 99.4% of the time, even when the candidate pool contains many false positives. This means the positive signal is strong enough to push liked games above the noise, which is exactly what a recommender needs.

Optimising for negative F1 (e.g. lowering the classification threshold, increasing negative class weight) would improve rejection of disliked games but risks suppressing liked ones — trading HR@10 for negative precision, which is the wrong tradeoff for this task.
