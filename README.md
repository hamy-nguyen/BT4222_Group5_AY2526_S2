# BT4222 Group 5 — Steam Game Recommendation System

A multi-phase recommendation system built on Steam review data, combining collaborative filtering, content-based filtering, and a Wide & Deep classification model.

---

## Project Structure

```
BT4222_Group5_AY2526_S2/
├── phase0_finalised/                   # Data preparation & feature engineering
│   ├── final_phase0_data_sampling.ipynb
│   ├── final_phase0_feature_engineering.ipynb
│   ├── processed_steam_data/           # Cleaned & filtered Steam tables
│   └── feature_engineered_data/       # Model-ready feature files
│
├── phase1_finalised/                   # Neural Collaborative Filtering
│   ├── phase_1_ncf.ipynb
│   ├── phase1_outputs/                # NCF embeddings & scores
│   └── phase1_outputs_alternative/    # Alternative NCF run outputs
│
├── phase2_finalised/                   # Content-Based Filtering
│   ├── phase 2.1/
│   │   ├── phase2.1_cbf_model.ipynb
│   │   └── phase2.1_outputs/          # Content similarity scores (7 parts)
│   └── phase 2.2/
│       ├── 2_2_cbf_enhanced_model_fixed.ipynb
│       ├── content_similarity_scores.parquet
│       ├── community_sentiment.parquet
│       └── user_taste_profiles.parquet
│
├── phase3_finalised/                   # Wide & Deep classification model
│   ├── with_extra_filter/             # Variant A: with additional filtering
│   │   ├── phase3_data_sampling.ipynb
│   │   ├── phase3_feature_engineering.ipynb
│   │   ├── phase3_wide_and_deep_extra_filter.ipynb
│   │   ├── phase3_processed_steam_data/
│   │   ├── phase3_feature_engineered_data/
│   │   └── phase3_outputs_extra_filter/
│   └── without_extra_filter/          # Variant B: ablation baseline
│       ├── phase3_wide_and_deep.ipynb
│       └── phase3_outputs/
│
└── phase4_finalised/                   # Hybrid stacking
    ├── phase4_logistic_regression.ipynb
    └── phase4_outputs/                 # LR scores and evaluation
```

---

## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/hamy-nguyen/BT4222_Group5_AY2526_S2.git
   cd BT4222_Group5_AY2526_S2
   ```

2. **Download the raw dataset** from Google Drive and place it at `steam_dataset_2025_csv/` in the repo root:
   [Steam Dataset 2025](https://drive.google.com/drive/folders/1zOhuPBiuPmvz5_gdEmsBfVNqpMuboV8k?usp=sharing)

3. **Activate the virtual environment**

   ```bash
   source .venv/bin/activate
   ```

4. **Run notebooks in order** — paths are pre-configured in cell 2 or 3 of each notebook, no additional setup required.

---

## Pipeline Overview

### Phase 0 — Data Preparation & Feature Engineering

Clean and filter raw Steam data into a modelling-ready dataset. Construct a leakage-safe temporal train/test split and pre-compute all features consumed by downstream phases: LOO aggregates, multi-hot encodings, BERT embeddings, user genre vectors, and domain-informed synthetic negatives.

### Phase 1 — Neural Collaborative Filtering

Train an NCF model (GMF + MLP) on user-game interactions to learn latent user and item representations. Produces embeddings consumed by Phase 3.

### Phase 2 — Content-Based Filtering

- **Phase 2.1:** Compute pairwise content similarity scores between games based on descriptions and metadata
- **Phase 2.2:** Enhance recommendations with community sentiment and user taste profiles

### Phase 3 — Wide & Deep Classification Model

Predict `voted_up` for (user, game) pairs by combining collaborative embeddings (Phase 1), content embeddings, and behavioural aggregates in a Wide & Deep architecture. Two variants are trained as an ablation study of additional data filtering steps.

### Phase 4 — Hybrid Stacking

Combine probability scores from Phases 1, 2, and 3 using logistic regression as a stacking meta-learner to produce final recommendations.
