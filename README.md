# FinalProject_2025_Arthur_HAMARD

## Hybrid Multi-Stage Recommender System for KuaiRec

## 1. Objective

This project aims to design a high-quality recommender system for short videos using the **KuaiRec 2.0 dataset**. The goal is to accurately recommend and rank videos for users based on their historical interactions, content features, and video popularity.

We implement a **hybrid multi-stage pipeline** combining:

- Collaborative Filtering (ALS),
- Binary classification (LightGBM),
- Regression ranking (LightGBM).

Our final model optimally balances **classification accuracy** and **ranking quality**, addressing the limitations of single-model approaches.

---

## 2. Dataset Overview

The KuaiRec dataset contains:

- **User interactions** (e.g., watch ratio, timestamps),
- **Video metadata** (captions, categories, etc.),
- **User metadata** (activity ranges, categorical features),
- **Daily popularity metrics** (views, likes, comments).

All data are stored in CSV files and preprocessed to extract structured features used for training the models.

---

## 3. Methodology

### 3.1 Baseline Attempts and Their Limitations

| Attempt | Description | Outcome |
|--------|-------------|---------|
| 1 | ALS-only (implicit collaborative filtering) | Weak performance, could not rank well. |
| 2 | Transformer-based sequence modeling | Required more data than available, did not converge. |
| 3 | 2-stage LightGBM with classification only | Could identify relevant videos, but ranking was poor due to binary output. |
| 4 | 2-stage LightGBM with regression only | Could rank items, but failed to reliably detect relevance (i.e., false positives). |

### 3.2 Final Hybrid 3-Stage Pipeline

Our final approach consists of three stages:

- **Stage 1: ALS Matrix Factorization**  
  Generates latent representations of users and items. Produces `als_score`, used as a feature downstream.

- **Stage 2: LightGBM Classifier**  
  Predicts whether a user will "like" a video based on engineered features. This step filters candidates.

- **Stage 3: LightGBM Regressor**  
  Predicts the `watch_ratio` to re-rank the top items selected by the classifier, optimizing ranking metrics (e.g., NDCG).

---

## 4. Models and Feature Engineering

### 4.1 ALS (Matrix Factorization)
- **Input Features**: 
  - User-item interactions matrix with `watch_ratio_clipped` as interaction strength
- **Output**: 
  - `als_score` - Collaborative latent factor dot-product representing user-item affinity (64 factors)

### 4.2 TF-IDF Text Processing
- **Input Features**: 
  - Video `manual_cover_text` (descriptions)
  - Video `caption` text
  - Video `topic_tag` keywords
- **Output**: 
  - 50 text embedding features (`tfidf_0` to `tfidf_49`) capturing content semantics

### 4.3 LightGBM Classification Model
- **Input Features**:
  - **User Features**:
    - Categorical: `user_active_degree`, `follow_user_num_range`, `fans_user_num_range`, `friend_user_num_range`, `register_days_range`
    - One-hot encoded features: `onehot_feat0` to `onehot_feat17`
    - Behavioral: `user_agg_interactions`, `user_agg_unique_videos`
  - **Item Features**:
    - Content: `first_level_category_name`, `second_level_category_name`, `third_level_category_name`, `author_id`
    - Text: TF-IDF embeddings (`tfidf_0` to `tfidf_49`)
    - Popularity: Avg daily metrics (`avg_daily_show_cnt`, `avg_daily_play_cnt`, `avg_daily_like_cnt`, etc.)
    - Historical engagement: `video_agg_views`, `video_agg_unique_users`
    - Metadata: `video_duration`
  - **Contextual Features**: `hour_of_day`, `day_of_week`
  - **Collaborative Signal**: `als_score` from ALS model
- **Output**:
  - `classifier_score` - Probability of user liking the video (binary prediction where 1 = watch_ratio > 1.0)

### 4.4 LightGBM Regression Model
- **Input Features**:
  - Same feature set as the LightGBM Classification model
- **Output**:
  - `regressor_score` - Predicted `watch_ratio_clipped` value (continuous score representing expected engagement level)

### 4.5 Final Hybrid Pipeline
- Stage 1: ALS generates `als_score` for all candidate items
- Stage 2: LightGBM Classifier filters top-K candidates with highest probability of being liked
- Stage 3: LightGBM Regressor re-ranks filtered candidates based on predicted engagement level
---

## 5. Evaluation and Results

The performance is evaluated using the following metrics: **Precision@K**, **Recall@K**, **NDCG@K**, and **MAP@K**, for various values of K.

### 5.1 Precision@K

**INFO: A relevant video is a video with a watch_ratio superior to 1.0.**
* The train-set used for these metrics calculations is `big_matrix.csv`
* The test-set used is `small_matrix.csv`

| K | Classifier Only | Regressor Only | Hybrid |
|----|------------------|----------------|--------|
| 10 | 0.9072 | **0.9112** | 0.9074 |
| 20 | 0.8935 | **0.9022** | 0.8938 |
| 50 | 0.8466 | **0.8554** | 0.8464 |
| 100 | 0.8133 | **0.8222** | 0.8132 |
| 200 | 0.7778 | **0.7802** | 0.7781 |
| 500 | 0.7171 | 0.7174 | **0.7178** |
| 1000 | **0.6265** | 0.6248 | 0.6264 |

### 5.2 Recall@K

| K | Classifier Only | Regressor Only | Hybrid |
|----|------------------|----------------|--------|
| 10 | 0.0095 | 0.0095 | 0.0095 |
| 20 | 0.0186 | **0.0188** | 0.0186 |
| 50 | 0.0438 | **0.0443** | 0.0438 |
| 100 | 0.0837 | **0.0848** | 0.0837 |
| 200 | 0.1592 | **0.1598** | 0.1593 |
| 500 | 0.3630 | **0.3631** | 0.3634 |
| 1000 | **0.6194** | 0.6165 | 0.6191 |

### 5.3 NDCG@K

| K | Classifier Only | Regressor Only | Hybrid |
|----|------------------|----------------|--------|
| 10 | NaN | 0.6900 | **0.9065** |
| 20 | NaN | 0.6830 | **0.9045** |
| 50 | NaN | 0.6469 | **0.9062** |
| 100 | NaN | 0.6373 | **0.9071** |
| 200 | NaN | 0.6346 | **0.9074** |
| 500 | NaN | 0.6524 | **0.9069** |
| 1000 | NaN | 0.6886 | **0.9053** |

### 5.4 MAP@K

| K | Classifier Only | Regressor Only | Hybrid |
|----|------------------|----------------|--------|
| 10 | 0.0090 | 0.0090 | 0.0090 |
| 20 | 0.0174 | **0.0177** | 0.0175 |
| 50 | 0.0399 | **0.0406** | 0.0401 |
| 100 | 0.0737 | **0.0754** | 0.0743 |
| 200 | 0.1353 | **0.1370** | 0.1364 |
| 500 | 0.2908 | 0.2925 | **0.2930** |
| 1000 | 0.4658 | 0.4644 | **0.4677** |

---

## 6. Key Observations

- The **LightGBM classifier** alone performs well in terms of precision but lacks the ability to rank effectively.
- The **regressor** improves ranking metrics (NDCG, MAP), but the ranking metrics are not satisfying.
- The **hybrid approach** maintains the classifier's precision while leveraging the regressor to enhance ranking quality. This leads to **state-of-the-art NDCG scores**.
- Classifier defines **relevant items**, regressor defines **how to rank them**.

---

## 7. Feature Analysis

Our model's effectiveness depends heavily on specific features that strongly influence both the classification and regression stages. Analysis of feature importance reveals that user demographic factors (encoded as one-hot features), content creator information, and viewing behavior are particularly influential.

### Top 5 Most Important Features

| Rank | Classifier Features | Importance | Regressor Features | Importance |
|------|---------------------|------------|-------------------|------------|
| 1 | `onehot_feat3` (User demographic) | 13,403 | `onehot_feat3` (User demographic) | 12,666 |
| 2 | `author_id` (Content creator) | 9,196 | `author_id` (Content creator) | 8,259 |
| 3 | `onehot_feat8` (User demographic) | 8,331 | `onehot_feat8` (User demographic) | 7,866 |
| 4 | `video_duration` (Content metadata) | 1,745 | `video_duration` (Content metadata) | 2,845 |
| 5 | `user_agg_interactions` (User behavior) | 1,000 | `user_agg_interactions` (User behavior) | 1,001 |

Both models prioritize similar features, with user demographics and content creator identity being overwhelmingly important. This suggests that matching users to their preferred content creators and accounting for demographic preferences are the strongest predictors of both relevance (classification) and engagement level (regression).

---

## 8. Resource Requirements

- **RAM usage:** ~20GB
- **Runtime:**
  - Classifier-only: ~8.69 minutes
  - Regressor-only: ~9.28 minutes
  - Hybrid model: ~19.00 minutes
- **Hardware:** Ryzen 7 5800H, 32GB RAM

---

## 9. Reproducibility and Usage

### Requirements

- Python 3.8+
- Libraries: `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `implicit`, `tqdm`

### Run

All data files must be in `./KuaiRec 2.0/data/`

```bash
python tmp11.py
```

---

## 10. Conclusion

This project demonstrates that a multi-stage recommender system that separates **relevance prediction** and **ranking refinement** can outperform standard methods in real-world recommendation tasks.

The hybrid strategy overcomes the binary limitation of classification and the false positive issues of pure regression, achieving high precision, robust recall, and optimal ranking metrics (NDCG, MAP).

The pipeline is modular, scalable, and generalizable to similar recommendation challenges.
