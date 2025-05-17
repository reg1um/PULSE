import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import gc
import time
import os
import re
from tqdm import tqdm

# Global constants
DATA_DIR = './KuaiRec 2.0/data'
USE_SMALL_TRAIN_SAMPLE = False 
TRAIN_SAMPLE_FRAC = 0.1
USE_SMALL_TEST_SAMPLE = False 
TEST_SAMPLE_FRAC = 0.2

# ALS parameters
ALS_FACTORS = 64
ALS_REGULARIZATION = 0.1
ALS_ITERATIONS = 20

# LightGBM Classifier parameters
LGBM_CLASSIFIER_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'n_estimators': 1000,
    'learning_rate': 0.03,
    'num_leaves': 41,
    'max_depth': -1,
    'seed': 42,
    'n_jobs': -1,
    'verbose': -1,
    'colsample_bytree': 0.7,
    'subsample': 0.7,
    'reg_alpha': 0.05,
    'reg_lambda': 0.05,
}

# LightGBM Regressor parameters
LGBM_REGRESSOR_PARAMS = {
    'objective': 'regression_l1',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'n_estimators': 1000,
    'learning_rate': 0.03,
    'num_leaves': 41,
    'max_depth': -1,
    'seed': 123,
    'n_jobs': -1,
    'verbose': -1,
    'colsample_bytree': 0.7,
    'subsample': 0.7,
    'reg_alpha': 0.05,
    'reg_lambda': 0.05,
}
LGBM_EARLY_STOPPING_ROUNDS = 100

K_VALUES_EVAL = [10, 20, 50, 100, 200, 500, 1000]
NUM_TFIDF_FEATURES = 50
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Utility Functions

# Function to reduce memory usage of DataFrame by downcasting numeric types
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            if df[col].isnull().all():
                if verbose: print(f"Column {col} is all NaN, skipping min/max for this column.")
                continue
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else: 
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print(f'Mem. usage decreased to {end_mem:5.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df

# Function to load and preprocess interaction data
def load_and_preprocess_interactions(file_path, sample_frac=1.0, use_sample=False):
    print(f"Loading {file_path}...")
    df = pd.read_csv(os.path.join(DATA_DIR, file_path),
                     usecols=['user_id', 'video_id', 'watch_ratio', 'time', 'video_duration'])
    if use_sample and sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)

    df = reduce_mem_usage(df)
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df['hour_of_day'] = df['time'].dt.hour.fillna(-1).astype(np.int8)
    df['day_of_week'] = df['time'].dt.dayofweek.fillna(-1).astype(np.int8)

    df['watch_ratio_original'] = pd.to_numeric(df['watch_ratio'], errors='coerce').fillna(0).astype(np.float32)
    df['liked'] = (df['watch_ratio_original'] > 1.0).astype(np.int8)

    if not df.empty and df['watch_ratio_original'].notna().any():
        quantile_val = df[df['watch_ratio_original'] <= 10]['watch_ratio_original'].quantile(0.99) if df['watch_ratio_original'].nunique() > 1 else 10.0
        if pd.isna(quantile_val) or quantile_val == 0 : quantile_val = 10.0 
    else:
        quantile_val = 10.0 
    df['watch_ratio_clipped'] = np.clip(df['watch_ratio_original'], 0, quantile_val)
    df['watch_ratio_clipped'] = df['watch_ratio_clipped'].fillna(0).astype(np.float32)

    print(f"Loaded {file_path}: {df.shape}")
    return df

# Function to get TF-IDF features from captions
def get_text_features(num_tfidf_feats=NUM_TFIDF_FEATURES):
    print("Generating TF-IDF features for videos...")
    captions_df_path = os.path.join(DATA_DIR, 'kuairec_caption_category.csv')
    if not os.path.exists(captions_df_path):
        print(f"Caption file not found: {captions_df_path}. Returning empty TF-IDF features.")
        empty_df = pd.DataFrame(columns=['video_id'] + [f'tfidf_{i}' for i in range(num_tfidf_feats)])
        empty_df['video_id'] = pd.Series(dtype=np.int32) # Ensure correct dtype even for empty
        return empty_df

    captions_df = pd.read_csv(captions_df_path,
                              usecols=['video_id', 'manual_cover_text', 'caption', 'topic_tag'],
                              engine='python', encoding='utf-8')
    captions_df['video_id'] = pd.to_numeric(captions_df['video_id'], errors='coerce')
    captions_df.dropna(subset=['video_id'], inplace=True)
    captions_df['video_id'] = captions_df['video_id'].astype(np.int32) 

    captions_df['combined_text'] = captions_df['manual_cover_text'].fillna('') + " " + \
                                   captions_df['caption'].fillna('') + " " + \
                                   captions_df['topic_tag'].fillna('')
    captions_df['combined_text'] = captions_df['combined_text'].str.lower().apply(lambda x: re.sub(r'[^\w\s]', '', x))

    # Check if combined_text is empty or contains only whitespace
    if captions_df.empty or captions_df['combined_text'].str.strip().eq('').all():
        print("No text data for TF-IDF. Returning empty df with video_id.")
        empty_df = pd.DataFrame(columns=['video_id'] + [f'tfidf_{i}' for i in range(num_tfidf_feats)])
        empty_df['video_id'] = pd.Series(dtype=np.int32)
        return empty_df

    tfidf = TfidfVectorizer(max_features=num_tfidf_feats, stop_words='english', ngram_range=(1,1))
    try:
        tfidf_matrix = tfidf.fit_transform(captions_df['combined_text'])
    except ValueError as e:
        print(f"TF-IDF fit_transform error: {e}. Returning empty features.")
        empty_df = pd.DataFrame(columns=['video_id'] + [f'tfidf_{i}' for i in range(num_tfidf_feats)])
        empty_df['video_id'] = pd.Series(dtype=np.int32)
        return empty_df

    actual_num_tfidf_features = tfidf_matrix.shape[1]
    tfidf_feature_names = [f'tfidf_{i}' for i in range(actual_num_tfidf_features)]

    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_feature_names)
    tfidf_df = pd.concat([captions_df[['video_id']].reset_index(drop=True), tfidf_df], axis=1)
    tfidf_df['video_id'] = tfidf_df['video_id'].astype(np.int32)

    tfidf_df = tfidf_df.groupby('video_id').mean().reset_index() 
    tfidf_df['video_id'] = tfidf_df['video_id'].astype(np.int32) 

    if actual_num_tfidf_features < num_tfidf_feats:
        for i in range(actual_num_tfidf_features, num_tfidf_feats):
            tfidf_df[f'tfidf_{i}'] = 0.0

    for col in tfidf_df.columns:
        if col != 'video_id':
            tfidf_df[col] = tfidf_df[col].astype(np.float16)

    print(f"Generated TF-IDF features: {tfidf_df.shape}, video_id dtype: {tfidf_df['video_id'].dtype}")
    return tfidf_df

# Function to load extended features (user/item features)
def load_extended_features(train_interactions_df):

    # User features
    print("Loading user features (user_features.csv)...")
    user_features_path = os.path.join(DATA_DIR, 'user_features.csv')
    user_features_df = pd.read_csv(user_features_path) if os.path.exists(user_features_path) else pd.DataFrame(columns=['user_id'])
    user_features_df['user_id'] = pd.to_numeric(user_features_df['user_id'], errors='coerce').fillna(-1).astype(np.int32) # FillNa before astype
    user_features_df.dropna(subset=['user_id'], inplace=True) # Should be redundant if filled with -1

    if 'user_id' in user_features_df.columns and len(user_features_df.columns) > 1:
        user_ids_temp_user = user_features_df['user_id'].copy()
        user_features_df_reduced = reduce_mem_usage(user_features_df.drop(columns=['user_id']), verbose=False)
        user_features_df = pd.concat([user_ids_temp_user.reset_index(drop=True), user_features_df_reduced.reset_index(drop=True)], axis=1)
    if 'user_id' not in user_features_df.columns: user_features_df['user_id'] = pd.Series(dtype=np.int32) # Ensure exists
    user_features_df['user_id'] = user_features_df['user_id'].astype(np.int32)

    # Categorical features
    user_cat_cols = ['user_active_degree', 'follow_user_num_range', 'fans_user_num_range', 'friend_user_num_range', 'register_days_range']
    for col in user_cat_cols:
        if col in user_features_df.columns:
            le = LabelEncoder()
            user_features_df[col] = le.fit_transform(user_features_df[col].astype(str))

    # One-hot encoded features
    onehot_cols = [f'onehot_feat{i}' for i in range(18)] # Ensure these cols exist or are created
    for col in onehot_cols:
        if col in user_features_df.columns:
             user_features_df[col] = pd.to_numeric(user_features_df[col], errors='coerce').fillna(0)
        else: user_features_df[col] = 0 

    # Item features
    print("Loading item category features (kuairec_caption_category.csv)...")
    item_cat_path = os.path.join(DATA_DIR, 'kuairec_caption_category.csv')
    item_main_features_df = pd.read_csv(item_cat_path, usecols=['video_id', 'first_level_category_name', 'second_level_category_name', 'third_level_category_name'], engine='python', encoding='utf-8') if os.path.exists(item_cat_path) else pd.DataFrame(columns=['video_id'])
    item_main_features_df['video_id'] = pd.to_numeric(item_main_features_df['video_id'], errors='coerce').fillna(-1).astype(np.int32)
    item_main_features_df.dropna(subset=['video_id'], inplace=True)
    item_main_features_df.drop_duplicates(subset=['video_id'], inplace=True)
    
    if 'video_id' in item_main_features_df.columns and len(item_main_features_df.columns) > 1:
        video_ids_temp_item = item_main_features_df['video_id'].copy()
        cols_to_reduce_item = [col for col in item_main_features_df.columns if col != 'video_id']
        item_main_features_df_reduced = reduce_mem_usage(item_main_features_df[cols_to_reduce_item], verbose=False) if cols_to_reduce_item else pd.DataFrame(index=item_main_features_df.index)
        item_main_features_df = pd.concat([video_ids_temp_item.reset_index(drop=True), item_main_features_df_reduced.reset_index(drop=True)], axis=1)
    if 'video_id' not in item_main_features_df.columns: item_main_features_df['video_id'] = pd.Series(dtype=np.int32)
    item_main_features_df['video_id'] = item_main_features_df['video_id'].astype(np.int32)

    # Categorical features
    item_cat_cols = ['first_level_category_name', 'second_level_category_name', 'third_level_category_name']
    for col in item_cat_cols:
        if col in item_main_features_df.columns: 
            le = LabelEncoder()
            item_main_features_df[col] = le.fit_transform(item_main_features_df[col].astype(str))

    # Text features
    text_features_df = get_text_features()
    item_main_features_df = item_main_features_df.merge(text_features_df, on='video_id', how='left')
    # Fill NaNs from TF-IDF merge, especially if text_features_df was empty or had missing video_ids
    tfidf_cols = [f'tfidf_{i}' for i in range(NUM_TFIDF_FEATURES)]
    for col in tfidf_cols:
        if col in item_main_features_df.columns:
            item_main_features_df[col] = item_main_features_df[col].fillna(0.0).astype(np.float16)
        else: # If a tfidf col wasn't created (e.g., due to fewer actual features)
            item_main_features_df[col] = 0.0 


    print("Loading and aggregating item daily features (item_daily_features.csv)...")
    item_daily_path = os.path.join(DATA_DIR, 'item_daily_features.csv')
    item_daily_df = pd.read_csv(item_daily_path) if os.path.exists(item_daily_path) else pd.DataFrame()

    if not item_daily_df.empty:
        item_daily_df['video_id'] = pd.to_numeric(item_daily_df['video_id'], errors='coerce').fillna(-1).astype(np.int32)
        item_daily_df.dropna(subset=['video_id'], inplace=True)
        if 'author_id' in item_daily_df.columns:
            item_daily_df['author_id'] = pd.to_numeric(item_daily_df['author_id'], errors='coerce').fillna(-1).astype(np.int32)
            video_author_map = item_daily_df.groupby('video_id')['author_id'].first().reset_index()
            video_author_map['video_id'] = video_author_map['video_id'].astype(np.int32) 
            video_author_map['author_id'] = video_author_map['author_id'].astype(np.int32)
            item_main_features_df = item_main_features_df.merge(video_author_map, on='video_id', how='left')
        if 'author_id' not in item_main_features_df.columns: item_main_features_df['author_id'] = -1 
        item_main_features_df['author_id'] = item_main_features_df['author_id'].fillna(-1).astype(np.int32)

        daily_numeric_cols = ['show_cnt', 'play_cnt', 'like_cnt', 'comment_cnt', 'share_cnt', 'collect_cnt'] 
        daily_numeric_cols_exist = [col for col in daily_numeric_cols if col in item_daily_df.columns]
        if daily_numeric_cols_exist:
            for col in daily_numeric_cols_exist: item_daily_df[col] = pd.to_numeric(item_daily_df[col], errors='coerce').fillna(0)
            aggregated_daily_item_features = item_daily_df.groupby('video_id')[daily_numeric_cols_exist].mean().reset_index()
            aggregated_daily_item_features['video_id'] = aggregated_daily_item_features['video_id'].astype(np.int32) 
            new_colnames = {col: f'avg_daily_{col}' for col in daily_numeric_cols_exist}
            aggregated_daily_item_features.rename(columns=new_colnames, inplace=True)
            item_main_features_df = item_main_features_df.merge(aggregated_daily_item_features, on='video_id', how='left')
            for col_name in new_colnames.values():
                 if col_name in item_main_features_df.columns: item_main_features_df[col_name] = item_main_features_df[col_name].fillna(0.0).astype(np.float32)
    else: 
        if 'author_id' not in item_main_features_df.columns: item_main_features_df['author_id'] = -1
        item_main_features_df['author_id'] = item_main_features_df['author_id'].fillna(-1).astype(np.int32)

    # Popularity features
    print("Generating popularity features from training interactions...")
    train_interactions_df['video_id'] = train_interactions_df['video_id'].astype(np.int32)
    train_interactions_df['user_id'] = train_interactions_df['user_id'].astype(np.int32)
    video_agg_views = train_interactions_df.groupby('video_id')['user_id'].count().reset_index(name='video_agg_views')
    video_agg_unique_users = train_interactions_df.groupby('video_id')['user_id'].nunique().reset_index(name='video_agg_unique_users')
    user_agg_interactions = train_interactions_df.groupby('user_id')['video_id'].count().reset_index(name='user_agg_interactions')
    user_agg_unique_videos = train_interactions_df.groupby('user_id')['video_id'].nunique().reset_index(name='user_agg_unique_videos')

    user_features_df = user_features_df.merge(user_agg_interactions, on='user_id', how='left')
    user_features_df = user_features_df.merge(user_agg_unique_videos, on='user_id', how='left')
    item_main_features_df = item_main_features_df.merge(video_agg_views, on='video_id', how='left')
    item_main_features_df = item_main_features_df.merge(video_agg_unique_users, on='video_id', how='left')

    for col in ['user_agg_interactions', 'user_agg_unique_videos']:
        if col in user_features_df.columns: user_features_df[col] = user_features_df[col].fillna(0).astype(np.int32)
    for col in ['video_agg_views', 'video_agg_unique_users']:
        if col in item_main_features_df.columns: item_main_features_df[col] = item_main_features_df[col].fillna(0).astype(np.int32)
    
    # Ensure IDs are final type
    user_features_df['user_id'] = user_features_df['user_id'].astype(np.int32)
    item_main_features_df['video_id'] = item_main_features_df['video_id'].astype(np.int32)
    if 'author_id' in item_main_features_df.columns:
        item_main_features_df['author_id'] = item_main_features_df['author_id'].astype(np.int32)


    print(f"Final user_features_df shape: {user_features_df.shape}, user_id dtype: {user_features_df['user_id'].dtype if 'user_id' in user_features_df else 'N/A'}")
    print(f"Final item_features_df shape: {item_main_features_df.shape}, video_id dtype: {item_main_features_df['video_id'].dtype if 'video_id' in item_main_features_df else 'N/A'}")
    return user_features_df, item_main_features_df

# Function to calculate ALS scores for user-item pairs
def get_als_scores_vectorized(df, user_factors, item_factors, user_id_map, video_id_map):
    user_indices = df['user_id'].map(user_id_map).fillna(-1).astype(int).values
    item_indices = df['video_id'].map(video_id_map).fillna(-1).astype(int).values
    scores = np.zeros(len(df), dtype=np.float32)
    valid_user_mask = (user_indices >= 0) & (user_indices < user_factors.shape[0])
    valid_item_mask = (item_indices >= 0) & (item_indices < item_factors.shape[0])
    valid_mask = valid_user_mask & valid_item_mask
    if np.any(valid_mask):
        valid_user_indices = user_indices[valid_mask]
        valid_item_indices = item_indices[valid_mask]
        dot_products = np.sum(user_factors[valid_user_indices] * item_factors[valid_item_indices], axis=1)
        scores[valid_mask] = dot_products
    return scores.astype(np.float32)

# Function to create final features for LightGBM
def create_lgbm_features_final(df, user_feats, item_feats):
    df['user_id'] = df['user_id'].astype(np.int32)
    if 'user_id' in user_feats.columns: user_feats['user_id'] = user_feats['user_id'].astype(np.int32)
    df['video_id'] = df['video_id'].astype(np.int32)
    if 'video_id' in item_feats.columns: item_feats['video_id'] = item_feats['video_id'].astype(np.int32)
    
    df = df.merge(user_feats, on='user_id', how='left')
    df = df.merge(item_feats, on='video_id', how='left')

    all_feature_columns = user_feats.columns.tolist() + item_feats.columns.tolist()
    for col in all_feature_columns:
        if col in ['user_id', 'video_id']: continue
        if col in df.columns: # Check if column exists after merge
            if df[col].dtype.kind in 'ifc': 
                df[col] = df[col].fillna(0.0)
                if df[col].min() >= np.finfo(np.float16).min and df[col].max() <= np.finfo(np.float16).max and not df[col].isnull().any():
                     df[col] = df[col].astype(np.float16)
                else:
                     df[col] = df[col].astype(np.float32) # Default to float32 for safety
            elif df[col].dtype.kind in 'i': # integer
                df[col] = df[col].fillna(-1) # Using -1 for missing int
            else: # object or other
                df[col] = df[col].fillna("-1_missing") # Using placeholder for missing categorical/object
    
    if 'video_duration' in df.columns: # video_duration is from original interaction df
        df['video_duration'] = df['video_duration'].fillna(df['video_duration'].median()).astype(np.float32)
    return df

# MAIN SCRIPT
start_total_time = time.time()

# 1. Load Data
train_df = load_and_preprocess_interactions('big_matrix.csv', 
                                            sample_frac=TRAIN_SAMPLE_FRAC, use_sample=USE_SMALL_TRAIN_SAMPLE)
test_df = load_and_preprocess_interactions('small_matrix.csv',
                                           sample_frac=TEST_SAMPLE_FRAC, use_sample=USE_SMALL_TEST_SAMPLE)
user_features_df, item_features_df = load_extended_features(train_df.copy())

# 2. ALS Model Training
als_train_user_ids = train_df['user_id'].unique()
als_train_video_ids = train_df['video_id'].unique()
user_id_to_idx_als = {id_val: i for i, id_val in enumerate(als_train_user_ids)}
video_id_to_idx_als = {id_val: i for i, id_val in enumerate(als_train_video_ids)}
num_users_als = len(user_id_to_idx_als)
num_items_als = len(video_id_to_idx_als)

if num_users_als > 0 and num_items_als > 0: # Only train ALS if data exists
    train_user_indices_als = train_df['user_id'].map(user_id_to_idx_als).values
    train_video_indices_als = train_df['video_id'].map(video_id_to_idx_als).values
    print("\nTraining ALS model...")
    start_als_time = time.time()
    interaction_data = (train_df['watch_ratio_clipped'] + 1).astype(np.float32) 
    interaction_matrix = csr_matrix((interaction_data, 
                                    (train_user_indices_als, train_video_indices_als)),
                                    shape=(num_users_als, num_items_als))
    als_model = AlternatingLeastSquares(factors=ALS_FACTORS, regularization=ALS_REGULARIZATION, 
                                        iterations=ALS_ITERATIONS, random_state=42, use_gpu=False)
    als_model.fit(interaction_matrix)
    print(f"ALS model training took {time.time() - start_als_time:.2f} seconds.")
    user_factors = als_model.user_factors.astype(np.float32)
    item_factors = als_model.item_factors.astype(np.float32)
    print("Calculating ALS scores for train_df...")
    train_df['als_score'] = get_als_scores_vectorized(train_df, user_factors, item_factors, user_id_to_idx_als, video_id_to_idx_als)
    print("Calculating ALS scores for test_df...")
    test_df['als_score'] = get_als_scores_vectorized(test_df, user_factors, item_factors, user_id_to_idx_als, video_id_to_idx_als)
    del interaction_matrix, user_factors, item_factors # Clean up ALS factors
else:
    print("Skipping ALS training due to empty user/item sets for ALS.")
    train_df['als_score'] = 0.0 # Add dummy als_score column
    test_df['als_score'] = 0.0


# 3. Feature Engineering for LightGBM
print("\nFeature engineering for LightGBM...")
train_lgbm_df = create_lgbm_features_final(train_df.copy(), user_features_df, item_features_df)
test_lgbm_df = create_lgbm_features_final(test_df.copy(), user_features_df, item_features_df)

# Define feature columns for LightGBM
base_feature_cols = ['hour_of_day', 'day_of_week', 'video_duration', 'als_score']
user_feature_cols_names = [col for col in user_features_df.columns if col not in ['user_id']]
item_feature_cols_names = [col for col in item_features_df.columns if col not in ['video_id']] # author_id is part of item_features_df

# Combine all feature columns
feature_cols = base_feature_cols + user_feature_cols_names + item_feature_cols_names
cols_to_remove_from_features = ['watch_ratio', 'watch_ratio_original', 'watch_ratio_clipped', 'liked', 'time']
feature_cols = [f for f in feature_cols if f not in cols_to_remove_from_features]
feature_cols = sorted(list(set(f for f in feature_cols if f in train_lgbm_df.columns))) # Ensure features exist in df

# Categorical features
categorical_features = ['hour_of_day', 'day_of_week'] 
if 'author_id' in feature_cols: categorical_features.append('author_id')
categorical_features += [col for col in user_feature_cols_names if ('_range' in col or 'degree' in col or 'onehot_feat' in col) and col in feature_cols]
categorical_features += [col for col in item_feature_cols_names if 'category_name' in col and col in feature_cols]
categorical_features = sorted(list(set(categorical_features)))

X_train_lgbm = train_lgbm_df[feature_cols].copy()
X_test_lgbm = test_lgbm_df[feature_cols].copy()

print("Converting specified columns to 'category' dtype for LightGBM...")
for col in categorical_features:
    if col in X_train_lgbm.columns:
        if X_train_lgbm[col].dtype == np.float16: X_train_lgbm[col] = X_train_lgbm[col].astype(np.float32)
        try: X_train_lgbm[col] = X_train_lgbm[col].astype('category')
        except Exception as e: print(f"  ERROR converting '{col}' in X_train_lgbm to category: {e}. Dtype: {X_train_lgbm[col].dtype}")
    if col in X_test_lgbm.columns:
        if X_test_lgbm[col].dtype == np.float16: X_test_lgbm[col] = X_test_lgbm[col].astype(np.float32)
        try: X_test_lgbm[col] = X_test_lgbm[col].astype('category')
        except Exception as e: print(f"  ERROR converting '{col}' in X_test_lgbm to category: {e}. Dtype: {X_test_lgbm[col].dtype}")

print(f"Number of features for LightGBM: {len(feature_cols)}")
del train_df, test_df, user_features_df, item_features_df # Main DFs are now train_lgbm_df, test_lgbm_df
gc.collect()

# 4. LightGBM Classifier Training
print("\nTraining LightGBM Classifier")
y_train_classifier = train_lgbm_df['liked']
X_train_part_c, X_val_part_c, y_train_part_c, y_val_part_c = train_test_split(
    X_train_lgbm, y_train_classifier, test_size=0.15, random_state=42, stratify=y_train_classifier
)
lgbm_classifier_model = lgb.LGBMClassifier(**LGBM_CLASSIFIER_PARAMS)
lgbm_classifier_model.fit(X_train_part_c, y_train_part_c,
                          eval_set=[(X_val_part_c, y_val_part_c)],
                          eval_metric='auc', 
                          callbacks=[lgb.early_stopping(LGBM_EARLY_STOPPING_ROUNDS, verbose=100)],
                          categorical_feature=categorical_features)
print("LightGBM Classifier training finished.")

# 5. LightGBM Regressor Training
print("\nTraining LightGBM Regressor")
y_train_regressor = train_lgbm_df['watch_ratio_clipped']
X_train_part_r, X_val_part_r, y_train_part_r, y_val_part_r = train_test_split(
    X_train_lgbm, y_train_regressor, test_size=0.15, random_state=123
)
lgbm_regressor_model = lgb.LGBMRegressor(**LGBM_REGRESSOR_PARAMS)
lgbm_regressor_model.fit(X_train_part_r, y_train_part_r,
                         eval_set=[(X_val_part_r, y_val_part_r)],
                         eval_metric='mae', 
                         callbacks=[lgb.early_stopping(LGBM_EARLY_STOPPING_ROUNDS, verbose=100)],
                         categorical_feature=categorical_features)
print("LightGBM Regressor training finished.")

# 6. Prediction on Test Set
print("\nPredicting on test set with both models...")
test_lgbm_df['classifier_score'] = lgbm_classifier_model.predict_proba(X_test_lgbm)[:, 1].astype(np.float32)
test_lgbm_df['regressor_score'] = lgbm_regressor_model.predict(X_test_lgbm).astype(np.float32)

# 7. Evaluation
print("\nEvaluating Recommender System (Focus on re-ranking classifier's top items)...")
all_precisions = {k: [] for k in K_VALUES_EVAL}
all_recalls = {k: [] for k in K_VALUES_EVAL}
all_ndcgs = {k: [] for k in K_VALUES_EVAL}
all_maps = {k: [] for k in K_VALUES_EVAL}

test_user_groups = test_lgbm_df.groupby('user_id')

for user_id, group in tqdm(test_user_groups, desc="Evaluating users"):
    if group.empty:
        for k_val in K_VALUES_EVAL:
            all_precisions[k_val].append(0); all_recalls[k_val].append(0); all_ndcgs[k_val].append(0); all_maps[k_val].append(0)
        continue

    # Get the classifier's ranking of ALL items for the user
    classifier_ranked_group = group.sort_values('classifier_score', ascending=False)
    
    # For P, R, MAP, and NDCG, we will consider the items *as selected by the classifier* for the top K,
    # but for NDCG and MAP, their *order within those K* will be determined by the regressor.

    for k_val in K_VALUES_EVAL:
        num_items_for_user_total = len(classifier_ranked_group)
        current_k = min(k_val, num_items_for_user_total)

        if current_k == 0:
            all_precisions[k_val].append(0); all_recalls[k_val].append(0); all_ndcgs[k_val].append(0); all_maps[k_val].append(0)
            continue

        # These are the exact top K items according to the classifier
        top_k_items_by_classifier = classifier_ranked_group.head(current_k)

        # Precision and Recall
        true_relevance_binary_classifier_top_k = top_k_items_by_classifier['liked'].values
        
        precision_at_k = np.sum(true_relevance_binary_classifier_top_k) / current_k if current_k > 0 else 0
        all_precisions[k_val].append(precision_at_k)

        total_relevant_items_for_user_in_original_group = np.sum(group['liked'].values) # Denominator from all user items
        recall_at_k = np.sum(true_relevance_binary_classifier_top_k) / total_relevant_items_for_user_in_original_group if total_relevant_items_for_user_in_original_group > 0 else 0
        all_recalls[k_val].append(recall_at_k)

        # Re-rank these top K items using the regressor for NDCG and MAP
        # The 'regressor_score' is already in top_k_items_by_classifier from the 'group' df
        reranked_top_k_by_regressor = top_k_items_by_classifier.sort_values('regressor_score', ascending=False)

        true_relevance_binary_reranked = reranked_top_k_by_regressor['liked'].values
        actual_watch_ratio_for_gain_reranked = reranked_top_k_by_regressor['watch_ratio_clipped'].values

        # MAP
        ap_sum, hits = 0, 0
        for i, pred_label_binary in enumerate(true_relevance_binary_reranked): # Iterate through reranked order
            if pred_label_binary == 1:
                hits += 1
                ap_sum += hits / (i + 1)
        total_relevant_in_classifier_top_k = np.sum(true_relevance_binary_classifier_top_k)
        # ap_at_k = ap_sum / total_relevant_in_classifier_top_k if total_relevant_in_classifier_top_k > 0 else 0
        ap_at_k = ap_sum / total_relevant_items_for_user_in_original_group if total_relevant_items_for_user_in_original_group > 0 else 0
        all_maps[k_val].append(ap_at_k)

        # NDCG
        gains_dcg_values = np.where(true_relevance_binary_reranked == 1, actual_watch_ratio_for_gain_reranked, 0.0)
        discounts_dcg = np.log2(np.arange(current_k) + 2)
        dcg_k = np.sum(gains_dcg_values / discounts_dcg)

        # IDCG: Ideal ranking of the *top_k_items_by_classifier* items.
        ideal_gains_within_classifier_top_k = np.where(top_k_items_by_classifier['liked'].values == 1, 
                                                       top_k_items_by_classifier['watch_ratio_clipped'].values, 
                                                       0.0)
        ideal_order_gains = np.sort(ideal_gains_within_classifier_top_k)[::-1]
        # The number of items for IDCG is 'current_k' (or number of non-zero gains in ideal_order_gains)
        discounts_idcg = np.log2(np.arange(len(ideal_order_gains)) + 2)
        idcg_k = np.sum(ideal_order_gains / discounts_idcg)

        ndcg_k = dcg_k / idcg_k if idcg_k > 0 else 0.0
        all_ndcgs[k_val].append(ndcg_k)


print("\nFinal Metrics (Averaged over users)")
for k_val in K_VALUES_EVAL:
    mean_precision = np.mean(all_precisions[k_val])
    mean_recall = np.mean(all_recalls[k_val])
    mean_ndcg = np.mean(all_ndcgs[k_val])
    mean_map = np.mean(all_maps[k_val])
    print(f"Precision@{k_val}: {mean_precision:.4f}")
    print(f"Recall@{k_val}:    {mean_recall:.4f}")
    print(f"NDCG@{k_val}:      {mean_ndcg:.4f}")
    print(f"MAP@{k_val}:       {mean_map:.4f}")

total_run_time = time.time() - start_total_time
print(f"\nTotal script execution time: {total_run_time/60:.2f} minutes.")

# Feature Analysis

if hasattr(lgbm_classifier_model, 'feature_importances_') and X_train_lgbm.shape[0] > 0:
    feature_imp_clf = pd.DataFrame({'feature': X_train_lgbm.columns, 'importance': lgbm_classifier_model.feature_importances_})
    print("\nTop 15 LightGBM Classifier Feature Importances:")
    print(feature_imp_clf.sort_values('importance', ascending=False).head(15))

if hasattr(lgbm_regressor_model, 'feature_importances_') and X_train_lgbm.shape[0] > 0 :
    feature_imp_reg = pd.DataFrame({'feature': X_train_lgbm.columns, 'importance': lgbm_regressor_model.feature_importances_})
    print("\nTop 15 LightGBM Regressor Feature Importances:")
    print(feature_imp_reg.sort_values('importance', ascending=False).head(15))
