import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

def load_kaggle_csv(path):
    """
    Expect CSV with timestamp column and sensor columns including 'power', 'wind_speed', etc.
    """
    df = pd.read_csv(path, parse_dates=['timestamp'], infer_datetime_format=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

def engineer_features(df, feature_cols=None):
    if feature_cols is None:
        # auto-detect numeric features except timestamp
        feature_cols = df.select_dtypes(include=np.number).columns.tolist()
    # rolling features
    df_feat = df.copy()
    for c in feature_cols:
        df_feat[f'{c}_rmean_3'] = df_feat[c].rolling(3, min_periods=1).mean()
        df_feat[f'{c}_rstd_3'] = df_feat[c].rolling(3, min_periods=1).std().fillna(0)
    return df_feat.fillna(method='bfill').fillna(0), feature_cols

def make_sequences(arr, seq_len=60):
    # arr shape (n_samples, n_features)
    seqs = []
    for i in range(len(arr) - seq_len + 1):
        seqs.append(arr[i:i+seq_len])
    return np.stack(seqs)

def preprocess_for_training(csv_path, out_dir='data/processed', seq_len=60, test_size=0.2):
    os.makedirs(out_dir, exist_ok=True)
    df = load_kaggle_csv(csv_path)
    df_feat, feature_cols = engineer_features(df)
    scaler = StandardScaler()
    X = scaler.fit_transform(df_feat[feature_cols])
    joblib.dump(scaler, os.path.join(out_dir, 'scaler.joblib'))
    seqs = make_sequences(X, seq_len=seq_len)
    train, test = train_test_split(seqs, test_size=test_size, shuffle=False)
    np.save(os.path.join(out_dir, 'train.npy'), train)
    np.save(os.path.join(out_dir, 'test.npy'), test)
    print(f"Saved processed data to {out_dir}")
    return train, test, feature_cols
