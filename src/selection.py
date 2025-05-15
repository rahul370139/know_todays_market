import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit


def drop_correlated_features(df: pd.DataFrame,
                             feature_cols: list,
                             threshold: float = 0.99,
                             retain: list = None) -> (list, list):
    """
    Remove features whose pairwise correlation exceeds the threshold,
    but keep any in the `retain` list regardless.
    Returns (reduced_feature_list, dropped_feature_list).
    """
    retain = set(retain or [])
    corr = df[feature_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns
               if any(upper[col] > threshold) and col not in retain]
    reduced = [c for c in feature_cols if c not in to_drop] + [c for c in feature_cols if c in retain]
    return reduced, to_drop


def compute_rf_importance(df: pd.DataFrame,
                          feature_cols: list,
                          target_cols: list,
                          n_splits: int = 10,
                          random_state: int = 42,
                          n_estimators: int = 100) -> pd.Series:
    """
    Compute average Random Forest feature importances via time-series CV.
    Returns a Series indexed by feature name.
    """
    X = df[feature_cols].values
    y = df[target_cols].values
    tscv = TimeSeriesSplit(n_splits=n_splits)
    importances = np.zeros(len(feature_cols))
    for train_idx, _ in tscv.split(X):
        rf = RandomForestRegressor(random_state=random_state, n_estimators=n_estimators)
        rf.fit(X[train_idx], y[train_idx])
        importances += rf.feature_importances_
    importances /= n_splits
    return pd.Series(importances, index=feature_cols)


def compute_xgb_importance(df: pd.DataFrame,
                           feature_cols: list,
                           target_cols: list,
                           n_splits: int = 10,
                           random_state: int = 42,
                           n_estimators: int = 100) -> pd.Series:
    """
    Compute average XGBoost feature importances via time-series CV.
    Returns a Series indexed by feature name.
    """
    X = df[feature_cols].values
    y = df[target_cols].values
    tscv = TimeSeriesSplit(n_splits=n_splits)
    importances = np.zeros(len(feature_cols))
    for train_idx, _ in tscv.split(X):
        xgb = XGBRegressor(objective='reg:squarederror',
                           random_state=random_state,
                           n_estimators=n_estimators)
        xgb.fit(X[train_idx], y[train_idx])
        importances += xgb.feature_importances_
    importances /= n_splits
    return pd.Series(importances, index=feature_cols)


def combine_importances(rf_imp: pd.Series,
                        xgb_imp: pd.Series) -> pd.DataFrame:
    """
    Merge RF and XGB importances and compute an average ranking.
    """
    df = pd.DataFrame({
        'Feature': rf_imp.index,
        'Importance_rf': rf_imp.values,
        'Importance_xgb': xgb_imp.values
    })
    df['Average_Importance'] = df[['Importance_rf', 'Importance_xgb']].mean(axis=1)
    return df.sort_values('Average_Importance', ascending=False).reset_index(drop=True)


def select_top_features(importance_df: pd.DataFrame,
                        top_n: int = 25,
                        extra_features: list = None) -> list:
    """
    Choose the top_n features by average importance,
    plus any extra_features you want to enforce.
    """
    extra = set(extra_features or [])
    top = importance_df.head(top_n)['Feature'].tolist()
    return list(set(top) | extra)
