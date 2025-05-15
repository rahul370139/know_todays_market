import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import timedelta
from sklearn.metrics import (mean_squared_error,
                             mean_absolute_error,
                             mean_absolute_percentage_error)
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from .models import create_sequences, create_gru_model


def directional_accuracy(y_true: np.ndarray,
                         y_pred: np.ndarray) -> float:
    """
    % of times the predicted direction matches the actual.
    """
    if len(y_true) < 2:
        return np.nan
    dir_true = np.sign(np.diff(y_true))
    dir_pred = np.sign(np.diff(y_pred))
    return float((dir_true == dir_pred).mean() * 100)


def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray) -> dict:
    """
    Return dict with MSE, MAE, RMSE, MAPE.
    """
    non_zero = y_true != 0
    mape = (mean_absolute_percentage_error(y_true[non_zero], y_pred[non_zero])
            * 100) if np.any(non_zero) else np.nan
    mse   = mean_squared_error(y_true, y_pred)
    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = np.sqrt(mse)
    return {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}


def walk_forward_gru(df_model: pd.DataFrame,
                     feature_cols: list,
                     target_cols: list,
                     best_params: dict,
                     forecast_horizon: int = 15,
                     retrain_window: int = 10,
                     mape_threshold: float = 2.0):
    """
    Walk-forward forecasting with a tuned GRU model.
    Returns (preds_df, actuals_df, metrics_df, trained_model).
    """
    # --- Split and scale ---
    train_df = df_model.iloc[:-forecast_horizon]
    test_df  = df_model.iloc[-forecast_horizon:]

    f_scaler = RobustScaler()
    t_scaler = RobustScaler()

    X_train = f_scaler.fit_transform(train_df[feature_cols])
    y_train = t_scaler.fit_transform(train_df[target_cols])
    X_test  = f_scaler.transform(test_df[feature_cols])
    y_test  = t_scaler.transform(test_df[target_cols])

    # --- Initial training sequences ---
    window = best_params['window_size']
    X_tr_seq, y_tr_seq = create_sequences(X_train, y_train, window)

    model = create_gru_model(
        input_shape=(window, len(feature_cols)),
        output_dim=len(target_cols),
        **best_params
    )
    from tensorflow.keras.callbacks import EarlyStopping
    es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(
        X_tr_seq, y_tr_seq,
        epochs=best_params.get('epochs', 30),
        batch_size=best_params.get('batch_size', 32),
        callbacks=[es],
        verbose=0
    )

    # --- Walk-forward loop ---
    preds = {c: [] for c in target_cols}
    acts  = {c: [] for c in target_cols}

    for i in tqdm(range(forecast_horizon),
                  desc="Walk-Forward GRU"):
        # build sequence
        if i < window:
            seq = np.vstack([
                X_train[-(window - i):],
                X_test[:i]
            ])
        else:
            seq = X_test[i-window:i]
        seq = seq.reshape(1, window, -1)
        # predict
        y_pred_scaled = model.predict(seq)
        y_pred = t_scaler.inverse_transform(y_pred_scaled)[0]
        y_true = test_df[target_cols].iloc[i].values
        # store
        for idx, col in enumerate(target_cols):
            preds[col].append(y_pred[idx])
            acts[col].append(y_true[idx])
        # update training set
        new_X = X_test[i].reshape(1, -1)
        new_y = y_test[i].reshape(1, -1)
        X_train = np.vstack([X_train, new_X])
        y_train = np.vstack([y_train, new_y])
        # retrain if needed
        if i >= retrain_window:
            recent_pred = np.array(preds[target_cols[0]][-retrain_window:])
            recent_act  = np.array(acts[target_cols[0]][-retrain_window:])
            recent_mape = mean_absolute_percentage_error(recent_act,
                                                         recent_pred) * 100
            if recent_mape > mape_threshold:
                model.fit(
                    *create_sequences(X_train, y_train, window),
                    epochs=best_params.get('epochs', 30),
                    batch_size=best_params.get('batch_size', 32),
                    callbacks=[es],
                    verbose=0
                )

    # --- Build DataFrames & metrics ---
    preds_df = pd.DataFrame(preds)
    acts_df  = pd.DataFrame(acts)
    metrics = {
        c: compute_metrics(acts_df[c].values, preds_df[c].values)
        for c in target_cols
    }
    metrics_df = pd.DataFrame(metrics).T
    return preds_df, acts_df, metrics_df, model


def walk_forward_rf_one_day(df_model: pd.DataFrame,
                            feature_cols: list,
                            target_cols: list,
                            best_params: dict,
                            horizon_days: int = 15,
                            retrain_window: int = 5,
                            mape_threshold: float = 1.0):
    """
    Walk-forward forecasting with a Random Forest model, one day at a time.
    Returns (preds_df, actuals_df, metrics_df, trained_rf).
    """
    df = df_model.reset_index(drop=True).copy()
    cutoff = len(df) - horizon_days
    train_df = df.iloc[:cutoff]
    test_df  = df.iloc[cutoff:]

    f_scaler = RobustScaler()
    t_scaler = RobustScaler()
    X_train  = f_scaler.fit_transform(train_df[feature_cols].values)
    y_train  = t_scaler.fit_transform(train_df[target_cols].values)

    rf = RandomForestRegressor(
        n_estimators=best_params.get("n_estimators", 100),
        max_depth=best_params.get("max_depth", 10),
        min_samples_split=best_params.get("min_samples_split", 2),
        min_samples_leaf=best_params.get("min_samples_leaf", 1),
        max_features=best_params.get("max_features", "sqrt"),
        random_state=42
    )
    rf.fit(X_train, y_train)

    preds = {c: [] for c in target_cols}
    acts  = {c: [] for c in target_cols}

    for i in tqdm(range(horizon_days),
                  desc="Walk-Forward RF"):
        X_row = test_df[feature_cols].iloc[i].values.reshape(1, -1)
        y_row = test_df[target_cols].iloc[i].values.reshape(1, -1)

        X_scaled = f_scaler.transform(X_row)
        y_pred_scaled = rf.predict(X_scaled)
        y_pred = t_scaler.inverse_transform(y_pred_scaled)[0]

        for idx, col in enumerate(target_cols):
            preds[col].append(y_pred[idx])
            acts[col].append(y_row[0, idx])

        # update training
        X_new_scaled = X_scaled
        y_new_scaled = t_scaler.transform(y_row)
        X_train = np.vstack([X_train, X_new_scaled])
        y_train = np.vstack([y_train, y_new_scaled])

        # retrain if needed
        if i >= retrain_window:
            recent_mape = np.mean([
                mean_absolute_percentage_error(
                    np.array(acts[c][-retrain_window:]),
                    np.array(preds[c][-retrain_window:])
                ) for c in target_cols
            ]) * 100
            if recent_mape > mape_threshold:
                rf.fit(X_train, y_train)

    preds_df = pd.DataFrame(preds)
    acts_df  = pd.DataFrame(acts)
    metrics = {
        c: compute_metrics(acts_df[c].values, preds_df[c].values)
        for c in target_cols
    }
    metrics_df = pd.DataFrame(metrics).T
    return preds_df, acts_df, metrics_df, rf
