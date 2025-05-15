import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import optuna
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional
from keras.optimizers.legacy import Adam


def train_xgb_baseline(df: pd.DataFrame,
                       feature_cols: list,
                       target_cols: list,
                       split_count: int = 30,
                       random_state: int = 42,
                       n_estimators: int = 100):
    """
    Train a baseline multi‐output XGBoost regressor.
    Returns (metrics_dict, trained_model).
    """
    # --- Split train/test ---
    train = df.iloc[:-split_count]
    test  = df.iloc[-split_count:]
    # --- Scale features & targets ---
    f_scaler = RobustScaler()
    t_scaler = RobustScaler()
    X_train = f_scaler.fit_transform(train[feature_cols])
    y_train = t_scaler.fit_transform(train[target_cols])
    X_test  = f_scaler.transform(test[feature_cols])
    y_test  = test[target_cols].values
    # --- Train ---
    xgb = MultiOutputRegressor(
        XGBRegressor(objective='reg:squarederror',
                     random_state=random_state,
                     n_estimators=n_estimators)
    )
    xgb.fit(X_train, y_train)
    # --- Predict & inverse‐scale ---
    y_pred_scaled = xgb.predict(X_test)
    y_pred = t_scaler.inverse_transform(y_pred_scaled)
    # --- Metrics ---
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
    metrics = {}
    for i, col in enumerate(target_cols):
        mae  = mean_absolute_error(y_test[:, i], y_pred[:, i])
        mape = mean_absolute_percentage_error(y_test[:, i], y_pred[:, i]) * 100
        metrics[col] = {'MAE': mae, 'MAPE': mape}
    return metrics, xgb


def train_rf_baseline(df: pd.DataFrame,
                      feature_cols: list,
                      target_cols: list,
                      split_count: int = 30,
                      random_state: int = 42,
                      n_estimators: int = 100):
    """
    Train a baseline multi‐output Random Forest regressor.
    Returns (metrics_dict, trained_model).
    """
    train = df.iloc[:-split_count]
    test  = df.iloc[-split_count:]
    f_scaler = RobustScaler()
    t_scaler = RobustScaler()
    X_train = f_scaler.fit_transform(train[feature_cols])
    y_train = t_scaler.fit_transform(train[target_cols])
    X_test  = f_scaler.transform(test[feature_cols])
    y_test  = test[target_cols].values

    rf = MultiOutputRegressor(
        RandomForestRegressor(random_state=random_state,
                              n_estimators=n_estimators)
    )
    rf.fit(X_train, y_train)
    y_pred_scaled = rf.predict(X_test)
    y_pred = t_scaler.inverse_transform(y_pred_scaled)

    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
    metrics = {}
    for i, col in enumerate(target_cols):
        mae  = mean_absolute_error(y_test[:, i], y_pred[:, i])
        mape = mean_absolute_percentage_error(y_test[:, i], y_pred[:, i]) * 100
        metrics[col] = {'MAE': mae, 'MAPE': mape}
    return metrics, rf


def create_sequences(features: np.ndarray,
                     targets: np.ndarray,
                     window_size: int):
    """
    Turn flat features+targets into LSTM/GRU sequences.
    """
    X, y = [], []
    for i in range(window_size, len(features)):
        X.append(features[i-window_size:i])
        y.append(targets[i])
    return np.array(X), np.array(y)


def create_gru_model(input_shape: tuple,
                     output_dim: int,
                     n_units: int,
                     dropout_rate: float,
                     learning_rate: float,
                     num_layers: int):
    """
    Build a Bidirectional GRU model (with optional stacked layers).
    """
    model = Sequential()
    # first layer
    model.add(Bidirectional(
        GRU(n_units, return_sequences=(num_layers > 1)),
        input_shape=input_shape
    ))
    model.add(Dropout(dropout_rate))
    # additional GRU layers if any
    for i in range(num_layers - 1):
        model.add(GRU(n_units,
                      return_sequences=(i < num_layers - 2)))
        model.add(Dropout(dropout_rate))
    model.add(Dense(output_dim))
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='mse')
    return model


def objective_gru(trial,
                  df_model: pd.DataFrame,
                  feature_cols: list,
                  target_cols: list,
                  n_splits: int = 8):
    """
    Optuna objective for GRU hyperparameters (minimize val_loss).
    """
    # hyperparameter search space
    n_units       = trial.suggest_int('n_units', 64, 128, step=32)
    dropout_rate  = trial.suggest_float('dropout_rate', 0.0, 0.5, step=0.1)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)
    num_layers    = trial.suggest_int('num_layers', 1, 2)
    window_size   = trial.suggest_int('window_size', 10, 40, step=10)
    batch_size    = trial.suggest_categorical('batch_size', [32, 64])

    tscv = TimeSeriesSplit(n_splits=n_splits)
    val_losses = []

    for train_idx, val_idx in tscv.split(df_model):
        train_df = df_model.iloc[train_idx]
        val_df   = df_model.iloc[val_idx]

        f_scaler = RobustScaler()
        t_scaler = RobustScaler()
        X_train  = f_scaler.fit_transform(train_df[feature_cols])
        X_val    = f_scaler.transform(val_df[feature_cols])
        y_train  = t_scaler.fit_transform(train_df[target_cols])
        y_val    = t_scaler.transform(val_df[target_cols])

        X_tr_seq, y_tr_seq = create_sequences(X_train, y_train, window_size)
        X_val_seq, y_val_seq = create_sequences(X_val, y_val, window_size)
        if len(X_val_seq) == 0:
            continue

        model = create_gru_model(
            input_shape=(window_size, len(feature_cols)),
            output_dim=len(target_cols),
            n_units=n_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            num_layers=num_layers
        )
        from tensorflow.keras.callbacks import EarlyStopping
        es = EarlyStopping(monitor='val_loss', patience=5,
                           restore_best_weights=True)
        history = model.fit(
            X_tr_seq, y_tr_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=30,
            batch_size=batch_size,
            callbacks=[es],
            verbose=0
        )
        val_losses.append(min(history.history['val_loss']))

    return np.mean(val_losses) if val_losses else float('inf')


def tune_gru_hyperparams(df_model: pd.DataFrame,
                         feature_cols: list,
                         target_cols: list,
                         n_splits: int = 8,
                         n_trials: int = 150):
    """
    Run Optuna to find best GRU hyperparameters.
    Returns (best_params, best_value).
    """
    def _obj(trial):
        return objective_gru(trial, df_model, feature_cols, target_cols, n_splits)

    study = optuna.create_study(direction='minimize')
    study.optimize(_obj, n_trials=n_trials)
    return study.best_params, study.best_value


def objective_rf(trial,
                 df: pd.DataFrame,
                 feature_cols: list,
                 target_cols: list,
                 n_splits: int = 5):
    """
    Optuna objective for RandomForest hyperparameter tuning (minimize MAPE).
    """
    n_estimators     = trial.suggest_int("n_estimators", 100, 400, step=50)
    max_depth        = trial.suggest_int("max_depth", 5, 25, step=5)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf  = trial.suggest_int("min_samples_leaf", 1, 5)
    max_features     = trial.suggest_categorical("max_features", ["sqrt", "log2", None])

    tscv = TimeSeriesSplit(n_splits=n_splits)
    mape_scores = []

    from sklearn.metrics import mean_absolute_percentage_error

    for train_idx, val_idx in tscv.split(df):
        X_train = df.iloc[train_idx][feature_cols].values
        y_train = df.iloc[train_idx][target_cols].values
        X_val   = df.iloc[val_idx][feature_cols].values
        y_val   = df.iloc[val_idx][target_cols].values

        f_scaler = RobustScaler()
        t_scaler = RobustScaler()
        X_tr = f_scaler.fit_transform(X_train)
        y_tr = t_scaler.fit_transform(y_train)
        X_v  = f_scaler.transform(X_val)
        y_v  = t_scaler.transform(y_val)

        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42
        )
        rf.fit(X_tr, y_tr)
        y_pred_scaled = rf.predict(X_v)
        y_pred = t_scaler.inverse_transform(y_pred_scaled)
        y_true = t_scaler.inverse_transform(y_v)

        mape_scores.append(
            mean_absolute_percentage_error(y_true, y_pred)
        )

    return np.mean(mape_scores)


def tune_rf_hyperparams(df_model: pd.DataFrame,
                        feature_cols: list,
                        target_cols: list,
                        n_splits: int = 5,
                        n_trials: int = 200):
    """
    Run Optuna to find best RF hyperparameters.
    Returns (best_params, best_value).
    """
    def _obj(trial):
        return objective_rf(trial, df_model, feature_cols, target_cols, n_splits)

    study = optuna.create_study(direction="minimize")
    study.optimize(_obj, n_trials=n_trials)
    return study.best_params, study.best_value
