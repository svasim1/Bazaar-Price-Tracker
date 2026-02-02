import optuna
import json
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import requests
import warnings
from event_utils import add_skyblock_time_features
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from data_utils import load_or_fetch_item_data, parse_timestamp
from mayor_utils import get_mayor_perks, match_mayor_perks
from datetime import datetime, timedelta, timezone
import os
warnings.filterwarnings("ignore")



def clean_dump(obj, path):
    tmp = path + ".tmp"
    joblib.dump(obj, tmp)

    with open(tmp, "rb") as f:
        os.fsync(f.fileno())

    os.replace(tmp, path)

def tukey_clip(y, k=3):
    q1 = np.percentile(y, 25)
    q3 = np.percentile(y, 75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return np.clip(y, lower, upper)

# =========================================================
# Feature Engineering
# =========================================================

def add_time_features(df, ts_col='timestamp'):
    dt = pd.to_datetime(df[ts_col])
    df['hour'] = dt.dt.hour
    df['minute'] = dt.dt.minute
    df['dayofweek'] = dt.dt.dayofweek

    df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
    df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)
    df['dow_sin'] = np.sin(2*np.pi*df['dayofweek']/7)
    df['dow_cos'] = np.cos(2*np.pi*df['dayofweek']/7)

    df['delta_minutes'] = df['timestamp'].diff().dt.total_seconds().fillna(0) / 60
    return df


def build_lagged_features(
    df,
    price_col='buy_price',
    vol_col='buy_volume',
    lags=(1, 2, 3, 6, 12),
    prefix=''
):
    ret = df[price_col].pct_change()
    df[f'{prefix}ret'] = ret

    for lag in lags:
        df[f'{prefix}ret_lag_{lag}'] = ret.shift(lag)
        df[f'{prefix}price_lag_{lag}'] = df[price_col].shift(lag)
        df[f'{prefix}vol_lag_{lag}'] = df[vol_col].shift(lag)

    df[f'{prefix}roll_mean_6'] = ret.rolling(6).mean()
    df[f'{prefix}roll_std_6'] = ret.rolling(6).std()
    df[f'{prefix}mom_6'] = ret.rolling(6).sum()

    return df

def prepare_dataframe_from_raw(data, mayor_data=None):
    rows = []

    for entry in data:
        try:
            ts = parse_timestamp(entry['timestamp'])
        except:
            continue

        def f(k):
            try:
                return float(entry.get(k, 0))
            except:
                return 0.0

        row = {
            'timestamp': ts,
            'buy_price': f('buy'),
            'sell_price': f('sell'),
            'buy_volume': f('buyVolume'),
            'sell_volume': f('sellVolume'),
            'max_buy': f('maxBuy'),
            'min_buy': f('minBuy'),
        }

        if mayor_data is not None:
            perks = match_mayor_perks(ts, mayor_data)
            for i, v in enumerate(perks):
                row[f'mayor_{i}'] = v

        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values('timestamp').reset_index(drop=True)
    df = add_time_features(df)
    df = build_lagged_features(df, price_col='buy_price', vol_col='buy_volume', prefix='buy_')
    df = add_skyblock_time_features(df, ts_col='timestamp')
    return df


# =========================================================
# ENTRY-ONLY LABELING (Regression Target)
# =========================================================

def build_entry_targets(df, horizon_minutes=1440, tax=0.0125):
    df = df.copy().sort_values('timestamp').reset_index(drop=True)
    
    timestamps = pd.to_datetime(df['timestamp']).astype('int64') // 10**9
    horizon_sec = horizon_minutes * 60
    
    # compute immediate future return
    future_return = (df['buy_price'].values * (1 - tax) - df['sell_price'].values) / (df['sell_price'].values + 1e-9)
    df['future_return'] = future_return
    
    # initialize lists to store features
    future_max_return = []
    first_event_good = []
    time_to_max = []
    time_to_min = []
    
    timestamps_np = timestamps.values if isinstance(timestamps, pd.Series) else timestamps
    
    # loop over each row
    for i in range(len(df)):
        # select rows within the horizon
        horizon_mask = (timestamps_np >= timestamps_np[i]) & (timestamps_np - timestamps_np[i] <= horizon_sec)
        idxs_horizon = np.where(horizon_mask)[0]
        returns_horizon = future_return[idxs_horizon]
        
        if len(returns_horizon) == 0:
            future_max_return.append(0.0)
            first_event_good.append(0)
            time_to_max.append(0)
            time_to_min.append(0)
            continue
        
        # find max/min and their absolute indices
        t_max_rel = np.argmax(returns_horizon)
        t_min_rel = np.argmin(returns_horizon)
        t_max = idxs_horizon[t_max_rel]
        t_min = idxs_horizon[t_min_rel]
        
        best_return = returns_horizon[t_max_rel]
        worst_return = returns_horizon[t_min_rel]
        
        # risk-aware heuristic
        if abs(worst_return) > best_return:
            label = worst_return
        elif t_max < t_min:
            label = best_return
        else:
            label = worst_return
        
        future_max_return.append(label)
        first_event_good.append(int(t_max < t_min))
        time_to_max.append(timestamps_np[t_max] - timestamps_np[i])
        time_to_min.append(timestamps_np[t_min] - timestamps_np[i])
    
    # assign features to DataFrame
    df['future_max_return'] = future_max_return
    df['first_event_good'] = first_event_good
    df['time_to_max'] = time_to_max
    df['time_to_min'] = time_to_min
    df['profitable_after_tax'] = (df['future_max_return'] > tax).astype(int)
    
    return df

# =========================================================
# Cleaning
# =========================================================

def clean_infinite_values(X):
    X = np.asarray(X, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=1e8, neginf=-1e8)
    return np.clip(X, -1e8, 1e8)

# =========================================================
# Quantile Loss
# =========================================================

def quantile_loss(y_true, y_pred, alpha):
    diff = y_true - y_pred
    return np.mean(np.maximum(alpha * diff, (alpha - 1) * diff))

# =========================================================
# Optuna Objective (Entry Regression)
# =========================================================

def entry_objective(trial, X, y, alpha=1e-4):
    params = {
        "objective": "quantile",
        "alpha": alpha,
        "learning_rate": trial.suggest_float("lr", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 64),
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 3,
        "verbosity": -1
    }

    dtrain = lgb.Dataset(X, label=y)
    model = lgb.train(params, dtrain, num_boost_round=300)

    preds = model.predict(X)

    loss = quantile_loss(y, preds, alpha)
    return -loss


# =========================================================
# Percent Error Stats
# =========================================================

def percent_error_stats(y_true, y_pred, eps=1e-9):
    pct_err = (y_pred - y_true) / (np.abs(y_true) + eps)

    stats = {
        "min_pct_error": np.min(pct_err),
        "max_pct_error": np.max(pct_err),
        "median_pct_error": np.median(pct_err),
        "mean_pct_error": np.mean(pct_err),
        "mean_abs_pct_error": np.mean(np.abs(pct_err)),
        "median_abs_pct_error": np.median(np.abs(pct_err)),
    }

    return stats



# =========================================================
# Training
# =========================================================

def train_model_system(item_id):
    mayor_data = get_mayor_perks()
    data = load_or_fetch_item_data(item_id)
    
    df = prepare_dataframe_from_raw(data, mayor_data)
    if len(df) < 50:
        print(f"{item_id}: not enough data")
        return

    df = build_entry_targets(df)
    
    exclude = {'timestamp', 'future_max_return'}
    feature_cols = [c for c in df.columns if c not in exclude]

    X = clean_infinite_values(df[feature_cols].values)
    y = df['future_max_return'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    y = tukey_clip(y, k=3)




    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: entry_objective(t, X_scaled, y), n_trials=30)

    params = study.best_params
    params.update({
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1
    })

    model = lgb.train(params, lgb.Dataset(X_scaled, label=y), num_boost_round=400)
    model_dir = os.path.expanduser("~/Model_Files") 
    os.makedirs(model_dir, exist_ok=True)
    
    base = os.path.join(model_dir, str(item_id))
    

    try:
        clean_dump(model, base + "_entry_model.pkl")
        clean_dump(scaler, base + "_entry_scaler.pkl")
        clean_dump(feature_cols, base + "_entry_features.pkl")
        print(f"✓ Successfully saved models for {item_id}")
    except Exception as e:
        print(f"✗ Error saving files for {item_id}: {e}")

# =========================================================
# Test Train Setup for Model Accuracy Metrics
# =========================================================


def test_train_model_system(item_id):
    mayor_data = get_mayor_perks()
    data = load_or_fetch_item_data(item_id)

    df = prepare_dataframe_from_raw(data, mayor_data)
    if len(df) < 50:
        print(f"{item_id}: not enough data")
        return

    df = build_entry_targets(df)
    split_idx = int(len(df) * 0.8)  # 80% for training, 20% for validation
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    exclude = {'timestamp', 'future_max_return'}
    feature_cols = [c for c in df.columns if c not in exclude]

    X_train = clean_infinite_values(train_df[feature_cols].values)
    y_train = train_df['future_max_return'].values

    X_val = clean_infinite_values(val_df[feature_cols].values)
    y_val = val_df['future_max_return'].values

    y_val = tukey_clip(y_val, k=3)
    y_train = tukey_clip(y_train, k=3)


    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    print("y_train: min =", y_train.min(), 
      "median =", np.median(y_train), 
      "max =", y_train.max())
    print("y_val: min =", y_val.min(), 
        "median =", np.median(y_val), 
        "max =", y_val.max())
    pos_train = np.sum(y_train > 0)
    neg_train = np.sum(y_train < 0)
    zero_train = np.sum(y_train == 0)



    print(f"y_train: positive={pos_train}, negative={neg_train}, zero={zero_train}")

    # Same for y_val
    pos_val = np.sum(y_val > 0)
    neg_val = np.sum(y_val < 0)
    zero_val = np.sum(y_val == 0)

    print(f"y_val: positive={pos_val}, negative={neg_val}, zero={zero_val}")
    print(f"y_train: {pos_train/len(y_train)*100:.1f}% positive, {neg_train/len(y_train)*100:.1f}% negative")
    print(f"y_val: {pos_val/len(y_val)*100:.1f}% positive, {neg_val/len(y_val)*100:.1f}% negative")


    print("X_val NaNs:", np.isnan(X_val).sum())
    print("X_val infs:", np.isinf(X_val).sum())




    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: entry_objective(t, X_train_scaled, y_train), n_trials=30)

    params = study.best_params
    params.update({
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1
    })

    model = lgb.train(params, lgb.Dataset(X_train_scaled, label=y_train), num_boost_round=400)

    y_pred = model.predict(X_val_scaled)
    rmse = np.sqrt(np.mean((y_pred - y_val)**2))
    mae = np.mean(np.abs(y_pred - y_val))
    r2 = r2_score(y_val, y_pred)  # compute manually
    y_mean = np.full_like(y_val, y_train.mean())
    baseline_r2 = r2_score(y_val, y_mean)
    print("Baseline R^2:", baseline_r2)
    print(f"RMSE: {rmse}, MAE: {mae}, R^2: {r2}")
    pred_sign = np.sign(model.predict(X_val))
    true_sign = np.sign(y_val)
    accuracy = np.mean(pred_sign == true_sign)
    print("Sign accuracy:", accuracy)
    stats = percent_error_stats(y_val, y_pred)

    for k, v in stats.items():
        print(f"{k}: {v*100:.2f}%")

    mask = y_val > 0.1
    safe_sign_acc = np.mean((y_pred[mask] > 0) == (y_val[mask] > 0))
    print("Safe sign accuracy (true positive returns):", safe_sign_acc)

# =========================================================
# FUTURE PREDICTIONS (MULTI-TIMESTAMP)
# =========================================================



def predict_entries(model, scaler, feature_cols, item_id, mayor_data=None, horizon_hours=24, step_minutes=5):
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=horizon_hours)
    start_str = start.strftime("%Y-%m-%dT%H:%M:%S.000").replace(":", "%3A")
    end_str = now.strftime("%Y-%m-%dT%H:%M:%S.000").replace(":", "%3A")
    base_url = "https://sky.coflnet.com/api/bazaar"
    url = f"{base_url}/{item_id}/history?start={start_str}&end={end_str}"
    raw = requests.get(url).json()
    df = prepare_dataframe_from_raw(raw, mayor_data)
    if df.empty:
        return []

    last_row = df.iloc[-1:].copy()

    future_times = pd.date_range(start=now, periods=int(horizon_hours*60/step_minutes), freq=f"{step_minutes}T")

    preds = []
    for ts in future_times:
        row = last_row.copy()
        row['timestamp'] = ts

        if mayor_data is not None:
            perks = match_mayor_perks(ts, mayor_data)
            for i, v in enumerate(perks):
                row[f'mayor_{i}'] = v

        for c in feature_cols:
            if c not in row.columns:
                row[c] = 0.0
        row[feature_cols] = row[feature_cols].fillna(0.0)

        X = clean_infinite_values(row[feature_cols].values)
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)[0]
        row['entry_score'] = y_pred
        row['timestamp'] = ts.isoformat()
        preds.append(row[['timestamp', 'buy_price', 'sell_price', 'entry_score']].to_dict(orient='records')[0])


    return preds


# =========================================================
# ANALYZE TOP PREDICTIONS
# =========================================================

def analyze_entries(pred_list, top_k=5):
    if not pred_list:
        return []

    now = datetime.now(timezone.utc)
    enriched = []

    for e in pred_list:
        try:
            score = float(e.get('entry_score', 0.0))
        except Exception:
            continue

        if score <= 0:
            continue

        ts_str = e.get('timestamp')
        if not ts_str:
            continue

        try:
            ts = datetime.fromisoformat(ts_str)
        except Exception:
            continue

        delta_minutes = (ts - now).total_seconds() / 60.0
        if delta_minutes < 0:
            continue

        enriched_entry = dict(e)
        enriched_entry['delta_minutes'] = float(delta_minutes)
        enriched.append(enriched_entry)


    enriched.sort(key=lambda x: (x['delta_minutes'], -x['entry_score']))

    return enriched[:top_k]





# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    with open("bazaar_full_items_ids.json") as f:
        items = json.load(f)
    for entry in items:
        print(f"Training {entry}")
        train_model_system(entry)


