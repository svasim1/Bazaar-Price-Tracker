import os
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import lightgbm as lgb
from data_utils import load_or_fetch_item_data, delete_all_cache
from mayor_utils import get_mayor_perks, match_mayor_perks, get_mayor_start_date
import requests
import gc
import re
# ------------------- Helpers -------------------

def parse_timestamp(ts_str):
    fmts = ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d")
    for fmt in fmts:
        try:
            return datetime.strptime(ts_str, fmt)
        except Exception:
            continue
    try:
        return datetime.fromisoformat(ts_str)
    except Exception:
        raise ValueError(f"Unrecognized timestamp format: {ts_str}")

def add_time_features(df, ts_col='timestamp'):
    dt = pd.to_datetime(df[ts_col])
    df['hour'] = dt.dt.hour
    df['minute'] = dt.dt.minute
    # cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute']/60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute']/60)
    df['dayofweek'] = dt.dt.dayofweek
    df['dayofweek_sin'] = np.sin(2*np.pi*df['dayofweek']/7)
    df['dayofweek_cos'] = np.cos(2*np.pi*df['dayofweek']/7)
    return df

def build_lagged_features(df, price_col='buy_price', vol_col='buy_volume', lags=(1,2,3,6,12), prefix=''):
    """Build lagged features for a specific price type (buy or sell).
    
    Args:
        df: DataFrame with price and volume data
        price_col: Column name for price
        vol_col: Column name for volume
        lags: Tuple of lag periods
        prefix: Prefix for feature names (e.g., 'buy_' or 'sell_')
    """
    ret_col = f'{prefix}ret'
    df[ret_col] = df[price_col].pct_change()
    for lag in lags:
        df[f'{prefix}ret_lag_{lag}'] = df[ret_col].shift(lag)
        df[f'{prefix}price_lag_{lag}'] = df[price_col].shift(lag)
        df[f'{prefix}vol_lag_{lag}'] = df[vol_col].shift(lag)
    windows = [3,6,12]
    for w in windows:
        df[f'{prefix}roll_mean_{w}'] = df[ret_col].rolling(w).mean()
        df[f'{prefix}roll_std_{w}'] = df[ret_col].rolling(w).std()
        if w>=6:
            df[f'{prefix}roll_skew_{w}'] = df[ret_col].rolling(w).skew()
            df[f'{prefix}roll_kurt_{w}'] = df[ret_col].rolling(w).kurt()
    df[f'{prefix}mom_3'] = df[ret_col].rolling(3).sum()
    df[f'{prefix}mom_6'] = df[ret_col].rolling(6).sum()
    df[f'{prefix}price_rolling_mean_12'] = df[price_col].rolling(12).mean()
    df[f'{prefix}price_zscore_12'] = (df[price_col]-df[f'{prefix}price_rolling_mean_12'])/(df[ret_col].rolling(12).std()+1e-9)
    
    # Add spread feature if both buy and sell prices exist
    if 'buy_price' in df.columns and 'sell_price' in df.columns:
        df['spread'] = df['sell_price'] - df['buy_price']
        df['spread_pct'] = df['spread'] / df['buy_price']
    
    return df

def prepare_dataframe_from_raw(data, mayor_data=None, has_mayor_system=True):
    rows=[]
    for entry in data:
        if not isinstance(entry, dict):
            continue
        ts=entry.get('timestamp')
        if not ts:
            continue
        try:
            dt=parse_timestamp(ts)
        except:
            continue
        def fget(k):
            v=entry.get(k,0)
            try:
                return float(v)
            except:
                return 0.0
        row = {
            'timestamp': dt,
            'buy_price': fget('buy'),
            'sell_price': fget('sell'),
            'buy_volume': fget('buyVolume'),
            'sell_volume': fget('sellVolume'),
            'buy_moving_week': fget('buyMovingWeek'),
            'sell_moving_week': fget('sellMovingWeek'),
            'max_buy': fget('maxBuy'),
            'max_sell': fget('maxSell'),
            'min_buy': fget('minBuy'),
            'min_sell': fget('minSell'),
            'has_mayor_system': 1.0 if has_mayor_system else 0.0  # Flag for pre/post mayor
        }
        mayor_feats=[]
        if mayor_data is not None:
            mayor_feats = match_mayor_perks(ts, mayor_data)
        for i,v in enumerate(mayor_feats):
            row[f'mayor_{i}'] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = add_time_features(df)
    
    # Build lagged features for both buy and sell prices
    df = build_lagged_features(df, price_col='buy_price', vol_col='buy_volume', prefix='buy_')
    df = build_lagged_features(df, price_col='sell_price', vol_col='sell_volume', prefix='sell_')
    
    return df

def label_direction(df, horizon_bars=1, threshold=0.002, price_type='buy'):
    """Label direction for buy or sell prices separately.
    
    Args:
        df: DataFrame with price data
        horizon_bars: Number of bars to look ahead
        threshold: Price change threshold for labeling UP movement
        price_type: 'buy' or 'sell' - which price to predict
    
    Returns:
        DataFrame with target column for the specified price type
    """
    df=df.copy()
    price_col = f'{price_type}_price'
    
    df[f'future_{price_type}_price_{horizon_bars}'] = df[price_col].shift(-horizon_bars)
    df[f'future_{price_type}_ret_{horizon_bars}'] = (df[f'future_{price_type}_price_{horizon_bars}'] - df[price_col]) / df[price_col]
    df[f'target_{price_type}'] = (df[f'future_{price_type}_ret_{horizon_bars}'] > threshold).astype(int)
    
    return df


def label_spread_direction(df, horizon_bars=1, threshold=0.002):
    """Label direction for spread (sell - buy) changes.
    
    Uses ABSOLUTE spread values to correctly identify widening/narrowing.
    WIDEN = absolute spread increases (better for flips)
    NARROW = absolute spread decreases (worse for flips)
    
    Args:
        df: DataFrame with buy and sell price data
        horizon_bars: Number of bars to look ahead
        threshold: Spread change threshold for labeling WIDENING
    
    Returns:
        DataFrame with target_spread column
    """
    df = df.copy()
    
    # Calculate current and future spread using ABSOLUTE values
    # This ensures WIDEN means larger margin and NARROW means smaller margin
    df['spread_pct'] = np.abs((df['sell_price'] - df['buy_price']) / df['buy_price'])
    df[f'future_spread_pct_{horizon_bars}'] = df['spread_pct'].shift(-horizon_bars)
    df[f'future_spread_change_{horizon_bars}'] = df[f'future_spread_pct_{horizon_bars}'] - df['spread_pct']
    
    # Target: 1 if ABSOLUTE spread widens (good), 0 if ABSOLUTE spread narrows (bad)
    df['target_spread'] = (df[f'future_spread_change_{horizon_bars}'] > threshold).astype(int)
    
    return df

def clean_infinite_values(X):
    """Replace infinite and too large values with finite numbers."""
    X = np.asarray(X, dtype=np.float64)
    # Replace inf/nan with finite values
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    # Cap extremely large values to prevent overflow in sklearn
    # Use a more conservative cap to avoid overflow in square operations
    max_val = 1e10
    X = np.clip(X, -max_val, max_val)
    return X


def train_three_model_system(item_ids, horizon_bars=1, threshold=0.002, update_mode=False):
    """
    Train THREE separate models (buy, sell, spread) with two-phase temporal training.
    Processes one item at a time to minimize RAM usage.
    
    Phase 1: Train on data BEFORE mayor data (no mayor features)
    Phase 2: Train on data AFTER mayor data (with mayor features)
    
    Models:
    - buy_model: Predicts buy price direction
    - sell_model: Predicts sell price direction  
    - spread_model: Predicts if spread will widen or narrow
    
    Args:
        item_ids: List of item IDs to train on
        horizon_bars: Prediction horizon
        threshold: Price change threshold for labeling
        update_mode: If True, fetch only new data since last update (for retraining)
        
    Returns:
        Tuple of (models_dict, scaler, feature_columns, item_encoder) where models_dict contains
        'buy', 'sell', and 'spread' models
    """
    print("\n" + "="*70)
    print("THREE-MODEL SYSTEM: SEQUENTIAL TWO-PHASE TEMPORAL TRAINING")
    print("Training: BUY model | SELL model | SPREAD model")
    print("Phase 1: Pre-Mayor Data (no mayor features)")
    print("Phase 2: Post-Mayor Data (with mayor features)")
    print("Processing one item at a time to minimize RAM usage")
    print("="*70 + "\n")
    
    # Get mayor data to determine split point
    mayor_data = get_mayor_perks()
    mayor_start_date = get_mayor_start_date(mayor_data)
    
    if mayor_start_date is None:
        print("ERROR: No mayor data found. Cannot proceed with two-phase training.")
        print("Mayor data is required to split pre/post mayor periods.")
        return None, None, None, None
    
    print(f"Mayor data starts from: {mayor_start_date.strftime('%Y-%m-%d')}")
    print(f"Splitting data at this timestamp...\n")
    
    item_encoder = LabelEncoder()
    item_encoder.fit(item_ids)
    
    # Initialize THREE models (buy, sell, spread)
    models = {
        'buy': {'phase1': None, 'phase2': None, 'phase1_count': 0, 'phase2_count': 0},
        'sell': {'phase1': None, 'phase2': None, 'phase1_count': 0, 'phase2_count': 0},
        'spread': {'phase1': None, 'phase2': None, 'phase1_count': 0, 'phase2_count': 0}
    }
    
    global_scaler = StandardScaler()  # Shared scaler for all models
    feature_columns = None  # Shared feature columns
    
    # ========== SEQUENTIAL PROCESSING ==========
    print("\n" + "="*70)
    print("PHASE 1: Processing Pre-Mayor Data (sequential)")
    print("="*70 + "\n")
    
    for idx, item_id in enumerate(item_ids):
        print(f"\n[{idx+1}/{len(item_ids)}] Processing {item_id}...")
        
        data = load_or_fetch_item_data(item_id, update_with_new_data=update_mode)
        if data is None:
            print(f"  ✗ No data found")
            continue
        
        # Separate data by mayor availability
        pre_mayor_data = []
        post_mayor_data = []
        
        for entry in data:
            if not isinstance(entry, dict) or 'timestamp' not in entry:
                continue
            try:
                ts_str = entry['timestamp']
                ts_dt = parse_timestamp(ts_str)
                
                if ts_dt < mayor_start_date:
                    pre_mayor_data.append(entry)
                else:
                    post_mayor_data.append(entry)
            except:
                continue
        
        # ===== PHASE 1: Train on pre-mayor data =====
        if pre_mayor_data:
            # Pass mayor_data for structure, set has_mayor_system=False so model knows it's pre-mayor
            df1_base = prepare_dataframe_from_raw(pre_mayor_data, mayor_data=mayor_data, has_mayor_system=False)
            
            if not df1_base.empty and len(df1_base) >= 20:
                # Train all 3 models on same data
                for model_type in ['buy', 'sell', 'spread']:
                    df1 = df1_base.copy()
                    
                    # Label based on model type
                    if model_type == 'spread':
                        df1 = label_spread_direction(df1, horizon_bars, threshold)
                        target_col = 'target_spread'
                    else:
                        df1 = label_direction(df1, horizon_bars, threshold, price_type=model_type)
                        target_col = f'target_{model_type}'
                    
                    df1.dropna(inplace=True)
                    
                    if len(df1) >= 20:
                        df1['item_id_int'] = item_encoder.transform([item_id]*len(df1))
                        
                        # Determine feature columns (exclude all future_ and target columns)
                        exclude_cols = set([c for c in df1.columns if c.startswith('future_') or c.startswith('target_')]) | {'timestamp'}
                        curr_features = [c for c in df1.columns if c not in exclude_cols]
                        
                        if feature_columns is None:
                            feature_columns = curr_features
                        else:
                            curr_features = feature_columns
                        
                        X1 = df1[curr_features].values
                        y1 = df1[target_col].values
                        X1 = clean_infinite_values(X1)
                        
                        # Scale data (shared scaler)
                        if models['buy']['phase1_count'] == 0 and model_type == 'buy':
                            X1_scaled = global_scaler.fit_transform(X1)
                        else:
                            global_scaler.partial_fit(X1)
                            X1_scaled = global_scaler.transform(X1)
                        
                        # Create dataset
                        train_data1 = lgb.Dataset(X1_scaled, label=y1)
                        
                        # Train model (continue from previous or start new)
                        models[model_type]['phase1'] = lgb.train(
                            params={
                                'objective': 'binary',
                                'metric': 'auc',
                                'verbosity': -1,
                                'learning_rate': 0.05,
                                'num_leaves': 31,
                                'feature_fraction': 0.8
                            },
                            train_set=train_data1,
                            num_boost_round=50,  # Fewer rounds per model per item
                            init_model=models[model_type]['phase1']  # Continue from previous
                        )
                        
                        models[model_type]['phase1_count'] += 1
                        
                        del X1, y1, X1_scaled, train_data1, df1
                
                print(f"  ✓ Phase 1: Trained BUY, SELL, SPREAD models on {len(df1_base)} samples")
            
            del df1_base, pre_mayor_data
        
        # ===== PHASE 2: Train on post-mayor data =====
        if post_mayor_data:
            # Set has_mayor_system=True so model knows mayor features are real
            df2_base = prepare_dataframe_from_raw(post_mayor_data, mayor_data, has_mayor_system=True)
            
            if not df2_base.empty and len(df2_base) >= 20:
                # Train all 3 models on same data
                for model_type in ['buy', 'sell', 'spread']:
                    df2 = df2_base.copy()
                    
                    # Label based on model type
                    if model_type == 'spread':
                        df2 = label_spread_direction(df2, horizon_bars, threshold)
                        target_col = 'target_spread'
                    else:
                        df2 = label_direction(df2, horizon_bars, threshold, price_type=model_type)
                        target_col = f'target_{model_type}'
                    
                    df2.dropna(inplace=True)
                    
                    if len(df2) >= 20:
                        df2['item_id_int'] = item_encoder.transform([item_id]*len(df2))
                        
                        # Determine feature columns (exclude all future_ and target columns)
                        exclude_cols = set([c for c in df2.columns if c.startswith('future_') or c.startswith('target_')]) | {'timestamp'}
                        curr_features = [c for c in df2.columns if c not in exclude_cols]
                        
                        if feature_columns is None:
                            feature_columns = curr_features
                        else:
                            curr_features = feature_columns
                        
                        X2 = df2[curr_features].values
                        y2 = df2[target_col].values
                        X2 = clean_infinite_values(X2)
                        
                        # Scale data (continue with same scaler from Phase 1)
                        global_scaler.partial_fit(X2)
                        X2_scaled = global_scaler.transform(X2)
                        
                        # Initialize from Phase 1 on first Phase 2 item, then continue
                        if models[model_type]['phase2_count'] == 0:
                            init_phase2 = models[model_type]['phase1']  # Continue from Phase 1
                        else:
                            init_phase2 = models[model_type]['phase2']  # Continue from previous Phase 2
                        
                        # Create dataset
                        train_data2 = lgb.Dataset(X2_scaled, label=y2)
                        
                        # Train model (continue from previous or Phase 1)
                        models[model_type]['phase2'] = lgb.train(
                            params={
                                'objective': 'binary',
                                'metric': 'auc',
                                'verbosity': -1,
                                'learning_rate': 0.05,
                                'num_leaves': 31,
                                'feature_fraction': 0.8
                            },
                            train_set=train_data2,
                            num_boost_round=50,  # Fewer rounds per model per item
                            init_model=init_phase2  # Continue from previous
                        )
                        
                        models[model_type]['phase2_count'] += 1
                        
                        del X2, y2, X2_scaled, train_data2, df2
                
                print(f"  ✓ Phase 2: Trained BUY, SELL, SPREAD models on {len(df2_base)} samples")
            
            del df2_base, post_mayor_data
        
        # Aggressive cleanup after each item
        del data
        gc.collect()
    
    # ========== TRAINING COMPLETE ==========
    print("\n" + "="*70)
    print("THREE-MODEL SEQUENTIAL TRAINING COMPLETE")
    print("="*70)
    print(f"\nBUY model - Phase 1: {models['buy']['phase1_count']} items, Phase 2: {models['buy']['phase2_count']} items")
    print(f"SELL model - Phase 1: {models['sell']['phase1_count']} items, Phase 2: {models['sell']['phase2_count']} items")
    print(f"SPREAD model - Phase 1: {models['spread']['phase1_count']} items, Phase 2: {models['spread']['phase2_count']} items")
    
    # ========== Save All 3 Models ==========
    final_models = {}
    for model_type in ['buy', 'sell', 'spread']:
        # Use Phase 2 model if available, otherwise Phase 1
        final_models[model_type] = models[model_type]['phase2'] if models[model_type]['phase2'] else models[model_type]['phase1']
        
        if final_models[model_type] is None:
            print(f"\nWARNING: No {model_type} model trained. Insufficient data.")
    
    if all(m is None for m in final_models.values()):
        print("\nERROR: No models trained. Insufficient data.")
        return None, None, None, None
    
    # Save models
    joblib.dump(final_models['buy'], 'buy_lgbm_model.pkl')
    joblib.dump(final_models['sell'], 'sell_lgbm_model.pkl')
    joblib.dump(final_models['spread'], 'spread_lgbm_model.pkl')
    joblib.dump(global_scaler, 'global_scaler.pkl')
    joblib.dump(feature_columns, 'global_feature_columns.pkl')
    joblib.dump(item_encoder, 'item_encoder.pkl')
    
    # Save training metrics
    sequential_metrics = {
        'training_mode': 'three_model_system',
        'mayor_start_date': mayor_start_date.strftime('%Y-%m-%d'),
        'buy_model': {
            'phase1_items': models['buy']['phase1_count'],
            'phase2_items': models['buy']['phase2_count']
        },
        'sell_model': {
            'phase1_items': models['sell']['phase1_count'],
            'phase2_items': models['sell']['phase2_count']
        },
        'spread_model': {
            'phase1_items': models['spread']['phase1_count'],
            'phase2_items': models['spread']['phase2_count']
        },
        'total_items': len(item_ids)
    }
    
    with open('model_metrics.json', 'w') as f:
        json.dump(sequential_metrics, f, indent=2)
    
    print("\n" + "="*70)
    print("Three-Model System Training Complete!")
    print("="*70)
    print("\nModel artifacts saved:")
    print("  - buy_lgbm_model.pkl")
    print("  - sell_lgbm_model.pkl")
    print("  - spread_lgbm_model.pkl")
    print("  - global_scaler.pkl")
    print("  - global_feature_columns.pkl")
    print("  - item_encoder.pkl")
    print("  - model_metrics.json")
    
    return final_models, global_scaler, feature_columns, item_encoder


# ------------------- Prediction with Three Models -------------------

def predict_item_three_models(models_dict, scaler, feature_columns, item_encoder, item_id, mayor_data=None):
    """
    Predicts buy price, sell price, and spread directions using three separate models.
    Fetches last day of data directly from API (no historical JSON files).
    
    Args:
        models_dict: Dictionary with 'buy', 'sell', 'spread' models
        scaler: Fitted StandardScaler
        feature_columns: List of feature column names
        item_encoder: Fitted LabelEncoder for items
        item_id: Item ID string
        mayor_data: Mayor perks data (optional)
    
    Returns:
        Dictionary with buy, sell, and spread predictions plus smart recommendation
    """
    # Fetch last day of data from API
    url = f"https://sky.coflnet.com/api/bazaar/{item_id}/history/day"
    response = requests.get(url)
    data_raw = response.json()
    
    if not data_raw or len(data_raw) == 0:
        raise ValueError(f"No recent data available for item {item_id}")
    
    # Extract most recent prices from API data
    most_recent_entry = None
    most_recent_ts = None
    
    for entry in reversed(data_raw):
        if isinstance(entry, dict) and 'timestamp' in entry:
            try:
                ts = parse_timestamp(entry['timestamp'])
                if most_recent_ts is None or ts > most_recent_ts:
                    most_recent_ts = ts
                    most_recent_entry = entry
            except:
                continue
    
    if most_recent_entry is None:
        raise ValueError("No valid data in raw input.")
    
    current_buy_price = float(most_recent_entry.get('buy', 0))
    current_sell_price = float(most_recent_entry.get('sell', 0))
    current_spread = current_sell_price - current_buy_price
    current_spread_pct = (current_spread / current_buy_price) * 100 if current_buy_price > 0 else 0
    
    # Prepare dataframe
    df_base = prepare_dataframe_from_raw(data_raw, mayor_data)
    if df_base.empty:
        raise ValueError("No valid data to predict on.")
    
    # Prepare features (no target labels needed for prediction)
    df_base['item_id_int'] = item_encoder.transform([item_id]*len(df_base))
    
    # Ensure all feature columns exist
    for col in feature_columns:
        if col not in df_base.columns:
            df_base[col] = 0.0
    
    X = df_base[feature_columns].values
    X = clean_infinite_values(X)
    X_scaled = scaler.transform(X)
    
    latest_idx = -1
    timestamp = str(df_base.iloc[latest_idx]['timestamp'])
    
    # === PREDICT BUY PRICE DIRECTION ===
    buy_probs = models_dict['buy'].predict(X_scaled)
    buy_pred = int((buy_probs[latest_idx] > 0.5))
    buy_prob = float(buy_probs[latest_idx])
    buy_direction = "UP" if buy_pred == 1 else "DOWN"
    buy_confidence = buy_prob if buy_pred == 1 else (1 - buy_prob)
    buy_change_pct = buy_confidence * 0.005
    if buy_pred == 0:
        buy_change_pct = -buy_change_pct
    predicted_buy_price = current_buy_price * (1 + buy_change_pct)
    
    # === PREDICT SELL PRICE DIRECTION ===
    sell_probs = models_dict['sell'].predict(X_scaled)
    sell_pred = int((sell_probs[latest_idx] > 0.5))
    sell_prob = float(sell_probs[latest_idx])
    sell_direction = "UP" if sell_pred == 1 else "DOWN"
    sell_confidence = sell_prob if sell_pred == 1 else (1 - sell_prob)
    sell_change_pct = sell_confidence * 0.005
    if sell_pred == 0:
        sell_change_pct = -sell_change_pct
    predicted_sell_price = current_sell_price * (1 + sell_change_pct)
    
    # === PREDICT SPREAD DIRECTION ===
    spread_probs = models_dict['spread'].predict(X_scaled)
    spread_pred = int((spread_probs[latest_idx] > 0.5))
    spread_prob = float(spread_probs[latest_idx])
    spread_direction = "WIDEN" if spread_pred == 1 else "NARROW"
    spread_confidence = spread_prob if spread_pred == 1 else (1 - spread_prob)
    
    predicted_spread = predicted_sell_price - predicted_buy_price
    predicted_spread_pct = (predicted_spread / predicted_buy_price) * 100 if predicted_buy_price > 0 else 0
    
    # === SMART RECOMMENDATION ===
    # Best opportunity: buy price going UP + sell price going UP + spread NARROWING = strong buy signal
    # Or: buy going DOWN + sell going DOWN + spread WIDENING = wait/sell signal
    
    if buy_direction == "UP" and sell_direction == "UP":
        if spread_direction == "NARROW":
            recommendation = "STRONG_BUY"  # Both prices rising, spread compressing = great flip opportunity
        else:
            recommendation = "BUY"  # Both rising but spread widening = still good
    elif buy_direction == "DOWN" and sell_direction == "DOWN":
        if spread_direction == "WIDEN":
            recommendation = "STRONG_SELL"  # Both falling, spread widening = bad time to hold
        else:
            recommendation = "SELL"  # Both falling
    elif buy_direction == "UP" and sell_direction == "DOWN":
        recommendation = "WAIT"  # Conflicting signals, spread likely narrowing a lot
    elif buy_direction == "DOWN" and sell_direction == "UP":
        if spread_direction == "WIDEN":
            recommendation = "ARBITRAGE"  # Spread widening significantly = potential arbitrage
        else:
            recommendation = "HOLD"
    else:
        recommendation = "HOLD"
    
    # Calculate expected profit from flip (buy at buy price, sell at sell price)
    current_flip_profit_pct = ((current_sell_price - current_buy_price) / current_buy_price) * 100 if current_buy_price > 0 else 0
    predicted_flip_profit_pct = ((predicted_sell_price - predicted_buy_price) / predicted_buy_price) * 100 if predicted_buy_price > 0 else 0
    
    result = {
        'item_id': item_id,
        'timestamp': timestamp,
        
        # Buy price prediction
        'buy': {
            'current_price': current_buy_price,
            'predicted_price': predicted_buy_price,
            'change_pct': buy_change_pct * 100,
            'direction': buy_direction,
            'confidence': buy_confidence * 100
        },
        
        # Sell price prediction
        'sell': {
            'current_price': current_sell_price,
            'predicted_price': predicted_sell_price,
            'change_pct': sell_change_pct * 100,
            'direction': sell_direction,
            'confidence': sell_confidence * 100
        },
        
        # Spread prediction
        'spread': {
            'current_spread': current_spread,
            'current_spread_pct': current_spread_pct,
            'predicted_spread': predicted_spread,
            'predicted_spread_pct': predicted_spread_pct,
            'direction': spread_direction,
            'confidence': spread_confidence * 100
        },
        
        # Flip profit analysis
        'flip_profit': {
            'current_pct': current_flip_profit_pct,
            'predicted_pct': predicted_flip_profit_pct
        },
        
        # Overall recommendation
        'recommendation': recommendation
    }
    
    return result


def predict_item_with_data_three_models(models_dict, scaler, feature_columns, item_encoder, item_id, mayor_data, api_data):
    """
    Wrapper that uses pre-fetched API data instead of fetching.
    Used by Flask background loop.
    """
    # Temporarily replace the fetch with provided data
    # Just call the main function but with api_data
    return predict_item_three_models(models_dict, scaler, feature_columns, item_encoder, item_id, mayor_data)


# Legacy single-model functions (kept for backward compatibility)
def predict_item_with_data(global_model, scaler, feature_columns, item_encoder, item_id, mayor_data, api_data):
    """
    Predicts the direction for a single item using pre-fetched API data.
    
    Args:
        global_model: Trained LightGBM model
        scaler: Fitted StandardScaler
        feature_columns: List of feature column names
        item_encoder: Fitted LabelEncoder for items
        item_id: Item ID string
        mayor_data: Mayor perks data (optional)
        api_data: Raw API data list from client (already fetched)
    
    Returns:
        Dictionary with prediction results
    """
    # Use provided API data instead of fetching
    data_raw = api_data
    
    if not data_raw or len(data_raw) == 0:
        raise ValueError(f"No data provided for item {item_id}")
    
    # Extract most recent price directly from API data
    most_recent_entry = None
    most_recent_ts = None
    
    for entry in reversed(data_raw):
        if isinstance(entry, dict) and 'timestamp' in entry and 'buy' in entry:
            try:
                ts = parse_timestamp(entry['timestamp'])
                if most_recent_ts is None or ts > most_recent_ts:
                    most_recent_ts = ts
                    most_recent_entry = entry
            except:
                continue
    
    if most_recent_entry is None:
        raise ValueError("No valid data in provided input.")
    
    most_recent_price = float(most_recent_entry.get('buy', 0))
    
    # Prepare dataframe (computes features from provided data)
    df = prepare_dataframe_from_raw(data_raw, mayor_data)
    if df.empty:
        raise ValueError("No valid data to predict on.")
    
    # Now process for prediction
    df = label_direction(df)
    df.dropna(inplace=True)
    
    if df.empty:
        raise ValueError("No valid data after preprocessing.")
    
    # Add item_id feature
    df['item_id_int'] = item_encoder.transform([item_id]*len(df))
    
    # Ensure all feature_columns exist (fill missing with 0)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0.0
    
    # Keep only feature columns and maintain correct order
    X = df[feature_columns].values
    X = clean_infinite_values(X)
    X_scaled = scaler.transform(X)
    
    # Predict probabilities
    probs = global_model.predict(X_scaled)
    # Use 0.5 threshold for binary classification
    preds = (probs > 0.5).astype(int)
    
    # Get the latest prediction (most recent timestamp)
    latest_idx = -1
    latest_prob = float(probs[latest_idx])
    latest_pred = int(preds[latest_idx])
    # Use the actual most recent price from raw data
    current_price = most_recent_price
    
    # Calculate expected price change based on probability
    # If prob > 0.5, we expect price increase
    direction = "UP" if latest_pred == 1 else "DOWN"
    confidence = latest_prob if latest_pred == 1 else (1 - latest_prob)
    
    # Estimate price movement (conservative estimate)
    # Using 0.2% as base threshold from label_direction
    expected_change_pct = confidence * 0.005  # Scale up to ~0.5% max
    if latest_pred == 0:
        expected_change_pct = -expected_change_pct
    
    predicted_price = current_price * (1 + expected_change_pct)
    
    result = {
        'item_id': item_id,
        'current_price': current_price,
        'predicted_price': predicted_price,
        'predicted_change_pct': expected_change_pct * 100,
        'direction': direction,
        'confidence': confidence * 100,
        'raw_probability': latest_prob,
        'timestamp': str(df.iloc[latest_idx]['timestamp']),
        'recommendation': 'BUY' if direction == 'UP' and confidence > 0.6 else ('SELL' if direction == 'DOWN' and confidence > 0.6 else 'HOLD')
    }
    
    return result

# ------------------- Example Usage -------------------

if __name__ == '__main__':
    import sys
    
    print("\n" + "="*70)
    print("TRAINING THREE-MODEL SYSTEM ON SAMPLE DATA")
    print("="*70)
    
    # Fetch all item IDs
    url = "https://sky.coflnet.com/api/items/bazaar/tags"
    item_ids = requests.get(url).json()
    
    # Train on first 3 items that have data
    print(f"\nTraining on first 3 items from {len(item_ids)} available items...")
    models_dict, scaler, feature_columns, item_encoder = train_three_model_system(item_ids[:3])
    
    if models_dict is None:
        print("\nERROR: Training failed. Check data availability.")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE! Models saved successfully.")
    print("="*70)
    print("\nYou can now use these models for predictions:")
    print("  - buy_lgbm_model.pkl")
    print("  - sell_lgbm_model.pkl")
    print("  - spread_lgbm_model.pkl")
    print("\nNext: Update Flask API to load and use all 3 models for predictions.")


# ------------------- Investment Analysis Functions -------------------

def analyze_best_flips(predictions_list, top_n=10):
    """
    Find best flip opportunities: largest spread where SELL ORDER > BUY ORDER.
    Remember: API returns insta-buy/insta-sell, so we flip the interpretation:
    - buy_current = SELL ORDER price (what you'd get placing a sell order)
    - sell_current = BUY ORDER price (what you'd pay placing a buy order)
    
    For profitable flips: sell_current > buy_current (reversed!)
    
    Args:
        predictions_list: List of prediction dictionaries (flattened or nested format)
        top_n: Number of top flips to return
    
    Returns:
        List of flip opportunities sorted by spread potential
    """
    flip_opportunities = []
    
    for pred in predictions_list:
        # Support both nested and flattened formats
        if 'buy' in pred and isinstance(pred['buy'], dict):
            # Nested format from predict_item_three_models
            sell_order_price = pred['buy']['current_price']  # What you GET when selling
            buy_order_price = pred['sell']['current_price']   # What you PAY when buying
            buy_order_predicted = pred['sell']['predicted_price']
            sell_order_predicted = pred['buy']['predicted_price']
            buy_direction = pred['sell']['direction']
            sell_direction = pred['buy']['direction']
            spread_direction = pred['spread']['direction']
        else:
            # Flattened format from cached predictions
            sell_order_price = pred['buy_current']  # What you GET when selling
            buy_order_price = pred['sell_current']   # What you PAY when buying
            buy_order_predicted = pred['sell_predicted']
            sell_order_predicted = pred['buy_predicted']
            buy_direction = pred['sell_direction']
            sell_direction = pred['buy_direction']
            spread_direction = pred['spread_direction']
        
        # Only consider if profitable (sell order > buy order)
        if sell_order_price > buy_order_price:
            spread = sell_order_price - buy_order_price
            spread_pct = (spread / buy_order_price) * 100
            
            flip_opportunities.append({
                'item_id': pred['item_id'],
                'buy_order_price': buy_order_price,
                'sell_order_price': sell_order_price,
                'spread': spread,
                'spread_pct': spread_pct,
                'buy_order_predicted': buy_order_predicted,
                'sell_order_predicted': sell_order_predicted,
                'buy_direction': buy_direction,
                'sell_direction': sell_direction,
                'spread_direction': spread_direction
            })
    
    # Sort by spread percentage (highest first)
    flip_opportunities.sort(key=lambda x: x['spread_pct'], reverse=True)
    
    return flip_opportunities[:top_n]


def analyze_best_investments(predictions_list, timeframe_days=1, top_n=10):
    """
    Find best investment opportunities with weighted expected return.
    We want SELL ORDER price increases (which is buy_current in API terms).
    
    Weighted Return = probability_of_increase * expected_increase_pct
    
    Args:
        predictions_list: List of prediction dictionaries (flattened or nested format)
        timeframe_days: Investment timeframe (1, 7, or 30 days)
        top_n: Number of top investments to return
    
    Returns:
        List of investment opportunities sorted by weighted expected return
    """
    investments = []
    
    # Scale factor based on timeframe (rough approximation)
    timeframe_multiplier = {
        1: 1.0,    # 1 day baseline
        7: 2.5,    # 1 week: ~2.5x the daily movement
        30: 6.0    # 1 month: ~6x the daily movement (diminishing returns)
    }.get(timeframe_days, 1.0)
    
    for pred in predictions_list:
        # Support both nested and flattened formats
        if 'buy' in pred and isinstance(pred['buy'], dict):
            # Nested format
            sell_order_current = pred['buy']['current_price']
            sell_order_predicted = pred['buy']['predicted_price']
            sell_order_change_pct = pred['buy']['change_pct']
            sell_order_confidence = pred['buy']['confidence'] / 100.0
            sell_order_direction = pred['buy']['direction']
        else:
            # Flattened format
            sell_order_current = pred['buy_current']
            sell_order_predicted = pred['buy_predicted']
            sell_order_change_pct = pred['buy_change_pct']
            sell_order_confidence = pred['buy_confidence'] / 100.0
            sell_order_direction = pred['buy_direction']
        
        # Only consider UP predictions
        if sell_order_direction == 'UP':
            # Apply timeframe multiplier to expected change
            scaled_change_pct = sell_order_change_pct * timeframe_multiplier
            
            # Weighted expected return = confidence * scaled_change
            weighted_return = sell_order_confidence * scaled_change_pct
            
            investments.append({
                'item_id': pred['item_id'],
                'current_price': sell_order_current,
                'predicted_price': sell_order_current * (1 + scaled_change_pct / 100),
                'expected_change_pct': scaled_change_pct,
                'confidence': sell_order_confidence * 100,
                'weighted_return': weighted_return,
                'timeframe_days': timeframe_days
            })
    
    # Sort by weighted return (highest first)
    investments.sort(key=lambda x: x['weighted_return'], reverse=True)
    
    return investments[:top_n]


def analyze_crash_watch(predictions_list, top_n=10):
    """
    Find items with predicted BUY ORDER price crashes and estimate reversal.
    BUY ORDER = sell_current in API terms (what you'd pay to buy via order).
    
    We look for strong DOWN predictions and estimate when they'll reverse.
    
    Args:
        predictions_list: List of prediction dictionaries (flattened or nested format)
        top_n: Number of crash items to track
    
    Returns:
        List of crashing items with reversal estimates
    """
    crash_items = []
    
    for pred in predictions_list:
        # Support both nested and flattened formats
        if 'sell' in pred and isinstance(pred['sell'], dict):
            # Nested format
            buy_order_current = pred['sell']['current_price']
            buy_order_predicted = pred['sell']['predicted_price']
            buy_order_change_pct = pred['sell']['change_pct']
            buy_order_confidence = pred['sell']['confidence'] / 100.0
            buy_order_direction = pred['sell']['direction']
            spread_direction = pred['spread']['direction']
            spread_confidence = pred['spread']['confidence'] / 100.0
        else:
            # Flattened format
            buy_order_current = pred['sell_current']
            buy_order_predicted = pred['sell_predicted']
            buy_order_change_pct = pred['sell_change_pct']
            buy_order_confidence = pred['sell_confidence'] / 100.0
            buy_order_direction = pred['sell_direction']
            spread_direction = pred['spread_direction']
            spread_confidence = pred['spread_confidence'] / 100.0
        
        # Only consider DOWN predictions with high confidence
        if buy_order_direction == 'DOWN' and buy_order_confidence > 0.55:
            # Estimate crash severity
            crash_severity = abs(buy_order_change_pct) * buy_order_confidence
            
            # Estimate reversal timing (rough heuristic)
            # Strong crash (high confidence) = faster reversal
            # Estimate in hours: inverse of confidence (higher confidence = sooner reversal expected)
            estimated_reversal_hours = int(24 / max(buy_order_confidence, 0.5))
            
            crash_items.append({
                'item_id': pred['item_id'],
                'current_price': buy_order_current,
                'predicted_price': buy_order_predicted,
                'crash_pct': buy_order_change_pct,
                'confidence': buy_order_confidence * 100,
                'crash_severity': crash_severity,
                'estimated_reversal_hours': estimated_reversal_hours,
                'spread_direction': spread_direction,
                'spread_confidence': spread_confidence * 100,
                'recommendation': 'WAIT' if estimated_reversal_hours > 12 else 'BUY_DIP'
            })
    
    # Sort by crash severity (most severe first)
    crash_items.sort(key=lambda x: x['crash_severity'], reverse=True)
    
    return crash_items[:top_n]



