import numpy as np


def calculate_tukey_outliers(prices, multiplier=1.5):
    """
    Detect outliers using Tukey's IQR method.
    
    Args:
        prices: Array or list of prices
        multiplier: IQR multiplier (1.5 = standard, 3.0 = extreme outliers only)
        
    Returns:
        tuple: (outlier_mask, lower_bound, upper_bound, q1, q3)
    """
    if len(prices) == 0:
        return np.array([]), 0, 0, 0, 0
    
    prices = np.array(prices)
    q1 = np.percentile(prices, 25)
    q3 = np.percentile(prices, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    outlier_mask = (prices < lower_bound) | (prices > upper_bound)
    
    return outlier_mask, lower_bound, upper_bound, q1, q3


def calculate_z_score_outliers(prices, threshold=3):
    """
    Detect outliers using Z-score method.
    
    Args:
        prices: Array or list of prices
        threshold: Z-score threshold (typically 3)
        
    Returns:
        tuple: (outlier_mask, mean, std_dev)
    """
    if len(prices) == 0:
        return np.array([]), 0, 0
    
    prices = np.array(prices)
    mean = np.mean(prices)
    std_dev = np.std(prices)
    
    if std_dev == 0:
        return np.zeros(len(prices), dtype=bool), mean, std_dev
    
    z_scores = np.abs((prices - mean) / std_dev)
    outlier_mask = z_scores > threshold
    
    return outlier_mask, mean, std_dev


def remove_outliers(X, y, timestamps=None, method='tukey', multiplier=1.5):
    """
    Remove outliers from training data based on target prices.
    
    Args:
        X: Feature matrix (numpy array)
        y: Target prices (numpy array)
        timestamps: Optional timestamps array
        method: 'tukey' or 'zscore'
        multiplier: For Tukey method (1.5 standard, 3.0 extreme only)
        
    Returns:
        tuple: (X_clean, y_clean, timestamps_clean, outlier_info)
    """
    if method == 'tukey':
        outlier_mask, lower, upper, q1, q3 = calculate_tukey_outliers(y, multiplier)
        outlier_info = {
            'method': 'tukey',
            'lower_bound': lower,
            'upper_bound': upper,
            'q1': q1,
            'q3': q3,
            'iqr': q3 - q1,
            'multiplier': multiplier
        }
    else:  # zscore
        outlier_mask, mean, std = calculate_z_score_outliers(y)
        outlier_info = {
            'method': 'zscore',
            'mean': mean,
            'std': std,
            'lower_bound': mean - 3 * std,
            'upper_bound': mean + 3 * std
        }
    
    # Invert mask to keep non-outliers
    keep_mask = ~outlier_mask
    
    outlier_info['total_samples'] = len(y)
    outlier_info['outliers_removed'] = np.sum(outlier_mask)
    outlier_info['outlier_percentage'] = (np.sum(outlier_mask) / len(y)) * 100
    
    X_clean = X[keep_mask]
    y_clean = y[keep_mask]
    
    if timestamps is not None:
        timestamps_clean = timestamps[keep_mask]
        return X_clean, y_clean, timestamps_clean, outlier_info
    
    return X_clean, y_clean, outlier_info


def detect_outliers_for_viz(prices, timestamps, method='tukey', multiplier=1.5):
    """
    Detect outliers for visualization purposes without removing them.
    
    Args:
        prices: Array of prices
        timestamps: Array of timestamps
        method: 'tukey' or 'zscore'
        multiplier: For Tukey method
        
    Returns:
        dict with outlier information for visualization
    """
    prices = np.array(prices)
    
    if method == 'tukey':
        outlier_mask, lower, upper, q1, q3 = calculate_tukey_outliers(prices, multiplier)
        bounds = {
            'lower': float(lower),
            'upper': float(upper),
            'q1': float(q1),
            'q3': float(q3),
            'iqr': float(q3 - q1)
        }
    else:
        outlier_mask, mean, std = calculate_z_score_outliers(prices)
        bounds = {
            'lower': float(mean - 3 * std),
            'upper': float(mean + 3 * std),
            'mean': float(mean),
            'std': float(std)
        }
    
    outlier_indices = np.where(outlier_mask)[0].tolist()
    outlier_prices = prices[outlier_mask].tolist()
    outlier_timestamps = [timestamps[i] for i in outlier_indices]
    
    return {
        'method': method,
        'bounds': bounds,
        'outlier_indices': outlier_indices,
        'outlier_prices': outlier_prices,
        'outlier_timestamps': outlier_timestamps,
        'count': len(outlier_indices),
        'percentage': (len(outlier_indices) / len(prices)) * 100 if len(prices) > 0 else 0
    }
