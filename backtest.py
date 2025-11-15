import numpy as np
from datetime import datetime
import joblib


def simple_trading_strategy(predictions, actuals, threshold=0.02):
    """
    Simple trading strategy: Buy when model predicts price will increase by threshold%
    
    Args:
        predictions: Array of predicted prices
        actuals: Array of actual prices
        threshold: Minimum predicted price increase % to trigger buy
        
    Returns:
        dict with trading results
    """
    capital = 100000000  # Starting capital in coins
    inventory = 0
    trades = []
    
    for i in range(len(predictions) - 1):
        current_price = actuals[i]
        predicted_next = predictions[i + 1]
        actual_next = actuals[i + 1]
        
        # Calculate predicted return
        predicted_return = (predicted_next - current_price) / current_price
        
        # Buy signal: predict price will increase
        if predicted_return > threshold and capital > 0:
            # Buy with all capital
            quantity = capital / current_price
            inventory += quantity
            capital = 0
            trades.append({
                'index': i,
                'action': 'BUY',
                'price': current_price,
                'quantity': quantity,
                'predicted_return': predicted_return * 100
            })
        
        # Sell signal: have inventory and predict price will decrease
        elif predicted_return < -threshold and inventory > 0:
            # Sell all inventory
            capital = inventory * current_price
            trades.append({
                'index': i,
                'action': 'SELL',
                'price': current_price,
                'quantity': inventory,
                'predicted_return': predicted_return * 100
            })
            inventory = 0
    
    # Liquidate remaining inventory at final price
    if inventory > 0:
        capital = inventory * actuals[-1]
        trades.append({
            'index': len(actuals) - 1,
            'action': 'SELL (Final)',
            'price': actuals[-1],
            'quantity': inventory,
            'predicted_return': 0
        })
        inventory = 0
    
    return {
        'final_capital': capital,
        'trades': trades,
        'num_trades': len(trades),
        'profit_loss': capital - 100000000,
        'return_pct': ((capital - 100000000) / 100000000) * 100
    }


def buy_and_hold_strategy(actuals):
    """
    Baseline strategy: Buy at start, hold until end
    
    Args:
        actuals: Array of actual prices
        
    Returns:
        dict with results
    """
    starting_capital = 100000000
    quantity = starting_capital / actuals[0]
    final_value = quantity * actuals[-1]
    
    return {
        'final_capital': final_value,
        'num_trades': 2,  # Buy at start, sell at end
        'profit_loss': final_value - starting_capital,
        'return_pct': ((final_value - starting_capital) / starting_capital) * 100
    }


def run_backtest(X_test, y_test, model, scaler, timestamps_test):
    """
    Run backtest simulation on test data
    
    Args:
        X_test: Test features
        y_test: Actual test prices
        model: Trained model
        scaler: Fitted scaler
        timestamps_test: Test timestamps
        
    Returns:
        dict with backtest results
    """
    # Get predictions
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    # Run strategies
    ml_results = simple_trading_strategy(y_pred, y_test, threshold=0.02)
    hold_results = buy_and_hold_strategy(y_test)
    
    # Calculate metrics
    start_date = datetime.strptime(timestamps_test[0][:10], '%Y-%m-%d')
    end_date = datetime.strptime(timestamps_test[-1][:10], '%Y-%m-%d')
    days = (end_date - start_date).days
    
    return {
        'ml_strategy': ml_results,
        'buy_and_hold': hold_results,
        'period_days': days,
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'start_price': float(y_test[0]),
        'end_price': float(y_test[-1]),
        'price_change_pct': ((y_test[-1] - y_test[0]) / y_test[0]) * 100
    }


def print_backtest_report(results):
    """Print formatted backtest results"""
    print(f"\n{'='*70}")
    print(f"BACKTEST RESULTS")
    print(f"{'='*70}")
    print(f"Period: {results['start_date']} to {results['end_date']} ({results['period_days']} days)")
    print(f"Start Price: {results['start_price']:,.2f} coins")
    print(f"End Price: {results['end_price']:,.2f} coins")
    print(f"Market Change: {results['price_change_pct']:+.2f}%")
    print(f"{'='*70}\n")
    
    # ML Strategy
    ml = results['ml_strategy']
    print(f"ML TRADING STRATEGY:")
    print(f"  Starting Capital: 100,000,000 coins")
    print(f"  Final Capital: {ml['final_capital']:,.2f} coins")
    print(f"  Profit/Loss: {ml['profit_loss']:+,.2f} coins")
    print(f"  Return: {ml['return_pct']:+.2f}%")
    print(f"  Number of Trades: {ml['num_trades']}")
    
    if len(ml['trades']) > 0:
        print(f"\n  Sample Trades (first 5):")
        for trade in ml['trades'][:5]:
            action = trade['action']
            price = trade['price']
            qty = trade['quantity']
            pred_ret = trade.get('predicted_return', 0)
            print(f"    {action}: {qty:.2f} @ {price:,.2f} coins (predicted: {pred_ret:+.2f}%)")
    
    print(f"\n{'='*70}\n")
    
    # Buy and Hold
    hold = results['buy_and_hold']
    print(f"BUY & HOLD STRATEGY (Baseline):")
    print(f"  Starting Capital: 100,000,000 coins")
    print(f"  Final Capital: {hold['final_capital']:,.2f} coins")
    print(f"  Profit/Loss: {hold['profit_loss']:+,.2f} coins")
    print(f"  Return: {hold['return_pct']:+.2f}%")
    
    print(f"\n{'='*70}\n")
    
    # Comparison
    outperformance = ml['return_pct'] - hold['return_pct']
    print(f"ML Strategy vs Buy & Hold: {outperformance:+.2f}% {'✓' if outperformance > 0 else '✗'}")
    print(f"{'='*70}\n")
