"""
Hourly Market Prediction System
Uses trained Q-SHAP+ model to predict next-hour SPX direction

Usage:
    # Single prediction
    python hourly_market_predictor.py
    
    # Continuous hourly monitoring (runs every hour)
    python hourly_market_predictor.py --monitor
    
    # Backtest on historical data
    python hourly_market_predictor.py --backtest --days 30
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import schedule
import time
import argparse

# Import PennyLane for quantum model
try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    print("WARNING: PennyLane not installed. Quantum predictions unavailable.")
    PENNYLANE_AVAILABLE = False

# ============================================================================
# QUANTUM MODEL RECONSTRUCTION
# ============================================================================

class QuantumModel:
    """Wrapper for quantum model (must match training definition)"""
    def __init__(self, weights, quantum_circuit=None):
        self.weights = weights
        self.quantum_circuit = quantum_circuit
    
    def predict_proba(self, X):
        if self.quantum_circuit is None:
            raise RuntimeError("Quantum circuit not attached")
        probs = []
        for xi in X.values if hasattr(X, "values") else X:
            expval = self.quantum_circuit(xi, self.weights)
            p1 = (1 - expval) / 2
            probs.append([1 - p1, p1])
        return qml.numpy.array(probs)
    
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

# ============================================================================
# TECHNICAL INDICATOR CALCULATIONS
# ============================================================================

def calculate_ad(df):
    """Calculate Accumulation/Distribution indicator"""
    # Handle edge case where High == Low
    high_low_diff = df['High'] - df['Low']
    high_low_diff = high_low_diff.replace(0, np.nan)  # Avoid division by zero
    
    clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / high_low_diff
    clv = clv.fillna(0)  # Fill NaN with 0
    ad = (clv * df['Volume']).cumsum()
    
    # Ensure 1D array
    if isinstance(ad, pd.DataFrame):
        ad = ad.iloc[:, 0]
    
    return pd.Series(ad, index=df.index, name='A/D')

def calculate_adx(df, period=14):
    """Calculate Average Directional Index"""
    # Calculate +DM and -DM
    high_diff = df['High'].diff()
    low_diff = -df['Low'].diff()
    
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
    
    # True Range - ensure we get a Series
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - df['Close'].shift()).abs()
    tr3 = (df['Low'] - df['Close'].shift()).abs()
    
    # Create DataFrame and get max as Series
    tr_df = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3})
    tr = tr_df.max(axis=1)
    
    # Smoothed indicators
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    # ADX - handle division by zero
    di_sum = plus_di + minus_di
    di_sum = di_sum.replace(0, np.nan)  # Avoid division by zero
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    adx = dx.rolling(window=period).mean()
    
    # Ensure 1D and return as Series
    if isinstance(adx, pd.DataFrame):
        adx = adx.iloc[:, 0]
    
    return pd.Series(adx, index=df.index, name='ADX')

def calculate_macd(df, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    
    # Ensure 1D
    if isinstance(macd, pd.DataFrame):
        macd = macd.iloc[:, 0]
    
    return pd.Series(macd, index=df.index, name='MACD')

def calculate_rsi(df, period=14):
    """Calculate Relative Strength Index"""
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    
    # Avoid division by zero
    loss = loss.replace(0, np.nan)
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Ensure 1D
    if isinstance(rsi, pd.DataFrame):
        rsi = rsi.iloc[:, 0]
    
    return pd.Series(rsi, index=df.index, name='RSI')

def fetch_and_calculate_features(symbol='SPY', hours=100):
    """
    Fetch recent hourly data and calculate technical indicators
    
    Parameters:
    -----------
    symbol : str
        Ticker symbol (default: SPY for S&P 500)
    hours : int
        Number of hours of historical data to fetch
    
    Returns:
    --------
    pd.DataFrame with calculated features
    """
    print(f"📡 Fetching {hours} hours of data for {symbol}...")
    
    # Fetch hourly data
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=hours)
    
    df = yf.download(symbol, start=start_date, end=end_date, interval='1h', 
                     progress=False, auto_adjust=True)
    
    if df.empty:
        raise ValueError(f"No data fetched for {symbol}")
    
    print(f"✓ Fetched {len(df)} hourly bars")
    
    # Calculate indicators
    print("🔧 Calculating technical indicators...")
    df['A/D'] = calculate_ad(df)
    df['ADX'] = calculate_adx(df, period=14)
    df['MACD'] = calculate_macd(df, fast=12, slow=26)
    df['RSI'] = calculate_rsi(df, period=14)
    
    # Keep only needed columns
    df = df[['Close', 'A/D', 'ADX', 'MACD', 'RSI']].copy()
    
    # Drop rows with NaN (from indicator calculations)
    df = df.dropna()
    
    print(f"✓ Calculated features for {len(df)} bars")
    
    return df

# ============================================================================
# FEATURE BINARIZATION (MUST MATCH TRAINING)
# ============================================================================

def binarize_features(df):
    """
    Binarize features using the SAME strategy as training
    
    Strategy from training config:
    - Close: median split
    - A/D: zero-crossing
    - ADX: threshold >= 25
    - MACD: zero-crossing  
    - RSI: threshold >= 50
    """
    df_binary = df.copy()
    
    # Close: median split
    df_binary['Close'] = (df['Close'] >= df['Close'].median()).astype(int)
    
    # A/D: zero-crossing
    df_binary['A/D'] = (df['A/D'] >= 0).astype(int)
    
    # ADX: threshold >= 25
    df_binary['ADX'] = (df['ADX'] >= 25).astype(int)
    
    # MACD: zero-crossing
    df_binary['MACD'] = (df['MACD'] >= 0).astype(int)
    
    # RSI: threshold >= 50
    df_binary['RSI'] = (df['RSI'] >= 50).astype(int)
    
    return df_binary

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_models(dataset_name="spx_hourly"):
    """Load trained XGBoost and Quantum models"""
    xgb_path = f"../models/xgb_model_{dataset_name}.joblib"
    quantum_path = f"../models/quantum_model_{dataset_name}.joblib"
    
    models = {}
    
    # Load XGBoost
    if os.path.exists(xgb_path):
        models['xgboost'] = joblib.load(xgb_path)
        print(f"✓ Loaded XGBoost model from {xgb_path}")
    else:
        print(f"✗ XGBoost model not found: {xgb_path}")
    
    # Load Quantum model
    if os.path.exists(quantum_path) and PENNYLANE_AVAILABLE:
        quantum_model = joblib.load(quantum_path)
        
        # Reconstruct quantum circuit (must match training)
        n_qubits = 6  # 5 features + 1 ancilla
        dev = qml.device("lightning.qubit", wires=n_qubits, shots=None)
        
        @qml.qnode(dev)
        def quantum_circuit(x, weights):
            for i in range(5):  # 5 features
                qml.RY(qml.numpy.pi / 2 * x[i], wires=i)
            for i in range(5):
                qml.CRY(weights[i], wires=[i, 5])
            return qml.expval(qml.PauliZ(5))
        
        quantum_model.quantum_circuit = quantum_circuit
        models['quantum'] = quantum_model
        print(f"✓ Loaded Quantum model from {quantum_path}")
    else:
        print(f"✗ Quantum model not available")
    
    return models

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_next_hour(models, current_features):
    """
    Make predictions using all available models
    
    Parameters:
    -----------
    models : dict
        Dictionary of loaded models
    current_features : pd.DataFrame
        Single row with binarized features
    
    Returns:
    --------
    dict : Predictions from each model
    """
    predictions = {}
    
    if 'xgboost' in models:
        xgb_pred = models['xgboost'].predict(current_features)[0]
        xgb_proba = models['xgboost'].predict_proba(current_features)[0, 1]
        predictions['XGBoost'] = {
            'prediction': 'UP ⬆️' if xgb_pred == 1 else 'DOWN ⬇️',
            'probability': xgb_proba,
            'confidence': abs(xgb_proba - 0.5) * 2  # 0 to 1 scale
        }
    
    if 'quantum' in models:
        q_pred = models['quantum'].predict(current_features)[0]
        q_proba = models['quantum'].predict_proba(current_features)[0, 1]
        predictions['Q-SHAP+'] = {
            'prediction': 'UP ⬆️' if q_pred == 1 else 'DOWN ⬇️',
            'probability': q_proba,
            'confidence': abs(q_proba - 0.5) * 2
        }
    
    return predictions

def print_prediction_report(predictions, current_data, timestamp):
    """Print formatted prediction report"""
    print("\n" + "="*70)
    print(f"📊 HOURLY MARKET PREDICTION - {timestamp.strftime('%Y-%m-%d %H:%M')}")
    print("="*70)
    
    print("\n📈 Current Market State:")
    print(f"  Close: ${current_data['Close'].iloc[-1]:.2f}")
    print(f"  A/D: {current_data['A/D'].iloc[-1]:.2f} ({'Accumulation' if current_data['A/D'].iloc[-1] >= 0 else 'Distribution'})")
    print(f"  ADX: {current_data['ADX'].iloc[-1]:.1f} ({'Trending' if current_data['ADX'].iloc[-1] >= 25 else 'Ranging'})")
    print(f"  MACD: {current_data['MACD'].iloc[-1]:.2f} ({'Bullish' if current_data['MACD'].iloc[-1] >= 0 else 'Bearish'})")
    print(f"  RSI: {current_data['RSI'].iloc[-1]:.1f} ({'Overbought' if current_data['RSI'].iloc[-1] >= 70 else 'Oversold' if current_data['RSI'].iloc[-1] <= 30 else 'Neutral'})")
    
    print("\n🎯 Model Predictions for Next Hour:")
    for model_name, pred in predictions.items():
        confidence_bar = "█" * int(pred['confidence'] * 20)
        print(f"\n  {model_name}:")
        print(f"    Direction: {pred['prediction']}")
        print(f"    Probability: {pred['probability']:.1%}")
        print(f"    Confidence: {confidence_bar} {pred['confidence']:.1%}")
    
    # Ensemble prediction
    if len(predictions) > 1:
        avg_proba = np.mean([p['probability'] for p in predictions.values()])
        ensemble_pred = 'UP ⬆️' if avg_proba >= 0.5 else 'DOWN ⬇️'
        print(f"\n  📊 Ensemble Consensus: {ensemble_pred} ({avg_proba:.1%})")
    
    print("\n" + "="*70 + "\n")

# ============================================================================
# MAIN PREDICTION WORKFLOW
# ============================================================================

def run_prediction(symbol='SPY'):
    """Main prediction workflow"""
    try:
        # 1. Load models
        print("\n🔄 Loading trained models...")
        models = load_models()
        
        if not models:
            print("❌ No models available. Please train models first.")
            return
        
        # 2. Fetch and calculate features
        df = fetch_and_calculate_features(symbol=symbol, hours=100)
        
        # 3. Binarize features
        print("🔢 Binarizing features...")
        df_binary = binarize_features(df)
        
        # 4. Get latest features
        current_features = df_binary.iloc[[-1]]  # Most recent hour
        current_data = df.iloc[[-1]]
        
        # 5. Make predictions
        print("🔮 Making predictions...")
        predictions = predict_next_hour(models, current_features)
        
        # 6. Print report
        print_prediction_report(predictions, df, datetime.now())
        
        # 7. Save to log
        log_prediction(predictions, current_data, datetime.now())
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()

def log_prediction(predictions, current_data, timestamp):
    """Save predictions to CSV log"""
    log_file = "../results/prediction_log.csv"
    
    log_entry = {
        'Timestamp': timestamp,
        'Close': current_data['Close'].iloc[-1],
        'A/D': current_data['A/D'].iloc[-1],
        'ADX': current_data['ADX'].iloc[-1],
        'MACD': current_data['MACD'].iloc[-1],
        'RSI': current_data['RSI'].iloc[-1],
    }
    
    for model_name, pred in predictions.items():
        log_entry[f'{model_name}_Prediction'] = pred['prediction']
        log_entry[f'{model_name}_Probability'] = pred['probability']
    
    log_df = pd.DataFrame([log_entry])
    
    if os.path.exists(log_file):
        log_df.to_csv(log_file, mode='a', header=False, index=False)
    else:
        log_df.to_csv(log_file, index=False)
    
    print(f"💾 Logged to {log_file}")

# ============================================================================
# BACKTESTING
# ============================================================================

def backtest(symbol='SPY', days=30):
    """Backtest model on historical data"""
    print(f"\n📊 BACKTESTING - Last {days} days\n")
    
    models = load_models()
    if not models:
        return
    
    # Fetch historical data
    hours = days * 24
    df = fetch_and_calculate_features(symbol=symbol, hours=hours)
    df_binary = binarize_features(df)
    
    # Calculate actual next-hour returns
    df['Next_Return'] = df['Close'].shift(-1) / df['Close'] - 1
    df['Actual_Direction'] = (df['Next_Return'] > 0).astype(int)
    
    results = []
    
    for i in range(len(df_binary) - 1):  # Exclude last row (no next hour)
        features = df_binary.iloc[[i]]
        actual = df['Actual_Direction'].iloc[i]
        
        for model_name, model in models.items():
            pred = model.predict(features)[0]
            correct = int(pred == actual)
            
            results.append({
                'Timestamp': df.index[i],
                'Model': model_name,
                'Prediction': pred,
                'Actual': actual,
                'Correct': correct
            })
    
    results_df = pd.DataFrame(results)
    
    # Calculate accuracy per model
    print("🎯 Backtest Results:\n")
    for model_name in results_df['Model'].unique():
        model_results = results_df[results_df['Model'] == model_name]
        accuracy = model_results['Correct'].mean()
        print(f"  {model_name}: {accuracy:.1%} accuracy ({model_results['Correct'].sum()}/{len(model_results)} correct)")
    
    # Save results
    results_df.to_csv(f"../results/backtest_{days}days.csv", index=False)
    print(f"\n💾 Saved backtest results to ../results/backtest_{days}days.csv")

# ============================================================================
# CONTINUOUS MONITORING
# ============================================================================

def schedule_hourly_predictions(symbol='SPY'):
    """Run predictions every hour"""
    print("⏰ Starting hourly prediction monitor...")
    print("   Predictions will run at the top of every hour")
    print("   Press Ctrl+C to stop\n")
    
    # Run once immediately
    run_prediction(symbol)
    
    # Schedule hourly
    schedule.every().hour.at(":00").do(run_prediction, symbol=symbol)
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hourly Market Prediction System')
    parser.add_argument('--monitor', action='store_true', 
                       help='Run continuous hourly monitoring')
    parser.add_argument('--backtest', action='store_true',
                       help='Run backtest on historical data')
    parser.add_argument('--days', type=int, default=30,
                       help='Number of days for backtesting (default: 30)')
    parser.add_argument('--symbol', type=str, default='SPY',
                       help='Stock symbol (default: SPY)')
    
    args = parser.parse_args()
    
    if args.backtest:
        backtest(symbol=args.symbol, days=args.days)
    elif args.monitor:
        schedule_hourly_predictions(symbol=args.symbol)
    else:
        # Single prediction
        run_prediction(symbol=args.symbol)