"""
================================================================================
Complete Three-Method Analysis: SHAP vs SHAP+ vs Q-SHAP+
Optimized Quantum Training with Gradient-based Adam Optimizer
FULLY GENERIC VERSION - Configurable Data Loading & Binarization
================================================================================

Author: Pranav Sanghadia
Email: psanghadia@captechu.edu
Institution: Capitol Technology University
Date: October 2025

Description:
    This script implements a comprehensive comparison of three explainability 
    methods for binary classification with CONFIGURABLE data loading.
    
    - Method 1: SHAP (correlational, XGBoost-based)
    - Method 2: SHAP+ (causal, XGBoost-based)
    - Method 3: Q-SHAP+ (causal, Quantum computing-based)
    
    FULLY GENERIC VERSION:
    - Configuration-based data loading (multiple dataset support)
    - Multiple binarization strategies (quantile, median, zero-crossing)
    - No hardcoded feature names - uses column indices
    - Works with CSV, Excel, and other formats
    
Requirements:
    - Python 3.8+
    - numpy, pandas, scikit-learn
    - xgboost, shap
    - pennylane (for quantum computing)
    - matplotlib, seaborn
    - openpyxl (for Excel files)
    - joblib

About Exogenous Data:
=====================
1) SPX daily closing price (for target)
2) Gold daily close (via Yahoo Finance)
3) DXY (US Dollar Index) daily close (via Yahoo Finance or FRED, whichever is available)
4) Sunspot count (daily) from SILSO / LISIRD
5) Kp-index (planetary geomagnetic) historical daily values (or derived from 3-hr Kp/AP data)
6) Global earthquake daily count (using USGS event API)  

Usage:
    # Edit DATASET_CONFIG at top of script to select dataset
    python qshap+_framework.py

Output:
    ../models/     - Trained models
    ../results/    - CSV tables and PNG visualizations
    
================================================================================
"""

import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import shap
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path
import time
import warnings
import multiprocessing
warnings.filterwarnings('ignore')

# ============================================================================
# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
# ============================================================================
# Change this to switch between datasets
DATASET_CONFIG = {
    # ========================================================================
    # CREDIT CARD DEFAULT DATASET
    # ========================================================================
    "credit_default": {
        "file_path": "../data/default_of_credit_card_clients.xlsx",
        "file_type": "excel",  # 'excel' or 'csv'
        "header_row": 1,  # Row index for headers (0-based, None for first row)
        "target_column": "Default",  # Name of target column
        "target_rename": {"default payment next month": "Default"},  # Rename mapping
        "feature_columns": ["LIMIT_BAL", "AGE", "PAY_AMT1", "EDUCATION", "MARRIAGE"],
        "binarization_strategy": "quantile",  # 'quantile', 'median', 'zero', 'threshold'
        "binarization_config": {
            # Column index: (strategy, parameters)
            0: ("quantile", {"q": 2}),  # LIMIT_BAL: quantile split
            1: ("quantile", {"q": 2}),  # AGE: quantile split
            2: ("quantile", {"q": 2}),  # PAY_AMT1: quantile split
            3: ("threshold", {"threshold": 2, "operator": ">="}),  # EDUCATION: >=2 → 1
            4: ("threshold", {"threshold": 1, "operator": "=="}),  # MARRIAGE: ==1 → 1
        },
        "interference_feature_idx": 2,  # Index for quantum interference analysis
        "entanglement_pair_idxs": (1, 3),  # (Age, EDUCATION) for entanglement
    },
    
    # ========================================================================
    # SPX FEATURES DATASET (Financial Market Data - Daily)
    # ========================================================================
    "spx_features": {
        "file_path": "../spx_data/spx_features.csv",
        "file_type": "csv",
        "header_row": 0,
        "target_column": "Target",
        "target_rename": {},  # No renaming needed
        "feature_columns": ["ΔVIX", "ΔTNX", "SPY_Vol_Change", "GLD_USO_Ratio", "ΔDXY"],
        "binarization_strategy": "median",  # Use median split for continuous features
        "binarization_config": {
            # All features are continuous percent changes → median split
            0: ("median", {}),  # ΔVIX
            1: ("median", {}),  # ΔTNX
            2: ("median", {}),  # SPY_Vol_Change
            3: ("median", {}),  # GLD_USO_Ratio
            4: ("median", {}),  # ΔDXY
        },
        "interference_feature_idx": 2,  # SPY_Vol_Change
        "entanglement_pair_idxs": (0, 1),  # (ΔVIX, ΔTNX)
    },
    
    # ========================================================================
    # SPX HOURLY TECHNICAL INDICATORS (Financial Market Data - Hourly)
    # ========================================================================
    "spx_hourly": {
        "file_path": "../spx_data/spx_hourly_indicators.csv",
        "file_type": "csv",
        "header_row": 0,
        "target_column": "Target",
        "target_rename": {},  # Target already binary
        "feature_columns": ["Close", "A/D", "ADX", "MACD", "RSI"],
        "binarization_strategy": "mixed",  # Different strategies per feature
        "binarization_config": {
            # Strategic binarization based on technical analysis principles
            0: ("median", {}),              # Close: above/below median price
            1: ("zero", {}),                # A/D: positive (accumulation) vs negative (distribution)
            2: ("threshold", {"threshold": 25, "operator": ">="}),  # ADX: >25 = strong trend
            3: ("zero", {}),                # MACD: positive (bullish) vs negative (bearish)
            4: ("threshold", {"threshold": 50, "operator": ">="}),  # RSI: >50 = bullish momentum
        },
        "interference_feature_idx": 4,      # RSI (momentum indicator)
        "entanglement_pair_idxs": (3, 4),   # (MACD, RSI) - momentum indicators
    },
    
    # ========================================================================
    # SPX EXOGENOUS FACTORS (External Predictors of Market Direction)
    # ========================================================================
    "spx_exogenous": {
        "file_path": "../spx_data/spx_exogenous_features_merged.csv",
        "file_type": "csv",
        "header_row": 0,
        "target_column": "Target",
        "target_rename": {},
        "target_creation": "next_day_direction",  # Special: create target from SPX_Close
        "feature_columns": ["Gold_Close", "DXY", "Sunspot", "Kp", "Quake_Count"],
        "binarization_strategy": "mixed",
        "binarization_config": {
            # Exogenous factor binarization strategies
            0: ("pct_change_zero", {}),     # Gold_Close: positive vs negative daily change
            1: ("pct_change_zero", {}),     # DXY: dollar strength increase vs decrease
            2: ("median", {}),              # Sunspot: high vs low solar activity
            3: ("threshold", {"threshold": 3, "operator": ">="}),  # Kp: >=3 = geomagnetic storm
            4: ("threshold", {"threshold": 30, "operator": ">="}), # Quake_Count: >=30 = high seismic activity
        },
        "interference_feature_idx": 2,      # Sunspot (solar cycles)
        "entanglement_pair_idxs": (2, 3),   # (Sunspot, Kp) - space weather factors
    },
}

# ============================================================================
# SELECT ACTIVE DATASET
# ============================================================================
ACTIVE_DATASET = "spx_exogenous"  # Options: "credit_default", "spx_features", "spx_hourly", "spx_exogenous"
config = DATASET_CONFIG[ACTIVE_DATASET]

# ============================================================================
# HELPER FUNCTION: ADD NEW DATASET
# ============================================================================
def add_custom_dataset(name, file_path, feature_columns, target_column,
                       file_type="csv", binarization_strategy="median",
                       custom_binarization=None):
    """
    Helper function to quickly add a new dataset configuration.
    
    Parameters:
    -----------
    name : str
        Dataset identifier (e.g., "my_dataset")
    file_path : str
        Path to data file (CSV or Excel)
    feature_columns : list
        List of feature column names to use
    target_column : str
        Name of target/label column
    file_type : str, default="csv"
        File format: "csv" or "excel"
    binarization_strategy : str, default="median"
        Default strategy: "quantile", "median", "zero", "threshold", "pct_change_zero"
    custom_binarization : dict, optional
        Custom binarization per feature: {col_idx: (strategy, params)}
        Example: {0: ("zero", {}), 1: ("threshold", {"threshold": 50, "operator": ">="}),
                  2: ("pct_change_zero", {})}  # For price change features
    Returns:
    --------
    dict : Dataset configuration ready to add to DATASET_CONFIG
    
    Example:
    --------
    >>> config = add_custom_dataset(
    ...     name="my_crypto_data",
    ...     file_path="../data/bitcoin.csv",
    ...     feature_columns=["price", "volume", "volatility"],
    ...     target_column="direction",
    ...     custom_binarization={
    ...         0: ("median", {}),  # price
    ...         1: ("quantile", {"q": 2}),  # volume
    ...         2: ("zero", {})  # volatility
    ...     }
    ... )
    >>> DATASET_CONFIG["my_crypto_data"] = config
    """
    n_features = len(feature_columns)
    
    # Default binarization config
    if custom_binarization is None:
        binarization_config = {}
        for i in range(n_features):
            if binarization_strategy == "quantile":
                binarization_config[i] = ("quantile", {"q": 2})
            elif binarization_strategy == "median":
                binarization_config[i] = ("median", {})
            elif binarization_strategy == "zero":
                binarization_config[i] = ("zero", {})
            elif binarization_strategy == "pct_change_zero":
                binarization_config[i] = ("pct_change_zero", {})
            else:
                binarization_config[i] = ("median", {})  # Default fallback
    else:
        binarization_config = custom_binarization
    
    return {
        "file_path": file_path,
        "file_type": file_type,
        "header_row": 0,
        "target_column": target_column,
        "target_rename": {},
        "feature_columns": feature_columns,
        "binarization_strategy": binarization_strategy,
        "binarization_config": binarization_config,
        "interference_feature_idx": n_features // 2,  # Middle feature
        "entanglement_pair_idxs": (0, 1) if n_features >= 2 else (0, 0),
    }

# Example: Uncomment to add a custom dataset
# DATASET_CONFIG["my_dataset"] = add_custom_dataset(
#     name="my_dataset",
#     file_path="../data/my_data.csv",
#     feature_columns=["feature1", "feature2", "feature3"],
#     target_column="target"
# )

print("=" * 80)
print("COMPLETE THREE-METHOD ANALYSIS (Fully Generic Version)")
print("SHAP | SHAP+ | Q-SHAP+")
print("=" * 80)
print(f"\nActive Dataset: {ACTIVE_DATASET}")
print(f"File: {config['file_path']}")
print(f"Features: {len(config['feature_columns'])} - {config['feature_columns']}")
print(f"Binarization: {config['binarization_strategy']}")

# ============================================================================
# PENNYLANE IMPORT CHECK
# ============================================================================
try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    print("WARNING: PennyLane not installed. Install with: pip install pennylane")

# ============================================================================
# QUANTUM MODEL CLASS
# ============================================================================
class QuantumModel:
    """Wrapper class for quantum neural network model."""
    def __init__(self, weights, quantum_circuit=None):
        self.weights = weights
        self.quantum_circuit = quantum_circuit

    def predict_proba(self, X):
        """Returns probability predictions [P(class=0), P(class=1)]"""
        if self.quantum_circuit is None:
            raise RuntimeError("Quantum circuit not attached to loaded model.")
        probs = []
        for xi in X.values if hasattr(X, "values") else X:
            expval = self.quantum_circuit(xi, self.weights)
            p1 = (1 - expval) / 2
            probs.append([1 - p1, p1])
        return qml.numpy.array(probs)

    def predict(self, X):
        """Returns binary class predictions"""
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
    
    def __getstate__(self):
        """Custom serialization - exclude quantum_circuit function"""
        state = self.__dict__.copy()
        # Remove the unpicklable quantum_circuit
        state['quantum_circuit'] = None
        return state
    
    def __setstate__(self, state):
        """Custom deserialization"""
        self.__dict__.update(state)

# Create output directories
Path("../models").mkdir(exist_ok=True)
Path("../results").mkdir(exist_ok=True)

# ============================================================================
# GENERIC BINARIZATION FUNCTIONS
# ============================================================================
def binarize_quantile(series, q=2):
    """Binarize using quantile-based binning"""
    try:
        return pd.qcut(series, q=q, labels=False, duplicates='drop')
    except ValueError:
        # Handle case with insufficient unique values
        return pd.cut(series, bins=q, labels=False, duplicates='drop')

def binarize_median(series):
    """Binarize using median threshold (>= median → 1)"""
    return (series >= series.median()).astype(int)

def binarize_zero(series):
    """Binarize using zero threshold (>= 0 → 1)"""
    return (series >= 0).astype(int)

def binarize_threshold(series, threshold, operator=">="):
    """Binarize using custom threshold"""
    if operator == ">=":
        return (series >= threshold).astype(int)
    elif operator == ">":
        return (series > threshold).astype(int)
    elif operator == "==":
        return (series == threshold).astype(int)
    elif operator == "<=":
        return (series <= threshold).astype(int)
    elif operator == "<":
        return (series < threshold).astype(int)
    else:
        raise ValueError(f"Unknown operator: {operator}")

def binarize_pct_change_zero(series):
    """Binarize using percentage change from previous day (>= 0 → 1)"""
    pct_change = series.pct_change()
    return (pct_change >= 0).astype(int)

# ============================================================================
# STEP 1: GENERIC DATA LOADING
# ============================================================================
print("\n[STEP 1] Loading dataset...")

# Load based on file type
if config["file_type"] == "excel":
    df = pd.read_excel(
        config["file_path"],
        engine="openpyxl",
        header=config["header_row"]
    )
elif config["file_type"] == "csv":
    df = pd.read_csv(config["file_path"])
    if config["header_row"] is not None and config["header_row"] > 0:
        df = df.iloc[config["header_row"]:].reset_index(drop=True)
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)
else:
    raise ValueError(f"Unsupported file type: {config['file_type']}")

# Clean column names
df.columns = df.columns.str.strip()

# Rename target if specified
if config["target_rename"]:
    df = df.rename(columns=config["target_rename"])

# ============================================================================
# CREATE TARGET IF NEEDED (for exogenous datasets)
# ============================================================================
if "target_creation" in config and config["target_creation"] == "next_day_direction":
    print("📊 Creating target from SPX_Close (next-day direction)...")
    # Assume there's a column called SPX_Close or similar
    if "SPX_Close" in df.columns:
        spx_col = "SPX_Close"
    elif "Close" in df.columns:
        spx_col = "Close"
    else:
        raise ValueError("Cannot find SPX price column for target creation")
    
    # Create target: 1 if next day close > today close, else 0
    df["Next_Close"] = df[spx_col].shift(-1)
    df["Target"] = (df["Next_Close"] > df[spx_col]).astype(int)
    
    # Drop rows with NaN target (last row)
    df = df.dropna(subset=["Target"])
    
    print(f"   Target distribution: {df['Target'].value_counts().to_dict()}")

# Convert target to integer
df[config["target_column"]] = df[config["target_column"]].astype(int)

print(f"✓ Loaded {len(df):,} records")
print(f"✓ Columns in dataset: {list(df.columns)}")

# ============================================================================
# STEP 2: GENERIC FEATURE BINARIZATION
# ============================================================================
print("\n[STEP 2] Binarizing features using configured strategies...")

features = config["feature_columns"]

# Apply binarization strategy for each feature
for idx, feature_name in enumerate(features):
    strategy, params = config["binarization_config"][idx]
    
    print(f"  Feature {idx} ({feature_name}): {strategy} strategy", end="")
    
    if strategy == "quantile":
        df[feature_name] = binarize_quantile(df[feature_name], **params)
    elif strategy == "median":
        df[feature_name] = binarize_median(df[feature_name])
    elif strategy == "zero":
        df[feature_name] = binarize_zero(df[feature_name])
    elif strategy == "threshold":
        df[feature_name] = binarize_threshold(df[feature_name], **params)
    elif strategy == "pct_change_zero":
        df[feature_name] = binarize_pct_change_zero(df[feature_name])
    else:
        raise ValueError(f"Unknown binarization strategy: {strategy}")
    
    # Show distribution
    counts = df[feature_name].value_counts().sort_index()
    print(f" → 0: {counts.get(0, 0)}, 1: {counts.get(1, 0)}")

# Drop any rows with NaN created by pct_change or other operations
df = df.dropna(subset=features)

# Create feature matrix and target
X = df[features].astype(int)
y = df[config["target_column"]]

print(f"\n✓ Features: {features}")
print(f"✓ Feature matrix shape: {X.shape}")
print(f"✓ Target distribution: {y.value_counts().to_dict()}")

# ============================================================================
# CONFIGURATION: Quantum Phenomena Analysis Indices
# ============================================================================
INTERFERENCE_FEATURE_IDX = config["interference_feature_idx"]
ENTANGLEMENT_PAIR_IDXS = config["entanglement_pair_idxs"]

print(f"\n✓ Interference analysis on: {features[INTERFERENCE_FEATURE_IDX]} (index {INTERFERENCE_FEATURE_IDX})")
print(f"✓ Entanglement pair: {features[ENTANGLEMENT_PAIR_IDXS[0]]} - {features[ENTANGLEMENT_PAIR_IDXS[1]]}")

# ============================================================================
# STEP 3: TRAIN/VALIDATION/TEST SPLIT (70/15/15)
# ============================================================================
print("\n[STEP 3] Creating 70/15/15 split...")

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"✓ Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")

# ============================================================================
# STEP 4: XGBOOST MODEL TRAINING
# ============================================================================
print("\n[STEP 4] Training XGBoost model...")

scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

xgb_model = XGBClassifier(
    eval_metric="logloss",
    use_label_encoder=False,
    scale_pos_weight=scale_pos_weight,
    n_estimators=300,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))
xgb_auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])

print(f"✓ XGBoost: Acc={xgb_acc:.3f}, AUC={xgb_auc:.3f}")
joblib.dump(xgb_model, f"../models/xgb_model_{ACTIVE_DATASET}.joblib")

# ============================================================================
# STEP 5: QUANTUM MODEL TRAINING (WITH CACHING)
# ============================================================================
print("\n[STEP 5] Quantum Model (Adam optimizer with caching)...")

n_qubits = len(features) + 1
os.environ["OMP_NUM_THREADS"] = str(max(1, multiprocessing.cpu_count() - 1))
dev = qml.device("lightning.qubit", wires=n_qubits, shots=None)

@qml.qnode(dev)
def quantum_circuit(x, weights):
    """Quantum circuit for binary classification"""
    for i in range(len(features)):
        qml.RY(qml.numpy.pi / 2 * x[i], wires=i)
    for i in range(len(features)):
        qml.CRY(weights[i], wires=[i, n_qubits - 1])
    return qml.expval(qml.PauliZ(n_qubits - 1))

quantum_model_path = f"../models/quantum_model_{ACTIVE_DATASET}.joblib"

if os.path.exists(quantum_model_path):
    print(f"✓ Quantum model already exists → loading from {quantum_model_path}")
    quantum_model = joblib.load(quantum_model_path)
    # Reconstruct and attach the quantum circuit
    quantum_model.quantum_circuit = quantum_circuit
    print(f"✓ Quantum circuit reconstructed and attached")
else:
    print("✗ Quantum model not found → training new model...")

    def loss(weights, X, y):
        """Binary cross-entropy loss for quantum classifier"""
        preds = []
        for xi in X:
            expval = quantum_circuit(xi, weights)
            p1 = (1 - expval) / 2
            preds.append(p1)
        preds = qml.numpy.clip(qml.numpy.array(preds), 1e-9, 1 - 1e-9)
        return -qml.numpy.mean(y * qml.numpy.log(preds) + (1 - y) * qml.numpy.log(1 - preds))

    # Adaptive subset size based on training set size
    subset_size = min(1000, len(X_train))
    qml.numpy.random.seed(42)
    idx = qml.numpy.random.choice(len(X_train), subset_size, replace=False)
    X_train_q = X_train.iloc[idx].values
    y_train_q = y_train.iloc[idx].values

    weights = qml.numpy.array(qml.numpy.random.randn(len(features)), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=0.05)
    n_steps = 150

    print(f"  Using {subset_size} samples, {n_steps} steps (Adam)")
    start_time = time.time()
    for step in range(n_steps):
        weights, curr_loss = opt.step_and_cost(lambda w: loss(w, X_train_q, y_train_q), weights)
        if step % 10 == 0:
            print(f"    Step {step:03d}: loss = {curr_loss:.4f}")
    elapsed = time.time() - start_time
    print(f"  ✓ Optimization complete: final loss = {curr_loss:.4f}")
    print(f"  Training time: {elapsed/60:.2f} minutes")

    quantum_model = QuantumModel(weights, quantum_circuit)
    q_acc = accuracy_score(y_test, quantum_model.predict(X_test))
    q_auc = roc_auc_score(y_test, quantum_model.predict_proba(X_test)[:, 1])
    print(f"✓ Quantum: Acc={q_acc:.3f}, AUC={q_auc:.3f}")

    # Save model (quantum_circuit will be excluded via __getstate__)
    joblib.dump(quantum_model, quantum_model_path)
    print(f"✓ Saved trained quantum model → {quantum_model_path}")
    print(f"  (Note: Weights saved, circuit will be reconstructed on load)")
    
    # Reattach circuit for immediate use
    quantum_model.quantum_circuit = quantum_circuit

# ============================================================================
# STEP 6: TEST SAMPLE SELECTION
# ============================================================================
print("\n[STEP 6] Selecting 54 profiles...")
qml.numpy.random.seed(42)
test_indices = qml.numpy.random.choice(len(X_test), size=54, replace=False)
X_sample = X_test.iloc[test_indices].reset_index(drop=True)
y_sample = y_test.iloc[test_indices].reset_index(drop=True)
print(f"✓ Selected 54 profiles")

# ============================================================================
# STEP 7: METHOD 1 - SHAP (CORRELATIONAL)
# ============================================================================
print("\n[STEP 7] Method 1: SHAP (TreeExplainer)...")
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_sample.values)
print(f"✓ SHAP values calculated")

# ============================================================================
# STEP 8: METHOD 2 - SHAP+ (CAUSAL, XGBOOST)
# ============================================================================
print("\n[STEP 8] Method 2: SHAP+ (Causal on XGBoost)...")
shap_plus_values = qml.numpy.zeros((len(X_sample), len(features)))
for i in range(len(X_sample)):
    for j in range(len(features)):
        profile = X_sample.iloc[i].values
        p_high, p_low = profile.copy(), profile.copy()
        p_high[j], p_low[j] = 1, 0
        prob_high = xgb_model.predict_proba([p_high])[0, 1]
        prob_low = xgb_model.predict_proba([p_low])[0, 1]
        shap_plus_values[i, j] = prob_high - prob_low
print(f"✓ SHAP+ values calculated")

# ============================================================================
# STEP 9: METHOD 3 - Q-SHAP+ (CAUSAL, QUANTUM)
# ============================================================================
print("\n[STEP 9] Method 3: Q-SHAP+ (Causal on Quantum)...")
if quantum_model is None or quantum_model.quantum_circuit is None:
    print("✗ Quantum model not available - using SHAP+ as placeholder")
    qshap_values = shap_plus_values.copy()
else:
    qshap_values = qml.numpy.zeros((len(X_sample), len(features)))
    for i in range(len(X_sample)):
        for j in range(len(features)):
            profile = X_sample.iloc[i].values
            p_high, p_low = profile.copy(), profile.copy()
            p_high[j], p_low[j] = 1, 0
            prob_high = quantum_model.predict_proba([p_high])[0, 1]
            prob_low = quantum_model.predict_proba([p_low])[0, 1]
            qshap_values[i, j] = prob_high - prob_low
    print(f"✓ Q-SHAP+ values calculated")

# ============================================================================
# STEP 10: INTERPRETABILITY METRICS CALCULATION
# ============================================================================
print("\n[STEP 10] Calculating interpretability metrics...")

def calculate_metrics(attributions, X, model):
    """Compute faithfulness, stability, responsiveness, and clarity"""
    metrics = []
    for i in range(len(X)):
        profile = X.iloc[i].values
        most_important = qml.numpy.argmax(qml.numpy.abs(attributions[i]))
        
        # Faithfulness
        p_orig = model.predict_proba([profile])[0, 1]
        p_mod = profile.copy()
        p_mod[most_important] = 1 - p_mod[most_important]
        p_changed = model.predict_proba([p_mod])[0, 1]
        pred_change = abs(p_orig - p_changed)
        attr_mag = abs(attributions[i, most_important])
        faith = (pred_change / (attr_mag + 0.1)) * 50 if attr_mag > 0.001 else 0
        faith = qml.numpy.clip(faith, 0, 100)
        
        # Stability
        mean_attr = qml.numpy.mean(attributions, axis=0)
        dist = qml.numpy.linalg.norm(attributions[i] - mean_attr)
        stab = 100 / (1 + dist)
        
        # Responsiveness
        p_orig = model.predict_proba([profile])[0, 1]
        sens = []
        for j in range(len(profile)):
            p_mod = profile.copy()
            p_mod[j] = 1 - p_mod[j]
            sens.append(abs(p_orig - model.predict_proba([p_mod])[0, 1]))
        resp = qml.numpy.mean(sens) * 100
        
        # Clarity
        clar = qml.numpy.clip(5 / (1 + qml.numpy.var(attributions[i]) * 10), 0, 5)
        
        metrics.append([faith, stab, resp, clar])
    return qml.numpy.array(metrics)

shap_metrics = calculate_metrics(shap_values, X_sample, xgb_model)
shapplus_metrics = calculate_metrics(shap_plus_values, X_sample, xgb_model)
qshap_metrics = calculate_metrics(qshap_values, X_sample,
                                  quantum_model if (quantum_model and quantum_model.quantum_circuit) else xgb_model)

def composite(m):
    """Weighted composite score"""
    return 0.30*m[:, 0] + 0.30*m[:, 1] + 0.20*m[:, 2] + 0.20*(m[:, 3]/5*100)

shap_scores = composite(shap_metrics)
shapplus_scores = composite(shapplus_metrics)
qshap_scores = composite(qshap_metrics)
print(f"✓ All metrics calculated")

# ============================================================================
# STEP 11: STATISTICAL HYPOTHESIS TESTING
# ============================================================================
print("\n[STEP 11] Statistical analysis...")

t2, p2 = stats.ttest_rel(qshap_scores, shap_scores)
d2 = (qshap_scores.mean() - shap_scores.mean()) / qml.numpy.std(qshap_scores - shap_scores, ddof=1)

print(f"\nComparison: SHAP vs Q-SHAP+ (MAIN HYPOTHESIS)")
print(f"  Δ={qshap_scores.mean() - shap_scores.mean():.2f}, d={d2:.3f}, p={p2:.6f}")
if p2 < 0.05 and d2 >= 0.5:
    print(f"  ✓ REJECT H₀: Q-SHAP+ significantly better")

# ============================================================================
# STEP 12: SAVE PRIMARY RESULTS
# ============================================================================
print("\n[STEP 12] Saving results...")
comparison_df = pd.DataFrame({
    'Method': ['SHAP', 'SHAP+', 'Q-SHAP+'],
    'Faithfulness': [shap_metrics[:, 0].mean(), shapplus_metrics[:, 0].mean(), qshap_metrics[:, 0].mean()],
    'Stability': [shap_metrics[:, 1].mean(), shapplus_metrics[:, 1].mean(), qshap_metrics[:, 1].mean()],
    'Responsiveness': [shap_metrics[:, 2].mean(), shapplus_metrics[:, 2].mean(), qshap_metrics[:, 2].mean()],
    'Clarity': [shap_metrics[:, 3].mean(), shapplus_metrics[:, 3].mean(), qshap_metrics[:, 3].mean()],
    'Composite': [shap_scores.mean(), shapplus_scores.mean(), qshap_scores.mean()]
})
comparison_df.to_csv(f'../results/three_method_comparison_{ACTIVE_DATASET}.csv', index=False)
print("✓ Results saved")

# ============================================================================
# STEP 13: VALIDATION METRICS & FEATURE IMPORTANCE VISUALIZATION
# ============================================================================
print("\n[STEP 13] Creating validation and feature importance comparison chart...")

# Feature importance
shap_importance = np.mean(np.abs(shap_values), axis=0)
shapplus_importance = np.mean(np.abs(shap_plus_values), axis=0)
qshap_importance = np.mean(np.abs(qshap_values), axis=0)

# Validation metrics
val_metrics = pd.DataFrame({
    "Method": ["SHAP", "SHAP+", "Q-SHAP+"],
    "Accuracy": [
        accuracy_score(y_test, xgb_model.predict(X_test)),
        accuracy_score(y_test, xgb_model.predict(X_test)),
        accuracy_score(y_test, quantum_model.predict(X_test))
            if (quantum_model and quantum_model.quantum_circuit) else accuracy_score(y_test, xgb_model.predict(X_test))
    ],
    "AUC": [
        roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1]),
        roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1]),
        roc_auc_score(
            y_test,
            quantum_model.predict_proba(X_test)[:, 1]
            if (quantum_model and quantum_model.quantum_circuit) else xgb_model.predict_proba(X_test)[:, 1]
        )
    ]
})

importance_df = pd.DataFrame({
    "Feature": features,
    "SHAP": shap_importance,
    "SHAP+": shapplus_importance,
    "Q-SHAP+": qshap_importance
})

# Create dual-panel visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"Validation Metrics & Feature Importance ({ACTIVE_DATASET})", 
             fontsize=14, fontweight="bold")

# Left: Validation metrics
width = 0.35
x = np.arange(len(val_metrics))
axes[0].bar(x - width/2, val_metrics["Accuracy"], width, label="Accuracy", color="lightcoral")
axes[0].bar(x + width/2, val_metrics["AUC"], width, label="AUC", color="lightblue")
axes[0].set_xticks(x)
axes[0].set_xticklabels(val_metrics["Method"], fontsize=10)
axes[0].set_ylim(0, 1.0)
axes[0].set_ylabel("Score")
axes[0].set_title("Validation Accuracy and AUC")
axes[0].legend()
axes[0].grid(axis="y", alpha=0.3)

# Right: Feature importance
x = np.arange(len(features))
width = 0.25
axes[1].bar(x - width, importance_df["SHAP"], width, label="SHAP", color="coral")
axes[1].bar(x, importance_df["SHAP+"], width, label="SHAP+", color="skyblue")
axes[1].bar(x + width, importance_df["Q-SHAP+"], width, label="Q-SHAP+", color="lightgreen")
axes[1].set_xticks(x)
axes[1].set_xticklabels(features, rotation=45, ha="right")
axes[1].set_ylabel("Mean |Attribution|")
axes[1].set_title("Feature Importance Comparison")
axes[1].legend()
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(f"../results/validation_feature_importance_{ACTIVE_DATASET}.png", 
            dpi=300, bbox_inches="tight")
plt.close()
print(f"✓ Saved: validation_feature_importance_{ACTIVE_DATASET}.png")

# ============================================================================
# STEP 14: INTERPRETABILITY METRICS VISUALIZATION (WITH ERROR BARS)
# ============================================================================
print("\n[STEP 14] Plotting interpretability metrics comparison (with error bars)...")

metric_names = ["Faithfulness", "Stability", "Responsiveness", "Clarity"]

means_shap = np.mean(shap_metrics, axis=0)
means_shapplus = np.mean(shapplus_metrics, axis=0)
means_qshap = np.mean(qshap_metrics, axis=0)

std_shap = np.std(shap_metrics, axis=0)
std_shapplus = np.std(shapplus_metrics, axis=0)
std_qshap = np.std(qshap_metrics, axis=0)

x = np.arange(len(metric_names))
width = 0.25

plt.figure(figsize=(10, 5))
plt.bar(x - width, means_shap, width,
        yerr=std_shap, capsize=5, label="SHAP",
        color="skyblue", edgecolor="black")
plt.bar(x, means_shapplus, width,
        yerr=std_shapplus, capsize=5, label="SHAP+ (Causal)",
        color="gold", edgecolor="black")
plt.bar(x + width, means_qshap, width,
        yerr=std_qshap, capsize=5, label="Q-SHAP+ (Quantum)",
        color="purple", edgecolor="black")

plt.xticks(x, metric_names, rotation=15, fontsize=11)
plt.ylabel("Metric Value", fontsize=12)
plt.title(f"Interpretability Metrics: SHAP vs SHAP+ vs Q-SHAP+ ({ACTIVE_DATASET})", 
          fontsize=13, fontweight="bold")
plt.legend(frameon=False, fontsize=10)
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(f"../results/interpretability_metrics_comparison_{ACTIVE_DATASET}.png",
            dpi=300, bbox_inches="tight")
plt.close()
print(f"✓ Saved: interpretability_metrics_comparison_{ACTIVE_DATASET}.png")

# ============================================================================
# STEP 15: EXPORT ALL DATA TABLES
# ============================================================================
print("\n[STEP 15] Saving chart data and 54-profile appendix tables...")

# Validation metrics
val_metrics_path = f"../results/validation_metrics_{ACTIVE_DATASET}.csv"
val_metrics.to_csv(val_metrics_path, index=False)
print(f"✓ Saved validation metrics → {val_metrics_path}")

# Interpretability metrics
interp_metrics_df = pd.DataFrame({
    "Metric": metric_names,
    "SHAP_Mean": np.mean(shap_metrics, axis=0),
    "SHAP_SD": np.std(shap_metrics, axis=0),
    "SHAP+_Mean": np.mean(shapplus_metrics, axis=0),
    "SHAP+_SD": np.std(shapplus_metrics, axis=0),
    "Q-SHAP+_Mean": np.mean(qshap_metrics, axis=0),
    "Q-SHAP+_SD": np.std(qshap_metrics, axis=0),
})
interp_metrics_path = f"../results/interpretability_metrics_data_{ACTIVE_DATASET}.csv"
interp_metrics_df.to_csv(interp_metrics_path, index=False)
print(f"✓ Saved interpretability metrics → {interp_metrics_path}")

# Feature importance
importance_path = f"../results/feature_importance_data_{ACTIVE_DATASET}.csv"
importance_df.to_csv(importance_path, index=False)
print(f"✓ Saved feature importance → {importance_path}")

# Three-method comparison (already saved, re-save with dataset name)
comparison_df.to_csv(f'../results/three_method_comparison_{ACTIVE_DATASET}.csv', index=False)
print(f"✓ Re-saved three-method comparison → ../results/three_method_comparison_{ACTIVE_DATASET}.csv")

# Detailed 54-profile appendix
appendix_df = X_sample.copy()
appendix_df["Target"] = y_sample.values

for j, f in enumerate(features):
    appendix_df[f"SHAP_{f}"] = shap_values[:, j]
    appendix_df[f"SHAP+_{f}"] = shap_plus_values[:, j]
    appendix_df[f"Q-SHAP+_{f}"] = qshap_values[:, j]

appendix_path = f"../results/appendix_54_profiles_{ACTIVE_DATASET}.csv"
appendix_df.to_csv(appendix_path, index=False)
print(f"✓ Saved 54-profile appendix → {appendix_path}")

# ============================================================================
# STEP 16: THESIS TABLES (TABLE 4 & TABLE 5)
# ============================================================================
print("\n[STEP 16] Generating thesis-style tables (Table 4 & Table 5)...")

# Table 4: Feature Attribution
table4_df = pd.DataFrame({
    "Feature": features,
    "SHAP": np.mean(np.abs(shap_values), axis=0).round(4),
    "SHAP+": np.mean(np.abs(shap_plus_values), axis=0).round(4),
    "Q-SHAP+": np.mean(np.abs(qshap_values), axis=0).round(4)
})
table4_path = f"../results/table4_feature_attribution_{ACTIVE_DATASET}.csv"
table4_df.to_csv(table4_path, index=False)
print(f"✓ Saved Table 4 → {table4_path}\n")
print(f"Table 4. Feature Attribution Comparison ({ACTIVE_DATASET})\n")
print(table4_df.to_string(index=False))

# Table 5: Quality Metrics
table5_df = pd.DataFrame({
    "Metric": ["Faithfulness (%)", "Stability (%)", "Responsiveness (%)", "Clarity (/5)"],
    "SHAP": [np.mean(shap_metrics[:, 0]), np.mean(shap_metrics[:, 1]),
             np.mean(shap_metrics[:, 2]), np.mean(shap_metrics[:, 3])],
    "SHAP+": [np.mean(shapplus_metrics[:, 0]), np.mean(shapplus_metrics[:, 1]),
              np.mean(shapplus_metrics[:, 2]), np.mean(shapplus_metrics[:, 3])],
    "Q-SHAP+": [np.mean(qshap_metrics[:, 0]), np.mean(qshap_metrics[:, 1]),
                np.mean(qshap_metrics[:, 2]), np.mean(qshap_metrics[:, 3])]
}).round(1)

table5_path = f"../results/table5_quality_metrics_{ACTIVE_DATASET}.csv"
table5_df.to_csv(table5_path, index=False)
print(f"\n✓ Saved Table 5 → {table5_path}\n")
print(f"Table 5. Quality Metrics Comparison ({ACTIVE_DATASET})\n")
print(table5_df.to_string(index=False))

# Composite scores
composite_scores = {
    "SHAP": shap_scores.mean(),
    "SHAP+": shapplus_scores.mean(),
    "Q-SHAP+": qshap_scores.mean()
}
delta = composite_scores["Q-SHAP+"] - composite_scores["SHAP"]
print("\nComposite Interpretability Scores (mean across 54 samples):")
for k, v in composite_scores.items():
    print(f"  {k:7s} → {v:.2f}")
print(f"\nΔ = X̄(Q-SHAP⁺) − X̄(SHAP) = {delta:.2f}")

# ============================================================================
# STEP 17: QUANTUM PHENOMENA DETECTION (GENERIC VERSION)
# ============================================================================
print("\n[STEP 17] Computing quantum phenomena (entanglement, interference)...")

import seaborn as sns

# Feature entanglement matrix
qshap_corr = np.corrcoef(qshap_values.T)
qshap_corr = np.nan_to_num(qshap_corr, nan=0.0)
entanglement_df = pd.DataFrame(qshap_corr, index=features, columns=features)

entanglement_path = f"../results/quantum_entanglement_matrix_{ACTIVE_DATASET}.csv"
entanglement_df.to_csv(entanglement_path)
print(f"✓ Saved entanglement matrix → {entanglement_path}")

# Heatmap visualization
plt.figure(figsize=(7, 5))
sns.heatmap(
    entanglement_df, annot=True, cmap="Blues",
    vmin=0, vmax=1, cbar_kws={'label': 'Entanglement Strength'}
)
plt.title(f"Quantum Feature Entanglement ({ACTIVE_DATASET})\n(Q-SHAP+ Only)",
          fontsize=13, fontweight="bold")
plt.xlabel("Features")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig(f"../results/quantum_feature_entanglement_{ACTIVE_DATASET}.png",
            dpi=300, bbox_inches="tight")
plt.close()
print(f"✓ Saved entanglement heatmap → ../results/quantum_feature_entanglement_{ACTIVE_DATASET}.png")

# Quantum effects summary (GENERIC - uses configured indices)
max_entanglement = float(np.max(qshap_corr))
pair_indices = np.unravel_index(np.argmax(qshap_corr, axis=None), qshap_corr.shape)
entangled_pair = (features[pair_indices[0]], features[pair_indices[1]])

# Use configured feature index for interference analysis
interference_strength = float(
    np.var(qshap_values[:, INTERFERENCE_FEATURE_IDX]) / np.var(qshap_values)
)

# Get configured entanglement pair correlation
pair_entanglement = float(qshap_corr[ENTANGLEMENT_PAIR_IDXS[0], ENTANGLEMENT_PAIR_IDXS[1]])

# Table 7: Quantum Effects (GENERIC VERSION)
table7_data = [
    ["Feature Entanglement Strength", round(max_entanglement, 3),
     "Maximum quantum correlation between feature pairs"],
    [f"{entangled_pair[0]}–{entangled_pair[1]} Entanglement",
     round(max_entanglement, 3),
     "Strongest entangled feature pair (auto-detected)"],
    [f"{features[ENTANGLEMENT_PAIR_IDXS[0]]}–{features[ENTANGLEMENT_PAIR_IDXS[1]]} Entanglement",
     round(pair_entanglement, 3),
     "Selected feature pair quantum coupling"],
    [f"{features[INTERFERENCE_FEATURE_IDX]} Interference",
     round(float(interference_strength), 3),
     f"Quantum superposition effects in feature {INTERFERENCE_FEATURE_IDX}"]
]

table7_df = pd.DataFrame(table7_data,
                         columns=["Quantum Effect", "Measurement", "Interpretation"])

table7_path = f"../results/table7_quantum_effects_{ACTIVE_DATASET}.csv"
table7_df.to_csv(table7_path, index=False)
print(f"✓ Saved Table 7 → {table7_path}\n")
print(f"Table 7. Detected Quantum Effects ({ACTIVE_DATASET})\n")
print(table7_df.to_string(index=False))

# ============================================================================
# STEP 18: INTERPRETABILITY MATRIX (54 PROFILES)
# ============================================================================
print("\n[STEP 18] Generating interpretability matrix for all 54 profiles...")

def compute_composite(m):
    """Weighted composite: 30% faith, 30% stab, 20% resp, 20% clarity"""
    return (
        0.30 * m[:, 0] +
        0.30 * m[:, 1] +
        0.20 * m[:, 2] +
        0.20 * (m[:, 3] / 5 * 100)
    )

shap_comp_all = compute_composite(shap_metrics)
shapplus_comp_all = compute_composite(shapplus_metrics)
qshap_comp_all = compute_composite(qshap_metrics)

# Full detailed matrix
interpret_df = pd.DataFrame({
    "Profile_ID": np.arange(1, len(X_sample) + 1),
    "Faithfulness_SHAP": shap_metrics[:, 0].round(2),
    "Faithfulness_SHAP+": shapplus_metrics[:, 0].round(2),
    "Faithfulness_QSHAP+": qshap_metrics[:, 0].round(2),
    "Stability_SHAP": shap_metrics[:, 1].round(2),
    "Stability_SHAP+": shapplus_metrics[:, 1].round(2),
    "Stability_QSHAP+": qshap_metrics[:, 1].round(2),
    "Responsiveness_SHAP": shap_metrics[:, 2].round(2),
    "Responsiveness_SHAP+": shapplus_metrics[:, 2].round(2),
    "Responsiveness_QSHAP+": qshap_metrics[:, 2].round(2),
    "Clarity_SHAP": shap_metrics[:, 3].round(2),
    "Clarity_SHAP+": shapplus_metrics[:, 3].round(2),
    "Clarity_QSHAP+": qshap_metrics[:, 3].round(2),
    "Composite_SHAP": shap_comp_all.round(2),
    "Composite_SHAP+": shapplus_comp_all.round(2),
    "Composite_QSHAP+": qshap_comp_all.round(2)
})

matrix_path = f"../results/interpretability_matrix_54_profiles_{ACTIVE_DATASET}.csv"
interpret_df.to_csv(matrix_path, index=False)
print(f"✓ Saved interpretability matrix → {matrix_path}")

# Simplified version (composite only)
interpret_simple_df = pd.DataFrame({
    "Profile_ID": np.arange(1, len(X_sample) + 1),
    "SHAP_Score": shap_comp_all.round(2),
    "SHAP+_Score": shapplus_comp_all.round(2),
    "Q-SHAP+_Score": qshap_comp_all.round(2)
})

matrix_simple_path = f"../results/interpretability_matrix_54_profiles_simplified_{ACTIVE_DATASET}.csv"
interpret_simple_df.to_csv(matrix_simple_path, index=False)
print(f"✓ Saved simplified matrix → {matrix_simple_path}\n")
print(f"Simplified Interpretability Matrix (first 10 of 54) - {ACTIVE_DATASET}:\n")
print(interpret_simple_df.head(10).to_string(index=False))

# ============================================================================
# STEP 19: TWO-WAY COMPARISONS (SHAP VS Q-SHAP+ ONLY)
# ============================================================================
print("\n[STEP 19] Creating two-way comparisons: SHAP vs Q-SHAP+...")

# Feature attribution comparison
shap_mean = np.mean(np.abs(shap_values), axis=0)
qshap_mean = np.mean(np.abs(qshap_values), axis=0)

attr_comp_df = pd.DataFrame({
    "Feature": features,
    "SHAP": shap_mean.round(4),
    "Q-SHAP+": qshap_mean.round(4)
})
attr_comp_path = f"../results/feature_attribution_comparison_SHAP_vs_QSHAP_{ACTIVE_DATASET}.csv"
attr_comp_df.to_csv(attr_comp_path, index=False)
print(f"✓ Saved attribution data → {attr_comp_path}")

# Visualization
x = np.arange(len(features))
width = 0.35

plt.figure(figsize=(9, 4))
bars1 = plt.bar(x - width/2, shap_mean, width, label="SHAP", color="#54a3f7")
bars2 = plt.bar(x + width/2, qshap_mean, width, label="Q-SHAP+", color="#9b59b6")

for bar in bars1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.005,
             f"{height:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.005,
             f"{height:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.xticks(x, features, fontsize=10, rotation=45, ha="right")
plt.ylabel("Attribution Magnitude", fontsize=11)
plt.xlabel("Features", fontsize=11)
plt.title(f"Feature Attribution: SHAP vs Q-SHAP+ ({ACTIVE_DATASET})", 
          fontsize=13, fontweight="bold")
plt.legend(frameon=False, fontsize=10)
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(f"../results/feature_attribution_comparison_SHAP_vs_QSHAP_{ACTIVE_DATASET}.png",
            dpi=300, bbox_inches="tight")
plt.close()
print(f"✓ Saved plot → ../results/feature_attribution_comparison_SHAP_vs_QSHAP_{ACTIVE_DATASET}.png")
print(attr_comp_df)

# Interpretability metrics comparison
means_shap = np.mean(shap_metrics, axis=0)
std_shap = np.std(shap_metrics, axis=0)
means_qshap = np.mean(qshap_metrics, axis=0)
std_qshap = np.std(qshap_metrics, axis=0)

two_way_df = pd.DataFrame({
    "Metric": metric_names,
    "SHAP_Mean": means_shap.round(2),
    "SHAP_SD": std_shap.round(2),
    "Q-SHAP+_Mean": means_qshap.round(2),
    "Q-SHAP+_SD": std_qshap.round(2)
})
two_way_path = f"../results/interpretability_metrics_SHAP_vs_QSHAP_{ACTIVE_DATASET}.csv"
two_way_df.to_csv(two_way_path, index=False)
print(f"✓ Saved metrics data → {two_way_path}")

x = np.arange(len(metric_names))
plt.figure(figsize=(9, 4))
bars1 = plt.bar(x - width/2, means_shap, width,
                yerr=std_shap, capsize=4,
                label="SHAP", color="#54a3f7", edgecolor="black")
bars2 = plt.bar(x + width/2, means_qshap, width,
                yerr=std_qshap, capsize=4,
                label="Q-SHAP+", color="#9b59b6", edgecolor="black")

for bar in bars1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 1,
             f"{height:.1f}", ha="center", va="bottom",
             fontsize=9, fontweight="bold")
for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 1,
             f"{height:.1f}", ha="center", va="bottom",
             fontsize=9, fontweight="bold")

plt.xticks(x, metric_names, fontsize=11)
plt.ylabel("Metric Value", fontsize=11)
plt.title(f"Interpretability Metrics: SHAP vs Q-SHAP+ ({ACTIVE_DATASET})", 
          fontsize=13, fontweight="bold")
plt.legend(frameon=False, fontsize=10)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"../results/interpretability_metrics_SHAP_vs_QSHAP_{ACTIVE_DATASET}.png",
            dpi=300, bbox_inches="tight")
plt.close()
print(f"✓ Saved plot → ../results/interpretability_metrics_SHAP_vs_QSHAP_{ACTIVE_DATASET}.png")
print(two_way_df)

print("✓ All visualizations and data tables saved")

print("\n" + "=" * 80)
print(f"ANALYSIS COMPLETE: {ACTIVE_DATASET}")
print("=" * 80)
print("\nConfiguration Summary:")
print(f"  Dataset: {ACTIVE_DATASET}")
print(f"  Features: {len(features)} - {features}")
print(f"  Binarization: {config['binarization_strategy']}")
print(f"  Samples: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
print(f"  XGBoost Acc: {xgb_acc:.3f}")
print(f"\nGenerated Outputs:")
print(f"  CSV Files: 14 files")
print(f"    - three_method_comparison_{ACTIVE_DATASET}.csv")
print(f"    - feature_importance_data_{ACTIVE_DATASET}.csv")
print(f"    - validation_metrics_{ACTIVE_DATASET}.csv")
print(f"    - interpretability_metrics_data_{ACTIVE_DATASET}.csv")
print(f"    - table4_feature_attribution_{ACTIVE_DATASET}.csv")
print(f"    - table5_quality_metrics_{ACTIVE_DATASET}.csv")
print(f"    - table7_quantum_effects_{ACTIVE_DATASET}.csv")
print(f"    - quantum_entanglement_matrix_{ACTIVE_DATASET}.csv")
print(f"    - appendix_54_profiles_{ACTIVE_DATASET}.csv")
print(f"    - interpretability_matrix_54_profiles_{ACTIVE_DATASET}.csv")
print(f"    - interpretability_matrix_54_profiles_simplified_{ACTIVE_DATASET}.csv")
print(f"    - feature_attribution_comparison_SHAP_vs_QSHAP_{ACTIVE_DATASET}.csv")
print(f"    - interpretability_metrics_SHAP_vs_QSHAP_{ACTIVE_DATASET}.csv")
print(f"\n  PNG Files: 7 files")
print(f"    - validation_feature_importance_{ACTIVE_DATASET}.png")
print(f"    - interpretability_metrics_comparison_{ACTIVE_DATASET}.png")
print(f"    - quantum_feature_entanglement_{ACTIVE_DATASET}.png")
print(f"    - feature_attribution_comparison_SHAP_vs_QSHAP_{ACTIVE_DATASET}.png")
print(f"    - interpretability_metrics_SHAP_vs_QSHAP_{ACTIVE_DATASET}.png")
print(f"\nAll outputs saved to:")
print(f"  ../models/     - Trained models for {ACTIVE_DATASET}")
print(f"  ../results/    - CSV tables and PNG charts for {ACTIVE_DATASET}")
print("=" * 80)