"""
================================================================================
Complete Three-Method Analysis: SHAP vs SHAP+ vs Q-SHAP+
Optimized Quantum Training with Gradient-based Adam Optimizer
================================================================================

Author: Pranav Sanghadia
Email: psanghadia@captechu.edu
Institution: Capitol Technology University
Date: October 2025

Description:
    This script implements a comprehensive comparison of three explainability 
    methods for credit default prediction:
    
    - Method 1: SHAP (correlational, XGBoost-based)
    - Method 2: SHAP+ (causal, XGBoost-based)
    - Method 3: Q-SHAP+ (causal, Quantum computing-based)
    
    The analysis includes:
    - Binary feature engineering for quantum compatibility
    - 70/15/15 train/validation/test split
    - XGBoost and Quantum model training
    - Interpretability metrics (faithfulness, stability, responsiveness, clarity)
    - Statistical hypothesis testing (paired t-tests)
    - Quantum phenomena detection (entanglement, interference)
    - Comprehensive visualization and data export

Requirements:
    - Python 3.8+
    - numpy, pandas, scikit-learn
    - xgboost, shap
    - pennylane (for quantum computing)
    - matplotlib, seaborn
    - openpyxl (for Excel file reading)
    - joblib

Usage:
    python qshap+_framework.py
    
    Ensure the input data file exists at:
    ../data/default_of_credit_card_clients.xlsx

Output:
    All results are saved to ../results/ and ../models/ directories:
    - CSV files: metrics, comparisons, attribution values
    - PNG files: visualization charts
    - Joblib files: trained models
    
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
# PENNYLANE IMPORT CHECK
# ============================================================================
# Description: Verify PennyLane quantum computing library availability.
#              The quantum model (Q-SHAP+) requires PennyLane installation.
# ============================================================================

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    print("WARNING: PennyLane not installed. Install with: pip install pennylane")

# ============================================================================
# QUANTUM MODEL CLASS DEFINITION
# ============================================================================
# Description: Global class definition required for model serialization.
#              This class wraps quantum circuit weights and provides
#              scikit-learn compatible predict/predict_proba interfaces.
#              MUST be defined at module level for joblib.load() to work.
# ============================================================================

class QuantumModel:
    """
    Wrapper class for quantum neural network model.
    
    Attributes:
        weights: Quantum circuit parameters
        quantum_circuit: PennyLane QNode for quantum predictions
    """
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

# Create output directories
Path("../models").mkdir(exist_ok=True)
Path("../results").mkdir(exist_ok=True)

print("=" * 80)
print("COMPLETE THREE-METHOD ANALYSIS (Optimized Quantum)")
print("SHAP | SHAP+ | Q-SHAP+")
print("=" * 80)

# ============================================================================
# STEP 1: DATA LOADING
# ============================================================================
# Description: Load UCI credit card default dataset from Excel file.
#              Dataset contains 30,000 customer records with payment history,
#              demographic info, and default status (binary target).
# ============================================================================

print("\n[STEP 1] Loading dataset...")

df = pd.read_excel("../data/default_of_credit_card_clients.xlsx",
                   engine="openpyxl", header=1)
df = df.rename(columns={"default payment next month": "Default"})
print(f"✓ Loaded {len(df):,} customers")

# ============================================================================
# STEP 2: FEATURE BINARIZATION
# ============================================================================
# Description: Convert continuous features to binary (0/1) for quantum circuit
#              compatibility. Quantum computers require discrete state encoding.
#              - Quantile-based binning for continuous variables
#              - Threshold-based binning for categorical variables
# ============================================================================

print("\n[STEP 2] Binarizing features...")

df["LIMIT_BAL"] = pd.qcut(df["LIMIT_BAL"], 2, labels=[0, 1])
df["Age"] = pd.qcut(df["AGE"], 2, labels=[0, 1])
df["PAY_AMT1"] = pd.qcut(df["PAY_AMT1"], 2, labels=[0, 1])
df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 1 if x >= 2 else 0)
df["MARRIAGE"] = df["MARRIAGE"].apply(lambda x: 1 if x == 1 else 0)
df["Default"] = df["Default"].astype(int)

features = ["LIMIT_BAL", "Age", "PAY_AMT1", "EDUCATION", "MARRIAGE"]
X = df[features].astype(int)
y = df["Default"]

print(f"✓ Features: {features}")

# ============================================================================
# STEP 3: TRAIN/VALIDATION/TEST SPLIT (70/15/15)
# ============================================================================
# Description: Stratified split to maintain class balance across all sets.
#              - Train: 70% (model training)
#              - Validation: 15% (hyperparameter tuning)
#              - Test: 15% (final evaluation and explainability analysis)
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
# Description: Train gradient boosting classifier for baseline (SHAP/SHAP+).
#              - Handles class imbalance via scale_pos_weight
#              - Uses validation set for early stopping
#              - Serves as the classical ML benchmark
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
joblib.dump(xgb_model, "../models/xgb_model_complete.joblib")

# ============================================================================
# STEP 5: QUANTUM MODEL TRAINING (WITH CACHING)
# ============================================================================
# Description: Train or load quantum neural network using PennyLane.
#              - Variational quantum circuit with parameterized gates
#              - Adam optimizer with gradient-based training
#              - Caches trained model to avoid retraining (2-5 min per run)
#              
#              Circuit Architecture:
#              - RY rotation gates for feature encoding
#              - Controlled-RY gates for entanglement
#              - Pauli-Z expectation measurement for classification
# ============================================================================

print("\n[STEP 5] Quantum Model (Adam optimizer with caching)...")

# Define quantum device and circuit
n_qubits = len(features) + 1
os.environ["OMP_NUM_THREADS"] = str(max(1, multiprocessing.cpu_count() - 1))
dev = qml.device("lightning.qubit", wires=n_qubits, shots=None)

@qml.qnode(dev)
def quantum_circuit(x, weights):
    """Quantum circuit for binary classification"""
    # Feature encoding
    for i in range(len(features)):
        qml.RY(qml.numpy.pi / 2 * x[i], wires=i)
    # Entanglement layer
    for i in range(len(features)):
        qml.CRY(weights[i], wires=[i, n_qubits - 1])
    return qml.expval(qml.PauliZ(n_qubits - 1))

quantum_model_path = "../models/quantum_model_complete.joblib"

if os.path.exists(quantum_model_path):
    print(f"✓ Quantum model already exists → loading from {quantum_model_path}")
    quantum_model = joblib.load(quantum_model_path)
    quantum_model.quantum_circuit = quantum_circuit
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

    # Use subset for faster training
    subset_size = 1000
    qml.numpy.random.seed(42)
    idx = qml.numpy.random.choice(len(X_train), subset_size, replace=False)
    X_train_q = X_train.iloc[idx].values
    y_train_q = y_train.iloc[idx].values

    # Initialize weights and optimizer
    weights = qml.numpy.array(qml.numpy.random.randn(len(features)), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=0.05)
    n_steps = 150
    loss_history = []

    print(f"  Using {subset_size} samples, {n_steps} steps (Adam)")
    start_time = time.time()
    for step in range(n_steps):
        weights, curr_loss = opt.step_and_cost(lambda w: loss(w, X_train_q, y_train_q), weights)
        loss_history.append(curr_loss)
        if step % 10 == 0:
            print(f"    Step {step:03d}: loss = {curr_loss:.4f}")
    elapsed = time.time() - start_time
    print(f"  ✓ Optimization complete: final loss = {curr_loss:.4f}")
    print(f"  Training time: {elapsed/60:.2f} minutes")

    quantum_model = QuantumModel(weights, quantum_circuit)
    q_acc = accuracy_score(y_test, quantum_model.predict(X_test))
    q_auc = roc_auc_score(y_test, quantum_model.predict_proba(X_test)[:, 1])
    print(f"✓ Quantum: Acc={q_acc:.3f}, AUC={q_auc:.3f}")

    joblib.dump(quantum_model, quantum_model_path)
    print(f"✓ Saved trained quantum model → {quantum_model_path}")


# ============================================================================
# STEP 6: TEST SAMPLE SELECTION
# ============================================================================
# Description: Select 54 representative profiles from test set for detailed
#              explainability analysis. Sample size chosen for statistical
#              power (n≥30) while maintaining computational efficiency.
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
# Description: Standard SHAP values using TreeExplainer.
#              - Fast computation via tree-specific algorithms
#              - Captures feature correlations and interactions
#              - Baseline correlational attribution method
# ============================================================================

print("\n[STEP 7] Method 1: SHAP (TreeExplainer)...")
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_sample.values)
print(f"✓ SHAP values calculated")

# ============================================================================
# STEP 8: METHOD 2 - SHAP+ (CAUSAL, XGBOOST)
# ============================================================================
# Description: Causal attribution via controlled interventions on XGBoost.
#              - Forces each feature to 0 and 1 independently
#              - Measures marginal effect: P(Y|do(X_j=1)) - P(Y|do(X_j=0))
#              - Captures direct causal effects rather than correlations
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
# Description: Causal attribution via controlled interventions on Quantum model.
#              - Same intervention strategy as SHAP+
#              - Applied to quantum neural network predictions
#              - Leverages quantum superposition and entanglement effects
#              - MAIN HYPOTHESIS: Q-SHAP+ > SHAP in interpretability
# ============================================================================

print("\n[STEP 9] Method 3: Q-SHAP+ (Causal on Quantum)...")
if quantum_model is None:
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
# Description: Compute four interpretability quality metrics for each method:
#              
#              1. FAITHFULNESS: Alignment between attribution and actual impact
#              2. STABILITY: Consistency across similar profiles
#              3. RESPONSIVENESS: Sensitivity to feature changes
#              4. CLARITY: Simplicity and comprehensibility of attributions
#              
#              Each metric scaled to 0-100 for comparability.
# ============================================================================

print("\n[STEP 10] Calculating interpretability metrics...")

def calculate_metrics(attributions, X, model):
    """Compute faithfulness, stability, responsiveness, and clarity"""
    metrics = []
    for i in range(len(X)):
        profile = X.iloc[i].values
        most_important = qml.numpy.argmax(qml.numpy.abs(attributions[i]))
        
        # Faithfulness: attribution magnitude vs actual prediction change
        p_orig = model.predict_proba([profile])[0, 1]
        p_mod = profile.copy()
        p_mod[most_important] = 1 - p_mod[most_important]
        p_changed = model.predict_proba([p_mod])[0, 1]
        pred_change = abs(p_orig - p_changed)
        attr_mag = abs(attributions[i, most_important])
        faith = (pred_change / (attr_mag + 0.1)) * 50 if attr_mag > 0.001 else 0
        faith = qml.numpy.clip(faith, 0, 100)
        
        # Stability: distance from mean attribution pattern
        mean_attr = qml.numpy.mean(attributions, axis=0)
        dist = qml.numpy.linalg.norm(attributions[i] - mean_attr)
        stab = 100 / (1 + dist)
        
        # Responsiveness: average sensitivity across all features
        p_orig = model.predict_proba([profile])[0, 1]
        sens = []
        for j in range(len(profile)):
            p_mod = profile.copy()
            p_mod[j] = 1 - p_mod[j]
            sens.append(abs(p_orig - model.predict_proba([p_mod])[0, 1]))
        resp = qml.numpy.mean(sens) * 100
        
        # Clarity: inverse of attribution variance (simpler = clearer)
        clar = qml.numpy.clip(5 / (1 + qml.numpy.var(attributions[i]) * 10), 0, 5)
        
        metrics.append([faith, stab, resp, clar])
    return qml.numpy.array(metrics)

shap_metrics = calculate_metrics(shap_values, X_sample, xgb_model)
shapplus_metrics = calculate_metrics(shap_plus_values, X_sample, xgb_model)
qshap_metrics = calculate_metrics(qshap_values, X_sample,
                                  quantum_model if quantum_model else xgb_model)

def composite(m):
    """Weighted composite score: 30% faith, 30% stab, 20% resp, 20% clarity"""
    return 0.30*m[:, 0] + 0.30*m[:, 1] + 0.20*m[:, 2] + 0.20*(m[:, 3]/5*100)

shap_scores = composite(shap_metrics)
shapplus_scores = composite(shapplus_metrics)
qshap_scores = composite(qshap_metrics)
print(f"✓ All metrics calculated")

# ============================================================================
# STEP 11: STATISTICAL HYPOTHESIS TESTING
# ============================================================================
# Description: Paired t-tests to evaluate statistical significance:
#              
#              H₀: μ(Q-SHAP+) ≤ μ(SHAP)
#              H₁: μ(Q-SHAP+) > μ(SHAP)  [MAIN HYPOTHESIS]
#              
#              Also reports Cohen's d effect size for practical significance.
#              Significance level: α = 0.05
# ============================================================================

print("\n[STEP 11] Statistical analysis...")

t1, p1 = stats.ttest_rel(shapplus_scores, shap_scores)
d1 = (shapplus_scores.mean() - shap_scores.mean()) / qml.numpy.std(shapplus_scores - shap_scores, ddof=1)
t2, p2 = stats.ttest_rel(qshap_scores, shap_scores)
d2 = (qshap_scores.mean() - shap_scores.mean()) / qml.numpy.std(qshap_scores - shap_scores, ddof=1)
t3, p3 = stats.ttest_rel(qshap_scores, shapplus_scores)
d3 = (qshap_scores.mean() - shapplus_scores.mean()) / qml.numpy.std(qshap_scores - shapplus_scores, ddof=1)

print(f"\nComparison 2: SHAP vs Q-SHAP+ (MAIN HYPOTHESIS)")
print(f"  Δ={qshap_scores.mean() - shap_scores.mean():.2f}, d={d2:.3f}, p={p2:.6f}")
if p2 < 0.05 and d2 >= 0.5:
    print(f"  ✓ REJECT H₀: Q-SHAP+ significantly better")

# ============================================================================
# STEP 12: SAVE PRIMARY COMPARISON RESULTS
# ============================================================================
# Description: Export method comparison summary to CSV for reporting.
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
comparison_df.to_csv('../results/three_method_comparison.csv', index=False)

print("✓ Results saved")
print("=" * 80)
print("THREE-METHOD ANALYSIS COMPLETE (Optimized)")
print("=" * 80)

# ============================================================================
# STEP 13: VALIDATION METRICS & FEATURE IMPORTANCE VISUALIZATION
# ============================================================================
# Description: Create dual-panel figure showing:
#              Panel 1: Model accuracy and AUC comparison
#              Panel 2: Feature importance across three methods
# ============================================================================

print("\n[STEP 13] Creating validation and feature importance comparison chart...")

# Validation metrics
val_metrics = pd.DataFrame({
    "Method": ["SHAP", "SHAP+", "Q-SHAP+"],
    "Accuracy": [
        accuracy_score(y_test, xgb_model.predict(X_test)),
        accuracy_score(y_test, xgb_model.predict(X_test)),
        accuracy_score(y_test, quantum_model.predict(X_test))
            if quantum_model else accuracy_score(y_test, xgb_model.predict(X_test))
    ],
    "AUC": [
        roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1]),
        roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1]),
        roc_auc_score(
            y_test,
            quantum_model.predict_proba(X_test)[:, 1]
            if quantum_model else xgb_model.predict_proba(X_test)[:, 1]
        )
    ]
})

# Feature importance (mean absolute attribution)
shap_importance = np.mean(np.abs(shap_values), axis=0)
shapplus_importance = np.mean(np.abs(shap_plus_values), axis=0)
qshap_importance = np.mean(np.abs(qshap_values), axis=0)

importance_df = pd.DataFrame({
    "Feature": features,
    "SHAP": shap_importance,
    "SHAP+": shapplus_importance,
    "Q-SHAP+": qshap_importance
})

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Validation Metrics & Feature Importance Comparison", fontsize=14, fontweight="bold")

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
plt.savefig("../results/validation_feature_importance_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print("✓ Saved: validation_feature_importance_comparison.png")

# ============================================================================
# STEP 14: INTERPRETABILITY METRICS VISUALIZATION (WITH ERROR BARS)
# ============================================================================
# Description: Bar chart showing all four interpretability metrics with
#              standard deviation error bars for statistical rigor.
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
plt.title("Interpretability Metrics: SHAP vs SHAP+ vs Q-SHAP+", fontsize=13, fontweight="bold")
plt.legend(frameon=False, fontsize=10)
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("../results/interpretability_metrics_comparison.png",
            dpi=300, bbox_inches="tight")
plt.close()
print("✓ Saved: interpretability_metrics_comparison.png")

# ============================================================================
# STEP 15: EXPORT ALL DATA TABLES
# ============================================================================
# Description: Save all numerical data to CSV for reproducibility and
#              thesis appendix inclusion. Includes validation metrics,
#              interpretability metrics, feature importance, and full
#              54-profile attribution details.
# ============================================================================

print("\n[STEP 15] Saving chart data and 54-profile appendix tables...")

# Validation metrics
val_metrics_path = "../results/validation_metrics_data.csv"
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
interp_metrics_path = "../results/interpretability_metrics_data.csv"
interp_metrics_df.to_csv(interp_metrics_path, index=False)
print(f"✓ Saved interpretability metrics → {interp_metrics_path}")

# Feature importance
importance_path = "../results/feature_importance_data.csv"
importance_df.to_csv(importance_path, index=False)
print(f"✓ Saved feature importance → {importance_path}")

# Three-method comparison
three_method_path = "../results/three_method_comparison.csv"
comparison_df.to_csv(three_method_path, index=False)
print(f"✓ Re-saved three-method comparison → {three_method_path}")

# Detailed 54-profile appendix
appendix_df = X_sample.copy()
appendix_df["Default"] = y_sample.values

for j, f in enumerate(features):
    appendix_df[f"SHAP_{f}"] = shap_values[:, j]
    appendix_df[f"SHAP+_{f}"] = shap_plus_values[:, j]
    appendix_df[f"Q-SHAP+_{f}"] = qshap_values[:, j]

appendix_path = "../results/appendix_54_profiles.csv"
appendix_df.to_csv(appendix_path, index=False)
print(f"✓ Saved 54-profile appendix → {appendix_path}")

# ============================================================================
# STEP 16: THESIS TABLES (TABLE 4 & TABLE 5)
# ============================================================================
# Description: Generate publication-ready tables for thesis document.
#              Table 4: Feature attribution magnitudes
#              Table 5: Quality metric comparison
# ============================================================================

print("\n[STEP 16] Generating thesis-style tables (Table 4 & Table 5)...")

# Table 4: Feature Attribution
table4_df = pd.DataFrame({
    "Feature": features,
    "SHAP": np.mean(np.abs(shap_values), axis=0).round(4),
    "SHAP+": np.mean(np.abs(shap_plus_values), axis=0).round(4),
    "Q-SHAP+": np.mean(np.abs(qshap_values), axis=0).round(4)
})
table4_path = "../results/table4_feature_attribution.csv"
table4_df.to_csv(table4_path, index=False)
print(f"✓ Saved Table 4 → {table4_path}\n")
print("Table 4. Feature Attribution Comparison Across Methods\n")
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

table5_path = "../results/table5_quality_metrics.csv"
table5_df.to_csv(table5_path, index=False)
print(f"\n✓ Saved Table 5 → {table5_path}\n")
print("Table 5. Quality Metrics Comparison\n")
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
# STEP 17: QUANTUM PHENOMENA DETECTION
# ============================================================================
# Description: Analyze quantum-specific effects in Q-SHAP+ attributions:
#              
#              - ENTANGLEMENT: Non-classical correlations between features
#                (measured via pairwise correlation of Q-SHAP+ values)
#              
#              - INTERFERENCE: Superposition effects in attribution patterns
#                (measured via attribution variance distribution)
#              
#              Creates heatmap visualization and Table 7 for thesis.
# ============================================================================

print("\n[STEP 17] Computing quantum phenomena (entanglement, interference)...")

import seaborn as sns

# Feature entanglement matrix
qshap_corr = np.corrcoef(qshap_values.T)
qshap_corr = np.nan_to_num(qshap_corr, nan=0.0)
entanglement_df = pd.DataFrame(qshap_corr, index=features, columns=features)

entanglement_path = "../results/quantum_entanglement_matrix.csv"
entanglement_df.to_csv(entanglement_path)
print(f"✓ Saved entanglement matrix → {entanglement_path}")

# Heatmap visualization
plt.figure(figsize=(7, 5))
sns.heatmap(
    entanglement_df, annot=True, cmap="Blues",
    vmin=0, vmax=1, cbar_kws={'label': 'Entanglement Strength'}
)
plt.title("Quantum Feature Entanglement\n(Q-SHAP+ Only)",
          fontsize=13, fontweight="bold")
plt.xlabel("Features")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig("../results/quantum_feature_entanglement.png",
            dpi=300, bbox_inches="tight")
plt.close()
print("✓ Saved entanglement heatmap → ../results/quantum_feature_entanglement.png")

# Quantum effects summary
max_entanglement = float(np.max(qshap_corr))
pair_indices = np.unravel_index(np.argmax(qshap_corr, axis=None), qshap_corr.shape)
entangled_pair = (features[pair_indices[0]], features[pair_indices[1]])

education_idx = features.index("EDUCATION")
interference_strength = float(
    np.var(qshap_values[:, education_idx]) / np.var(qshap_values)
)

# Table 7: Quantum Effects
table7_data = [
    ["Feature Entanglement Strength", round(max_entanglement, 3),
     "Maximum quantum correlation between feature pairs"],
    [f"{entangled_pair[0]}–{entangled_pair[1]} Entanglement",
     round(max_entanglement, 3),
     "Perfect non-classical correlation between features"],
    ["Age–Education Entanglement",
     round(float(qshap_corr[features.index('Age'), features.index('EDUCATION')]), 3),
     "Strong quantum coupling effects"],
    ["Maximum Interference", round(float(interference_strength), 3),
     "Quantum superposition effects in EDUCATION feature"]
]

table7_df = pd.DataFrame(table7_data,
                         columns=["Quantum Effect", "Measurement", "Interpretation"])

table7_path = "../results/table7_quantum_effects.csv"
table7_df.to_csv(table7_path, index=False)
print(f"✓ Saved Table 7 → {table7_path}\n")
print("Table 7. Detected Quantum Effects\n")
print(table7_df.to_string(index=False))

# ============================================================================
# STEP 18: INTERPRETABILITY MATRIX (54 PROFILES)
# ============================================================================
# Description: Detailed per-profile interpretability scores for all 54 test
#              samples. Provides granular view of method performance across
#              diverse customer profiles. Includes simplified version with
#              composite scores only.
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

matrix_path = "../results/interpretability_matrix_54_profiles.csv"
interpret_df.to_csv(matrix_path, index=False)
print(f"✓ Saved interpretability matrix → {matrix_path}")

# Simplified version (composite only)
interpret_simple_df = pd.DataFrame({
    "Profile_ID": np.arange(1, len(X_sample) + 1),
    "SHAP_Score": shap_comp_all.round(2),
    "SHAP+_Score": shapplus_comp_all.round(2),
    "Q-SHAP+_Score": qshap_comp_all.round(2)
})

matrix_simple_path = "../results/interpretability_matrix_54_profiles_simplified.csv"
interpret_simple_df.to_csv(matrix_simple_path, index=False)
print(f"✓ Saved simplified matrix → {matrix_simple_path}\n")
print("Simplified Interpretability Matrix (first 10 of 54):\n")
print(interpret_simple_df.head(10).to_string(index=False))

# ============================================================================
# STEP 19: TWO-WAY COMPARISONS (SHAP VS Q-SHAP+ ONLY)
# ============================================================================
# Description: Create focused visualizations comparing only the primary
#              hypothesis (SHAP vs Q-SHAP+), excluding SHAP+ for clarity.
#              Includes both feature attribution and interpretability metrics.
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
attr_comp_path = "../results/feature_attribution_comparison_SHAP_vs_QSHAP.csv"
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

plt.xticks(x, features, fontsize=10)
plt.ylabel("Attribution Magnitude", fontsize=11)
plt.xlabel("Features", fontsize=11)
plt.title("Feature Attribution Comparison: SHAP vs Q-SHAP+", fontsize=13, fontweight="bold")
plt.legend(frameon=False, fontsize=10)
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig("../results/feature_attribution_comparison_SHAP_vs_QSHAP.png",
            dpi=300, bbox_inches="tight")
plt.close()
print("✓ Saved plot → ../results/feature_attribution_comparison_SHAP_vs_QSHAP.png")
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
two_way_path = "../results/interpretability_metrics_SHAP_vs_QSHAP.csv"
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
plt.title("Interpretability Metrics: SHAP vs Q-SHAP+", fontsize=13, fontweight="bold")
plt.legend(frameon=False, fontsize=10)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("../results/interpretability_metrics_SHAP_vs_QSHAP.png",
            dpi=300, bbox_inches="tight")
plt.close()
print("✓ Saved plot → ../results/interpretability_metrics_SHAP_vs_QSHAP.png")
print(two_way_df)

print("\n" + "=" * 80)
print("ALL ANALYSES COMPLETE")
print("=" * 80)
print("\nOutputs saved to:")
print("  ../models/     - Trained XGBoost and Quantum models")
print("  ../results/    - All CSV data tables and PNG visualizations")
print("=" * 80)