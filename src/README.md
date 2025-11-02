# Three-Method Explainability Analysis: SHAP vs SHAP+ vs Q-SHAP+

**Author:** Pranav Sanghadia  
**Email:** psanghadia@captechu.edu  
**Institution:** Capitol Technology University  
**Date:** October 2025

---

## 📋 Project Overview

This repository contains the complete implementation of a comparative analysis of three explainability methods for credit default prediction models:

1. **SHAP** - Standard correlational attribution using Shapley values
2. **SHAP+** - Causal attribution via controlled interventions on classical models
3. **Q-SHAP+** - Causal attribution via controlled interventions on quantum neural networks

### Research Hypothesis

**H₀:** The quantum explainability method (Q-SHAP+) provides equal or lower interpretability quality compared to classical SHAP.

**H₁:** The quantum explainability method (Q-SHAP+) provides significantly higher interpretability quality compared to classical SHAP (α = 0.05).

---

## Key Features

- **Binary Feature Engineering** - Converts continuous features to quantum-compatible binary states
- **Dual Model Architecture** - XGBoost (classical) and Quantum Neural Network
- **Four Interpretability Metrics** - Faithfulness, Stability, Responsiveness, Clarity
- **Statistical Hypothesis Testing** - Paired t-tests with Cohen's d effect size
- **Quantum Phenomena Detection** - Entanglement and interference analysis
- **Comprehensive Visualizations** - 8+ publication-ready charts
- **Complete Data Export** - All numerical results saved as CSV tables

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Required Libraries

```bash
pip install numpy pandas scikit-learn xgboost shap pennylane matplotlib seaborn openpyxl joblib scipy
```

### Optional: GPU Acceleration

For faster quantum circuit simulation:

```bash
pip install pennylane-lightning[gpu]
```

---

## Project Structure

```
project_root/
│
├── qshap+_framework.py  # Main analysis script
├── README.md                           # This file
│
├── data/
│   └── default_of_credit_card_clients.xlsx  # Input dataset (UCI)
│
├── models/                             # Generated model files
│   ├── xgb_model_complete.joblib
│   └── quantum_model_complete.joblib
│
└── results/                            # Generated outputs
    ├── *.csv                           # Data tables
    └── *.png                           # Visualization charts
```

---

## How to Usae

### Basic Execution

```bash
python complete_three_method_optimized.py
```

### Expected Runtime

- **Without cached quantum model:** 3-5 minutes (includes training)
- **With cached quantum model:** 30-60 seconds (loads from disk)

### Input Data

Place the UCI Credit Card Default dataset at:
```
../data/default_of_credit_card_clients.xlsx
```

**Dataset Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

---

## Output Files

### Models (Saved to `../models/`)

| File | Description |
|------|-------------|
| `xgb_model_complete.joblib` | Trained XGBoost classifier |
| `quantum_model_complete.joblib` | Trained quantum neural network |

### Data Tables (Saved to `../results/`)

| File | Contents |
|------|----------|
| `three_method_comparison.csv` | Summary metrics for all three methods |
| `validation_metrics_data.csv` | Accuracy and AUC comparison |
| `interpretability_metrics_data.csv` | Four metrics with mean and SD |
| `feature_importance_data.csv` | Attribution magnitudes per feature |
| `table4_feature_attribution.csv` | **Thesis Table 4** - Feature comparison |
| `table5_quality_metrics.csv` | **Thesis Table 5** - Quality metrics |
| `table7_quantum_effects.csv` | **Thesis Table 7** - Quantum phenomena |
| `appendix_54_profiles.csv` | Detailed per-profile attributions |
| `interpretability_matrix_54_profiles.csv` | Full interpretability scores |
| `interpretability_matrix_54_profiles_simplified.csv` | Composite scores only |
| `quantum_entanglement_matrix.csv` | Feature entanglement correlations |
| `feature_attribution_comparison_SHAP_vs_QSHAP.csv` | Two-way attribution data |
| `interpretability_metrics_SHAP_vs_QSHAP.csv` | Two-way metrics data |

### Visualizations (Saved to `../results/`)

| File | Description |
|------|-------------|
| `validation_feature_importance_comparison.png` | Model performance + feature importance |
| `interpretability_metrics_comparison.png` | Three-way metric comparison (with error bars) |
| `quantum_feature_entanglement.png` | Heatmap of quantum correlations |
| `feature_attribution_comparison_SHAP_vs_QSHAP.png` | Two-way attribution bars |
| `interpretability_metrics_SHAP_vs_QSHAP.png` | Two-way metrics bars |

---

## Methodology

### Feature Engineering

Five binary features selected for quantum compatibility:
- **LIMIT_BAL** - Credit limit (low/high)
- **Age** - Customer age (young/old)
- **PAY_AMT1** - Payment amount (low/high)
- **EDUCATION** - Education level (below/above university)
- **MARRIAGE** - Marital status (married/other)

### Data Split

- **Training:** 70% (model training)
- **Validation:** 15% (hyperparameter tuning)
- **Test:** 15% (final evaluation)
  - 54 profiles sampled for explainability analysis

### Interpretability Metrics

1. **Faithfulness (0-100%)** - Alignment between attribution and actual prediction change
2. **Stability (0-100%)** - Consistency across similar profiles
3. **Responsiveness (0-100%)** - Sensitivity to feature interventions
4. **Clarity (0-5)** - Simplicity of attribution pattern

**Composite Score:**
```
Score = 0.30×Faithfulness + 0.30×Stability + 0.20×Responsiveness + 0.20×Clarity
```

### Statistical Testing

- **Test:** Paired t-test (two-tailed)
- **Significance level:** α = 0.05
- **Effect size:** Cohen's d (small: 0.2, medium: 0.5, large: 0.8)
- **Power:** Post-hoc power analysis performed

---

## Quantum Circuit Architecture

```
Input Layer:     RY(π/2 × x_i) for each feature
Entanglement:    CRY(θ_i) between feature and ancilla qubit
Measurement:     ⟨Z⟩ on ancilla qubit
Classification:  P(Y=1) = (1 - ⟨Z⟩) / 2
```

**Training:**
- Optimizer: Adam (learning rate = 0.05)
- Loss: Binary cross-entropy
- Epochs: 150
- Training samples: 1,000 (subset for efficiency)

---

## Key Results

### Composite Interpretability Scores (Mean ± SD)

| Method | Score | Statistical Significance |
|--------|-------|--------------------------|
| SHAP | 65.43 ± 8.21 | Baseline |
| SHAP+ | 68.77 ± 7.95 | p = 0.042 vs SHAP |
| **Q-SHAP+** | **71.28 ± 7.44** | **p = 0.003 vs SHAP** ✓ |

### Hypothesis Test Result

**Conclusion:** Reject H₀ at α = 0.05. Q-SHAP+ demonstrates statistically significant improvement over classical SHAP (Δ = 5.85, d = 0.72, p < 0.01).

---

## Troubleshooting

### Common Issues

**1. "PennyLane not installed" warning**

```bash
pip install pennylane
```

**2. "File not found" error**

Ensure dataset is at `../data/default_of_credit_card_clients.xlsx`

**3. Slow quantum training**

- Normal for first run (3-5 minutes)
- Subsequent runs load cached model instantly
- Delete `../models/quantum_model_complete.joblib` to retrain

**4. Memory errors**

Reduce quantum training subset:
```python
subset_size = 500  # Line 158 (default: 1000)
```

---

## References

### Dataset

Yeh, I. C., & Lien, C. H. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients.* Expert Systems with Applications, 36(2), 2473-2480.

### Explainability Methods

- Lundberg, S. M., & Lee, S. I. (2017). *A unified approach to interpreting model predictions.* NeurIPS.
- Sanghadia, P. (2025). *Quantum-enhanced explainability for credit risk models.* Capitol Technology University.

### Quantum Computing

- Schuld, M., Bergholm, V., Gogolin, C., Izaac, J., & Killoran, N. (2019). *Evaluating analytic gradients on quantum hardware.* Physical Review A, 99(3), 032331.

---

## Author

**Pranav Sanghadia**  
Masters of Research in Quantum Computing  
Capitol Technology University  
Email: psanghadia@captechu.edu

---

## License

This code is provided for academic and research purposes. Please cite appropriately if used in publications.

---

## Acknowledgments

- Capitol Technology University
- UCI Machine Learning Repository
- PennyLane Development Team
- XGBoost and SHAP Library Maintainers

---

## Citation

If you use this code in your research, please cite:

```bibtex
@thesis{sanghadia2025qshap,
  author       = {Sanghadia, Pranav K.},
  title        = {Quantum-Enhanced Explainable AI (Q-SHAP⁺) for Credit Risk Assessment},
  school       = {Capitol Technology University},
  year         = {2025},
  type         = {Master of Research Thesis},
  address      = {Laurel, Maryland, USA},
  month        = {Oct},
  note         = {MRes in Quantum Computing},
}
```

---

**Last Updated:** October 2025  
**Version:** 1.0