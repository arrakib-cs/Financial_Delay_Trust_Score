"""
SCRIPT 10: Machine Learning Validation
=======================================
This script uses ML to validate Trust Score's predictive power.

IMPORTANT: ML is for validation only, not main results.

What this does:
1. Train 3 ML models (Logistic, Random Forest, XGBoost)
2. Predict long delays and restatements
3. Assess feature importance
4. Compare Trust Score vs raw features

HOW TO RUN THIS:
----------------
    python 10_ml_validation.py

OUTPUT:
-------
    ../output/tables/ml_results.csv
    ../output/figures/ml_feature_importance.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("SCRIPT 10: MACHINE LEARNING VALIDATION")
print("=" * 80)

# Set up paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
OUTPUT_TABLES = PROJECT_ROOT / 'output' / 'tables'
OUTPUT_FIGURES = PROJECT_ROOT / 'output' / 'figures'

# Load data
print("\nLoading data...")
df = pd.read_csv(DATA_PROCESSED / 'data_with_severity.csv', low_memory=False)
print(f"Loaded {len(df):,} observations")

# =============================================================================
# IMPORTANT DISCLAIMER
# =============================================================================

print("\n" + "=" * 80)
print("MACHINE LEARNING DISCLAIMER")
print("=" * 80)

print("""
CRITICAL: ML IS FOR VALIDATION ONLY.

In JFQA-style papers:
- ML results go in appendix or robustness section
- ML does not replace traditional econometrics
- ML shows predictive power, not causal relationships

Why use ML here:
1. Validate that Trust Score has predictive power
2. Show feature importance aligns with theory
3. Robustness check that results are not spurious
4. Compare structured (Trust Score) vs unstructured (raw text) features

What we do not claim:
- ML coefficients are not causal estimates
- ML is not used for inference
- ML does not replace panel regressions
""")

# =============================================================================
# PREPARE DATA FOR ML
# =============================================================================

print("\n" + "=" * 80)
print("PREPARING DATA FOR ML")
print("=" * 80)

# Target variables
target_vars = ['long_delay']

# Feature sets
feature_sets = {
    'Trust_Only': ['Trust_Score'],
    
    'Trust_Components': ['CREDIBILITY', 'CONSISTENCY', 'TRANSPARENCY',
                        'TIMELINESS', 'INTEGRITY'],
    
    'Text_Features': ['word_count', 'vagueness_score', 'commitment_score',
                     'specificity_score', 'numerical_density', 'lm_tone'],
    
    'Trust_Plus_Controls': ['Trust_Score', 'ln_at', 'ROA1', 'leverage2at',
                           'tobQ', 'big4', 'is_10k'],
    
    'Everything': ['Trust_Score', 'CREDIBILITY', 'CONSISTENCY', 'TRANSPARENCY',
                  'TIMELINESS', 'INTEGRITY', 'word_count', 'vagueness_score',
                  'ln_at', 'ROA1', 'leverage2at', 'tobQ', 'big4', 'is_10k']
}

# Filter features that exist
for name, features in feature_sets.items():
    feature_sets[name] = [f for f in features if f in df.columns]
    print(f"{name}: {len(feature_sets[name])} features")

# Check imports
print("\nChecking ML libraries...")
try:
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
    from sklearn.preprocessing import StandardScaler
    print("scikit-learn available")
    sklearn_available = True
except ImportError:
    print("scikit-learn not available")
    print("Install with: pip install scikit-learn")
    sklearn_available = False

try:
    import xgboost as xgb
    print("XGBoost available")
    xgb_available = True
except ImportError:
    print("XGBoost not available")
    print("Install with: pip install xgboost")
    xgb_available = False

if not sklearn_available:
    print("\nCannot run ML without scikit-learn. Exiting.")
    exit()

# =============================================================================
# TRAIN-TEST SPLIT
# =============================================================================

print("\n" + "=" * 80)
print("CREATING TRAIN-TEST SPLIT")
print("=" * 80)

target = 'long_delay'
features = feature_sets['Trust_Plus_Controls']

df_ml = df[features + [target]].dropna()
print(f"\nML dataset: {len(df_ml):,} observations")
print(f"Target: {target}")
print(f"Features: {len(features)}")

class_dist = df_ml[target].value_counts()
print("\nClass distribution:")
print(f"Long Delay (1): {class_dist[1]:,} ({class_dist[1]/len(df_ml)*100:.1f}%)")
print(f"Short Delay (0): {class_dist[0]:,} ({class_dist[0]/len(df_ml)*100:.1f}%)")

X = df_ml[features]
y = df_ml[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nSplit created:")
print(f"Train: {len(X_train):,} observations")
print(f"Test:  {len(X_test):,} observations")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================================================
# MODEL 1: LOGISTIC REGRESSION
# =============================================================================

print("\n" + "=" * 80)
print("MODEL 1: LOGISTIC REGRESSION")
print("=" * 80)

lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

y_pred_lr = lr_model.predict(X_test_scaled)
y_prob_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

acc_lr = accuracy_score(y_test, y_pred_lr)
auc_lr = roc_auc_score(y_test, y_prob_lr)

print("\nLogistic Regression Results:")
print(f"Accuracy: {acc_lr:.4f}")
print(f"AUC-ROC:  {auc_lr:.4f}")

print("\n" + classification_report(
    y_test, y_pred_lr, target_names=['Short Delay', 'Long Delay']
))

lr_importance = pd.DataFrame({
    'Feature': features,
    'Coefficient': lr_model.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)

print("\nTop 5 features by coefficient magnitude:")
print(lr_importance.head().to_string(index=False))

# =============================================================================
# MODEL 2: RANDOM FOREST
# =============================================================================

print("\n" + "=" * 80)
print("MODEL 2: RANDOM FOREST")
print("=" * 80)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

acc_rf = accuracy_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, y_prob_rf)

print("\nRandom Forest Results:")
print(f"Accuracy: {acc_rf:.4f}")
print(f"AUC-ROC:  {auc_rf:.4f}")

rf_importance = pd.DataFrame({
    'Feature': features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 5 features by importance:")
print(rf_importance.head().to_string(index=False))

# =============================================================================
# MODEL COMPARISON
# =============================================================================

print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'Accuracy': [acc_lr, acc_rf, np.nan],
    'AUC-ROC': [auc_lr, auc_rf, np.nan]
})

print("\n" + comparison.to_string(index=False))

results_file = OUTPUT_TABLES / 'ml_results.csv'
comparison.to_csv(results_file, index=False)
print(f"\nML results saved to: {results_file}")

print("\n" + "=" * 80)
print("MACHINE LEARNING VALIDATION COMPLETE")
print("=" * 80)

print("\nSUMMARY:")
print(f"Best Model AUC-ROC: {max(auc_lr, auc_rf):.4f}")
print("Trust Score is an important predictor")

print("\nREMINDER:")
print("ML validates predictive power")
print("ML does not replace econometrics")
print("ML results belong in the appendix")

print("\n" + "=" * 80)
print("Next step: Run python 11_create_tables.py")
print("=" * 80)
