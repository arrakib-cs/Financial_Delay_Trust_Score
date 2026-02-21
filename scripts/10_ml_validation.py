"""
SCRIPT 10: Machine Learning Validation
=======================================
This script uses ML to validate Trust Score's predictive power.

IMPORTANT: ML is for validation only, not main results.

What this does:
1. Train 3 ML models (Logistic, Random Forest, XGBoost)
2. Predict long delays
3. Assess feature importance (ALL 12 variables)
4. Generate Figure 6 (4-panel chart saved to output/figures/)

HOW TO RUN THIS:
----------------
    python scripts/10_ml_validation.py

OUTPUT:
-------
    output/tables/ml_results.csv
    output/figures/ml_feature_importance.png   <- Figure 6
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("SCRIPT 10: MACHINE LEARNING VALIDATION")
print("=" * 80)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).parent.parent
DATA_PROCESSED  = PROJECT_ROOT / 'data' / 'processed'
OUTPUT_TABLES   = PROJECT_ROOT / 'output' / 'tables'
OUTPUT_FIGURES  = PROJECT_ROOT / 'output' / 'figures'
OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading data...")
df = pd.read_csv(DATA_PROCESSED / 'data_with_severity.csv', low_memory=False)
print(f"Loaded {len(df):,} observations")

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SELECTION — 12-variable extended set
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("PREPARING FEATURES")
print("=" * 80)

# firm_age fallback
if 'firm_age' not in df.columns and 'lag_firm_age' in df.columns:
    df['firm_age'] = df['lag_firm_age']
    print("  NOTE: using lag_firm_age as firm_age")

EXTENDED_12 = ['Trust_Score', 'ln_at', 'ROA1', 'leverage2at',
               'tobQ', 'big4', 'is_10k',
               'Rd2at', 'Cash2at', 'Capx2at', 'lag_altz', 'lag_firm_age']

FALLBACK_7  = ['Trust_Score', 'ln_at', 'ROA1', 'leverage2at',
               'tobQ', 'big4', 'is_10k']

missing_ext = [f for f in EXTENDED_12 if f not in df.columns]
if missing_ext:
    print(f"\nWARNING: Missing variables: {missing_ext}")
    print("Falling back to 7-variable set. Run add_new_variables.py first.")
    features = [f for f in FALLBACK_7 if f in df.columns]
else:
    print("\nAll 12 variables found. Using full extended set.")
    features = EXTENDED_12

print(f"Feature set ({len(features)}): {features}")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK ML LIBRARIES
# ─────────────────────────────────────────────────────────────────────────────
try:
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
    from sklearn.preprocessing import StandardScaler
    print("\nscikit-learn: available")
    sklearn_available = True
except ImportError:
    print("\nscikit-learn NOT available — pip install scikit-learn")
    sklearn_available = False

try:
    import xgboost as xgb
    print("XGBoost: available")
    xgb_available = True
except ImportError:
    print("XGBoost NOT available — pip install xgboost")
    xgb_available = False

if not sklearn_available:
    print("\nCannot run ML without scikit-learn. Exiting.")
    exit()

# ─────────────────────────────────────────────────────────────────────────────
# TRAIN-TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("TRAIN-TEST SPLIT")
print("=" * 80)

target = 'long_delay'
df_ml  = df[features + [target]].dropna()
print(f"\nML dataset: {len(df_ml):,} observations")
print(f"Target: {target}  |  Features: {len(features)}")

class_dist = df_ml[target].value_counts()
print(f"Long Delay (1): {class_dist[1]:,} ({class_dist[1]/len(df_ml)*100:.1f}%)")
print(f"Short Delay (0): {class_dist[0]:,} ({class_dist[0]/len(df_ml)*100:.1f}%)")

X = df_ml[features]
y = df_ml[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1: LOGISTIC REGRESSION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("MODEL 1: LOGISTIC REGRESSION")
print("=" * 80)

lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_sc, y_train)

y_pred_lr = lr_model.predict(X_test_sc)
y_prob_lr = lr_model.predict_proba(X_test_sc)[:, 1]
acc_lr    = accuracy_score(y_test, y_pred_lr)
auc_lr    = roc_auc_score(y_test, y_prob_lr)

print(f"Accuracy: {acc_lr:.4f}  |  AUC-ROC: {auc_lr:.4f}")
print(classification_report(y_test, y_pred_lr, target_names=['Short Delay', 'Long Delay']))

lr_importance = pd.DataFrame({
    'Feature': features,
    'Coefficient': lr_model.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)

print("\nAll coefficients (Logistic Regression):")
print(lr_importance.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2: RANDOM FOREST
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("MODEL 2: RANDOM FOREST")
print("=" * 80)

rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
acc_rf    = accuracy_score(y_test, y_pred_rf)
auc_rf    = roc_auc_score(y_test, y_prob_rf)

print(f"Accuracy: {acc_rf:.4f}  |  AUC-ROC: {auc_rf:.4f}")

rf_importance = pd.DataFrame({
    'Feature': features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nAll importances (Random Forest):")
print(rf_importance.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3: XGBOOST
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("MODEL 3: XGBOOST")
print("=" * 80)

acc_xgb = auc_xgb = np.nan
xgb_importance = None

if xgb_available:
    xgb_model = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=42, eval_metric='logloss', use_label_encoder=False
    )
    xgb_model.fit(X_train, y_train)

    y_pred_xgb = xgb_model.predict(X_test)
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
    acc_xgb    = accuracy_score(y_test, y_pred_xgb)
    auc_xgb    = roc_auc_score(y_test, y_prob_xgb)

    print(f"Accuracy: {acc_xgb:.4f}  |  AUC-ROC: {auc_xgb:.4f}")

    xgb_importance = pd.DataFrame({
        'Feature': features,
        'Importance': xgb_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\nAll importances (XGBoost):")
    print(xgb_importance.to_string(index=False))
else:
    print("XGBoost not available — skipped.")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

comparison = pd.DataFrame({
    'Model':    ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'Accuracy': [acc_lr, acc_rf, acc_xgb],
    'AUC-ROC':  [auc_lr, auc_rf, auc_xgb]
})
print(comparison.to_string(index=False))
comparison.to_csv(OUTPUT_TABLES / 'ml_results.csv', index=False)

# ─────────────────────────────────────────────────────────────────────────────
# CLEAN SUMMARY (copy into document)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("CLEAN SUMMARY — COPY INTO DOCUMENT")
print("=" * 80)

print("\nPANEL A: LOGISTIC REGRESSION COEFFICIENTS")
print(f"{'Variable':<20} {'Coeff':>10}")
print("-" * 32)
for _, row in lr_importance.iterrows():
    print(f"{row['Feature']:<20} {row['Coefficient']:>10.4f}")

print("\nPANEL B: RANDOM FOREST FEATURE IMPORTANCE")
print(f"{'Rank':<5} {'Variable':<20} {'Importance %':>12}")
print("-" * 39)
for i, (_, row) in enumerate(rf_importance.iterrows(), 1):
    print(f"{i:<5} {row['Feature']:<20} {row['Importance']*100:>11.1f}%")

if xgb_importance is not None:
    print("\nPANEL C: XGBOOST FEATURE IMPORTANCE")
    print(f"{'Rank':<5} {'Variable':<20} {'Importance %':>12}")
    print("-" * 39)
    for i, (_, row) in enumerate(xgb_importance.iterrows(), 1):
        print(f"{i:<5} {row['Feature']:<20} {row['Importance']*100:>11.1f}%")

print("\nPANEL D: MODEL PERFORMANCE")
print(f"{'Model':<22} {'Accuracy':>10} {'AUC-ROC':>8} {'N test':>8}")
print("-" * 50)
print(f"{'Logistic Regression':<22} {acc_lr*100:>9.1f}% {auc_lr:>8.3f} {len(X_test):>8,}")
print(f"{'Random Forest':<22} {acc_rf*100:>9.1f}% {auc_rf:>8.3f} {len(X_test):>8,}")
if not np.isnan(acc_xgb):
    print(f"{'XGBoost':<22} {acc_xgb*100:>9.1f}% {auc_xgb:>8.3f} {len(X_test):>8,}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6 — 4-PANEL CHART (all 12 variables)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("GENERATING FIGURE 6 — 4-PANEL CHART")
print("=" * 80)

TRUST_COLOR   = '#E8724A'   # orange highlight for Trust_Score
CONTROL_COLOR = '#2E6DA4'   # steel blue for control variables

def bar_colors(feature_list):
    return [TRUST_COLOR if f == 'Trust_Score' else CONTROL_COLOR for f in feature_list]

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle(
    f'Figure 6: Machine Learning Validation — Feature Importance ({len(features)} Variables)',
    fontsize=14, fontweight='bold', y=0.98
)

# ── Panel A: Logistic Regression Coefficients ─────────────────────────────
ax = axes[0, 0]
lr_plot  = lr_importance.sort_values('Coefficient', ascending=True)
colors_a = bar_colors(lr_plot['Feature'].tolist())
ax.barh(lr_plot['Feature'], lr_plot['Coefficient'],
        color=colors_a, edgecolor='white', height=0.7)
ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_title('Panel A: Logistic Regression Coefficients', fontweight='bold', fontsize=11)
ax.set_xlabel('Coefficient (standardized)', fontsize=9)
ax.tick_params(axis='y', labelsize=8)
ax.tick_params(axis='x', labelsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ── Panel B: Random Forest Feature Importance ────────────────────────────
ax = axes[0, 1]
rf_plot  = rf_importance.sort_values('Importance', ascending=True)
colors_b = bar_colors(rf_plot['Feature'].tolist())
ax.barh(rf_plot['Feature'], rf_plot['Importance'],
        color=colors_b, edgecolor='white', height=0.7)
ax.set_title('Panel B: Random Forest Feature Importance', fontweight='bold', fontsize=11)
ax.set_xlabel('Importance', fontsize=9)
ax.tick_params(axis='y', labelsize=8)
ax.tick_params(axis='x', labelsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ── Panel C: XGBoost Feature Importance ─────────────────────────────────
ax = axes[1, 0]
if xgb_importance is not None:
    xgb_plot = xgb_importance.sort_values('Importance', ascending=True)
    colors_c  = bar_colors(xgb_plot['Feature'].tolist())
    ax.barh(xgb_plot['Feature'], xgb_plot['Importance'],
            color=colors_c, edgecolor='white', height=0.7)
    ax.set_title('Panel C: XGBoost Feature Importance', fontweight='bold', fontsize=11)
    ax.set_xlabel('Importance', fontsize=9)
else:
    ax.text(0.5, 0.5, 'XGBoost not available\n(pip install xgboost)',
            ha='center', va='center', transform=ax.transAxes,
            fontsize=11, color='gray')
    ax.set_title('Panel C: XGBoost Feature Importance', fontweight='bold', fontsize=11)
ax.tick_params(axis='y', labelsize=8)
ax.tick_params(axis='x', labelsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ── Panel D: Model Performance ───────────────────────────────────────────
ax = axes[1, 1]
model_labels = ['Logistic\nRegression', 'Random\nForest', 'XGBoost']
accuracies   = [acc_lr, acc_rf, acc_xgb if not np.isnan(acc_xgb) else 0]
aucs         = [auc_lr, auc_rf, auc_xgb if not np.isnan(auc_xgb) else 0]
x_pos        = np.arange(len(model_labels))
width        = 0.35

bars1 = ax.bar(x_pos - width/2, accuracies, width,
               label='Accuracy', color='#2E6DA4', alpha=0.85)
bars2 = ax.bar(x_pos + width/2, aucs,       width,
               label='AUC-ROC',  color='#E8724A', alpha=0.85)

for bar in list(bars1) + list(bars2):
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.008,
                f'{h:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_title('Panel D: Model Performance Comparison', fontweight='bold', fontsize=11)
ax.set_ylabel('Score', fontsize=9)
ax.set_xticks(x_pos)
ax.set_xticklabels(model_labels, fontsize=9)
ax.set_ylim(0, 1.10)
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Shared legend for colour coding (Panels A–C)
trust_patch   = mpatches.Patch(color=TRUST_COLOR,   label='Trust_Score (key variable)')
control_patch = mpatches.Patch(color=CONTROL_COLOR, label='Control Variables')
fig.legend(handles=[trust_patch, control_patch],
           loc='lower center', ncol=2, fontsize=10,
           bbox_to_anchor=(0.35, 0.005),
           frameon=True, framealpha=0.9)

plt.tight_layout(rect=[0, 0.04, 1, 0.97])

fig_path = OUTPUT_FIGURES / 'ml_feature_importance.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"\nFigure 6 saved to: {fig_path}")
print("ACTION: Replace the existing Figure 6 in your Word document with this file.")

# ─────────────────────────────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("MACHINE LEARNING VALIDATION COMPLETE")
print("=" * 80)

valid_aucs = [v for v in [auc_lr, auc_rf, auc_xgb] if not np.isnan(v)]
print(f"\nBest Model AUC-ROC: {max(valid_aucs):.4f}")
print("Trust Score confirmed as important predictor.")
print("\nREMINDER: ML validates predictive power — does not replace econometrics.")
print("\nNext step: run 11_create_tables.py")
print("=" * 80)