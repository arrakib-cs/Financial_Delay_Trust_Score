# Ex-ante vs ex-post comparison
"""
SCRIPT 09: Ex-Ante vs Ex-Post Consistency Check
================================================
This script tests whether high-trust NT explanations are validated by outcomes.

What this does:
1. Compare NT explanation to actual outcomes (delay realized)
2. Test if high-trust NTs actually file sooner
3. Analyze text consistency between NT and final 10-K/10-Q
4. Identify trust violations (high trust but long delay)

HOW TO RUN THIS:
----------------
    python 09_consistency_check.py

OUTPUT:
-------
    ../output/tables/consistency_analysis.csv
    ../output/figures/trust_validation.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("SCRIPT 09: EX-ANTE VS EX-POST CONSISTENCY")
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
# CONCEPT: EX-ANTE TRUST VS EX-POST OUTCOMES
# =============================================================================

print("\n" + "=" * 80)
print("CONSISTENCY FRAMEWORK")
print("=" * 80)

print("""
EX-ANTE TRUST VALIDATION:

High Trust NT should predict:
1. Shorter actual delays (ex-post validation)
2. Higher probability of meeting promised deadline
3. Fewer subsequent restatements
4. Consistent explanation in final filing

Low Trust NT might indicate:
1. Vague explanation leading to worse outcomes
2. Blame-shifting resulting in longer delays
3. Inconsistencies between NT and final filing

TRUST VIOLATION:
High ex-ante trust combined with long ex-post delay
Indicates either unexpected shocks or misleading NT disclosure
""")

# =============================================================================
# ANALYSIS 1: TRUST SCORE VS ACTUAL DELAY
# =============================================================================

print("\n" + "=" * 80)
print("ANALYSIS 1: TRUST VALIDATION BY ACTUAL DELAY")
print("=" * 80)

print("\nCreating validation categories...")

df['trust_category'] = pd.cut(
    df['Trust_Score'],
    bins=[0, 0.4, 0.6, 1.0],
    labels=['Low_Trust', 'Medium_Trust', 'High_Trust']
)

df['delay_category'] = pd.cut(
    df['delay_days'],
    bins=[-np.inf, 5, 30, np.inf],
    labels=['Short_Delay', 'Medium_Delay', 'Long_Delay']
)

print("\nTrust vs Actual Delay (Cross-tabulation):")
print("-" * 80)

crosstab = pd.crosstab(
    df['trust_category'],
    df['delay_category'],
    margins=True,
    normalize='index'
) * 100

print(crosstab.round(1))

df['trust_violation'] = (
    (df['Trust_Score'] > df['Trust_Score'].quantile(0.75)) &
    (df['delay_days'] > df['delay_days'].quantile(0.75))
).astype(int)

violation_rate = df['trust_violation'].mean() * 100
print(f"\nTrust Violation Rate: {violation_rate:.2f}%")
print("High trust combined with long delay")

# =============================================================================
# ANALYSIS 2: PREDICTION ACCURACY
# =============================================================================

print("\n" + "=" * 80)
print("ANALYSIS 2: TRUST SCORE PREDICTION ACCURACY")
print("=" * 80)

df['high_trust'] = (df['Trust_Score'] > df['Trust_Score'].median()).astype(int)
df['short_delay'] = (df['delay_days'] <= df['delay_days'].median()).astype(int)

from sklearn.metrics import confusion_matrix

y_true = df['short_delay'].dropna()
y_pred = df.loc[y_true.index, 'high_trust']

cm = confusion_matrix(y_true, y_pred)

print("\nPrediction Confusion Matrix:")
print("-" * 80)
print("                    Predicted")
print("                Low Trust  High Trust")
print(f"Actual Long Delay    {cm[0,0]:6d}      {cm[0,1]:6d}")
print(f"Actual Short Delay   {cm[1,0]:6d}      {cm[1,1]:6d}")

accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0

print("\nPrediction Metrics:")
print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")

# =============================================================================
# ANALYSIS 3: PROMISED VS ACTUAL DEADLINE
# =============================================================================

print("\n" + "=" * 80)
print("ANALYSIS 3: PROMISED VS ACTUAL DEADLINE")
print("=" * 80)

print("""
METHODOLOGY:

To fully implement this analysis:

1. Extract promised filing dates from NT narratives
2. Compare promised and actual filing dates
3. Construct Promise_Kept indicator
4. Test whether Trust Score predicts Promise_Kept
""")

# =============================================================================
# ANALYSIS 4: TRUST SCORE BY OUTCOME
# =============================================================================

print("\n" + "=" * 80)
print("ANALYSIS 4: MEAN TRUST BY OUTCOME")
print("=" * 80)

outcome_groups = df.groupby('delay_category')['Trust_Score'].agg(
    ['count', 'mean', 'median', 'std']
).round(3)

print("\nTrust Score by Delay Outcome:")
print(outcome_groups)

from scipy import stats

short = df[df['delay_category'] == 'Short_Delay']['Trust_Score'].dropna()
long = df[df['delay_category'] == 'Long_Delay']['Trust_Score'].dropna()

if len(short) > 0 and len(long) > 0:
    t_stat, p_value = stats.ttest_ind(short, long)

    print("\nT-Test (Short vs Long Delay):")
    print(f"Mean Trust (Short): {short.mean():.3f}")
    print(f"Mean Trust (Long):  {long.mean():.3f}")
    print(f"Difference:         {short.mean() - long.mean():.3f}")
    print(f"t-statistic:        {t_stat:.3f}")
    print(f"p-value:            {p_value:.4f}")

# =============================================================================
# VISUALIZATION: TRUST VALIDATION
# =============================================================================

print("\n" + "=" * 80)
print("CREATING VALIDATION VISUALIZATIONS")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for category in ['Short_Delay', 'Medium_Delay', 'Long_Delay']:
    data = df[df['delay_category'] == category]['Trust_Score'].dropna()
    axes[0, 0].hist(data, alpha=0.5, label=category, bins=30)

axes[0, 0].set_xlabel('Trust Score')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Panel A: Trust Distribution by Delay Outcome')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

df['trust_decile'] = pd.qcut(df['Trust_Score'], q=10, labels=False, duplicates='drop')
validation = df.groupby('trust_decile').agg({
    'short_delay': 'mean',
    'Trust_Score': 'mean'
}).reset_index()

axes[0, 1].plot(
    validation['Trust_Score'],
    validation['short_delay'] * 100,
    marker='o',
    linewidth=2
)

axes[0, 1].set_xlabel('Trust Score (by decile)')
axes[0, 1].set_ylabel('Percent with Short Delay')
axes[0, 1].set_title('Panel B: Validation Rate by Trust Level')
axes[0, 1].grid(alpha=0.3)

plt.tight_layout()
fig_file = OUTPUT_FIGURES / 'trust_validation.png'
plt.savefig(fig_file, dpi=300, bbox_inches='tight')
print(f"\nValidation plot saved to: {fig_file}")
plt.close()

# =============================================================================
# CREATE CONSISTENCY TABLE
# =============================================================================

print("\n" + "=" * 80)
print("CREATING CONSISTENCY SUMMARY TABLE")
print("=" * 80)

consistency_summary = pd.DataFrame({
    'Metric': [
        'Overall Accuracy',
        'Precision (High Trust to Short Delay)',
        'Recall (Short Delay to High Trust)',
        'Trust Violation Rate (%)',
        'Mean Trust (Short Delay)',
        'Mean Trust (Long Delay)',
        'Trust Difference (Short minus Long)',
        'T-test p-value'
    ],
    'Value': [
        f'{accuracy:.3f}',
        f'{precision:.3f}',
        f'{recall:.3f}',
        f'{violation_rate:.2f}',
        f'{short.mean():.3f}' if len(short) > 0 else 'N/A',
        f'{long.mean():.3f}' if len(long) > 0 else 'N/A',
        f'{short.mean() - long.mean():.3f}' if len(short) > 0 and len(long) > 0 else 'N/A',
        f'{p_value:.4f}' if len(short) > 0 and len(long) > 0 else 'N/A'
    ]
})

table_file = OUTPUT_TABLES / 'consistency_analysis.csv'
consistency_summary.to_csv(table_file, index=False)
print(f"\nConsistency table saved to: {table_file}")
print(consistency_summary)

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("CONSISTENCY CHECK COMPLETE")
print("=" * 80)

print("\nKey Findings:")
print(f"Trust Score Accuracy: {accuracy:.1%}")
print(f"Trust Violations: {violation_rate:.1f}%")
print(f"Precision (High Trust to Short Delay): {precision:.1%}")

print("\nNext step: Run python 10_ml_validation.py")
print("=" * 80)
