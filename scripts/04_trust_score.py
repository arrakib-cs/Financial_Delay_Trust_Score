"""
SCRIPT 04: Trust Score Construction
====================================
This script combines the 5 trust components into final Trust Score.

METHODS:
1. Equal weights (baseline)
2. PCA weights (robustness)

HOW TO RUN THIS:
----------------
    python 04_trust_score.py

OUTPUT:
-------
    ../data/processed/data_with_trust_score.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("SCRIPT 04: TRUST SCORE CONSTRUCTION")
print("=" * 80)

# Set up paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
OUTPUT_TABLES = PROJECT_ROOT / 'output' / 'tables'

# Load data with trust components
print("\nLoading data with trust components...")
df = pd.read_csv(DATA_PROCESSED / 'data_with_trust_components.csv', low_memory=False)
print(f"Loaded {len(df):,} observations")

# =============================================================================
# METHOD 1: EQUAL WEIGHTS (BASELINE)
# =============================================================================

print("\n" + "=" * 80)
print("METHOD 1: EQUAL WEIGHTS")
print("=" * 80)

components = ['CREDIBILITY', 'CONSISTENCY', 'TRANSPARENCY', 'TIMELINESS', 'INTEGRITY']

print("\nFormula:")
print("Trust_Score_EW = 0.20 × CREDIBILITY")
print("               + 0.20 × CONSISTENCY")
print("               + 0.20 × TRANSPARENCY")
print("               + 0.20 × TIMELINESS")
print("               + 0.20 × INTEGRITY")

# Calculate equal-weighted trust score
df['Trust_Score_EW'] = (
    0.20 * df['CREDIBILITY'] +
    0.20 * df['CONSISTENCY'] +
    0.20 * df['TRANSPARENCY'] +
    0.20 * df['TIMELINESS'] +
    0.20 * df['INTEGRITY']
)

print("\nEqual-weighted Trust Score calculated")
print(f"  Mean: {df['Trust_Score_EW'].mean():.3f}")
print(f"  Median: {df['Trust_Score_EW'].median():.3f}")
print(f"  Std: {df['Trust_Score_EW'].std():.3f}")
print(f"  Min: {df['Trust_Score_EW'].min():.3f}")
print(f"  Max: {df['Trust_Score_EW'].max():.3f}")

# Distribution
print("\nTRUST SCORE DISTRIBUTION:")
print("-" * 80)
bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
df['trust_bin'] = pd.cut(df['Trust_Score_EW'], bins=bins, labels=labels)
dist = df['trust_bin'].value_counts().sort_index()
for bin_label, count in dist.items():
    pct = count / len(df) * 100
    print(f"{bin_label:10s}: {count:6,} ({pct:5.1f}%)")

# =============================================================================
# METHOD 2: PCA WEIGHTS (ROBUSTNESS)
# =============================================================================

print("\n" + "=" * 80)
print("METHOD 2: PCA WEIGHTS")
print("=" * 80)

print("\nThis method uses Principal Component Analysis to determine")
print("optimal weights based on the variance structure of components.")

# Prepare data for PCA (drop missing values)
pca_data = df[components].dropna()
print(f"\nUsing {len(pca_data):,} observations for PCA (dropped {len(df) - len(pca_data):,} with missing)")

# Standardize components
print("\n1. Standardizing components...")
scaler = StandardScaler()
components_scaled = scaler.fit_transform(pca_data)

# Perform PCA
print("2. Performing PCA...")
pca = PCA(n_components=len(components))
pca_result = pca.fit_transform(components_scaled)

# Print explained variance
print("\nPCA RESULTS:")
print("-" * 80)
print("Explained variance by component:")
for i, var in enumerate(pca.explained_variance_ratio_, 1):
    cum_var = pca.explained_variance_ratio_[:i].sum()
    print(f"  PC{i}: {var:6.2%} (Cumulative: {cum_var:6.2%})")

# Get weights from first principal component
pc1_weights = pca.components_[0]

# Normalize weights to sum to 1 and ensure all positive
pc1_weights_abs = np.abs(pc1_weights)
pc1_weights_norm = pc1_weights_abs / pc1_weights_abs.sum()

print("\nPCA WEIGHTS (from PC1):")
print("-" * 80)
weight_df = pd.DataFrame({
    'Component': components,
    'Weight': pc1_weights_norm,
    'Equal_Weight': [0.20] * 5,
    'Difference': pc1_weights_norm - 0.20
})
print(weight_df.round(4))

# Calculate PCA-weighted trust score
print("\n3. Calculating PCA-weighted Trust Score...")

df['Trust_Score_PCA'] = np.nan

for idx in pca_data.index:
    df.loc[idx, 'Trust_Score_PCA'] = sum(
        df.loc[idx, comp] * weight 
        for comp, weight in zip(components, pc1_weights_norm)
    )

print("PCA-weighted Trust Score calculated")
print(f"  Mean: {df['Trust_Score_PCA'].mean():.3f}")
print(f"  Median: {df['Trust_Score_PCA'].median():.3f}")
print(f"  Std: {df['Trust_Score_PCA'].std():.3f}")

# Correlation between two methods
corr = df[['Trust_Score_EW', 'Trust_Score_PCA']].corr().iloc[0, 1]
print(f"\nCorrelation between EW and PCA scores: {corr:.4f}")

# =============================================================================
# SET PRIMARY TRUST SCORE
# =============================================================================

print("\n" + "=" * 80)
print("SETTING PRIMARY TRUST SCORE")
print("=" * 80)

df['Trust_Score'] = df['Trust_Score_EW']
print("Using Equal-Weighted as primary Trust_Score")
print("PCA version saved as Trust_Score_PCA for robustness checks")

# =============================================================================
# CREATE CATEGORICAL VERSIONS
# =============================================================================

print("\n" + "=" * 80)
print("CREATING CATEGORICAL VARIABLES")
print("=" * 80)

df['Trust_Tertile'] = pd.qcut(df['Trust_Score'], q=3, labels=['Low', 'Medium', 'High'])

print("\nTRUST SCORE TERTILES:")
print("-" * 80)
tertile_dist = df['Trust_Tertile'].value_counts().sort_index()
for tertile, count in tertile_dist.items():
    pct = count / len(df) * 100
    mean_score = df[df['Trust_Tertile'] == tertile]['Trust_Score'].mean()
    print(f"{tertile:10s}: {count:6,} ({pct:5.1f}%) - Mean Score: {mean_score:.3f}")

df['Trust_Quartile'] = pd.qcut(
    df['Trust_Score'], q=4,
    labels=['Q1_Lowest', 'Q2', 'Q3', 'Q4_Highest']
)

median_trust = df['Trust_Score'].median()
df['High_Trust'] = (df['Trust_Score'] >= median_trust).astype(int)

print(f"\nBinary High_Trust created (threshold = median = {median_trust:.3f})")
print(f"  High Trust: {df['High_Trust'].sum():,} ({df['High_Trust'].mean()*100:.1f}%)")
print(f"  Low Trust: {(1-df['High_Trust']).sum():,} ({(1-df['High_Trust']).mean()*100:.1f}%)")

# =============================================================================
# VALIDATION CHECKS
# =============================================================================

print("\n" + "=" * 80)
print("VALIDATION CHECKS")
print("=" * 80)

print("\n1. Checking relationship: Trust Score vs Delay Days")
corr_delay = df[['Trust_Score', 'delay_days']].corr().iloc[0, 1]
print(f"   Correlation: {corr_delay:.4f}")
print("   Expected: Negative (higher trust leads to shorter delays)")

print("\n2. Mean delay by trust tertile:")
delay_by_tertile = df.groupby('Trust_Tertile')['delay_days'].agg(['count', 'mean', 'median'])
print(delay_by_tertile.round(2))

print("\n3. Trust score by filing type:")
trust_by_type = df.groupby('is_10k')['Trust_Score'].agg(['count', 'mean', 'std'])
trust_by_type.index = ['10-Q', '10-K']
print(trust_by_type.round(3))

# =============================================================================
# SAVE RESULTS
# =============================================================================

print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

output_file = DATA_PROCESSED / 'data_with_trust_score.csv'
df.to_csv(output_file, index=False)
print(f"Saved main dataset to: {output_file}")

OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
weight_df.to_csv(OUTPUT_TABLES / 'trust_score_weights.csv', index=False)
print(f"Saved weights table to: {OUTPUT_TABLES / 'trust_score_weights.csv'}")

summary_stats = pd.DataFrame({
    'Variable': ['Trust_Score_EW', 'Trust_Score_PCA'] + components,
    'N': [df[var].count() for var in ['Trust_Score_EW', 'Trust_Score_PCA'] + components],
    'Mean': [df[var].mean() for var in ['Trust_Score_EW', 'Trust_Score_PCA'] + components],
    'Median': [df[var].median() for var in ['Trust_Score_EW', 'Trust_Score_PCA'] + components],
    'Std': [df[var].std() for var in ['Trust_Score_EW', 'Trust_Score_PCA'] + components],
    'Min': [df[var].min() for var in ['Trust_Score_EW', 'Trust_Score_PCA'] + components],
    'Max': [df[var].max() for var in ['Trust_Score_EW', 'Trust_Score_PCA'] + components]
})

summary_stats.to_csv(OUTPUT_TABLES / 'trust_components_summary.csv', index=False)
print(f"Saved summary stats to: {OUTPUT_TABLES / 'trust_components_summary.csv'}")

print("\n" + "=" * 80)
print("TRUST SCORE CONSTRUCTION COMPLETE")
print("=" * 80)
print("\nKey variables created:")
print("  Trust_Score (primary, equal-weighted)")
print("  Trust_Score_PCA (robustness)")
print("  Trust_Tertile (Low/Medium/High)")
print("  Trust_Quartile (Q1/Q2/Q3/Q4)")
print("  High_Trust (binary)")
print("\nNext step: Run python 05_delay_severity.py")
print("=" * 80)
