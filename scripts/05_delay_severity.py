# Build Delay Severity Index
"""
SCRIPT 05: Delay Severity Index
================================
This script constructs the Delay Severity Index combining:
- Realized delay length
- Trust score (inverted)
- Vagueness

HOW TO RUN THIS:
----------------
    python 05_delay_severity.py

OUTPUT:
-------
    ../data/processed/data_with_severity.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("SCRIPT 05: DELAY SEVERITY INDEX")
print("=" * 80)

# Set up paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'

# Load data
print("\nLoading data with trust scores...")
df = pd.read_csv(DATA_PROCESSED / 'data_with_trust_score.csv', low_memory=False)
print(f"Loaded {len(df):,} observations")

# =============================================================================
# CONSTRUCT SEVERITY INDEX
# =============================================================================

print("\n" + "=" * 80)
print("CONSTRUCTING DELAY SEVERITY INDEX")
print("=" * 80)

print("\nFormula:")
print("Severity = α₁ × Delay_Days (normalized)")
print("         + α₂ × (1 - Trust_Score)")
print("         + α₃ × Vagueness (normalized)")

# Normalize each component to 0-1 scale
print("\n1. Normalizing components...")

# Delay days (clip extreme values)
df['delay_days_clipped'] = df['delay_days'].clip(0, 365)  # Cap at 1 year
df['delay_norm'] = (df['delay_days_clipped'] - df['delay_days_clipped'].min()) / \
                   (df['delay_days_clipped'].max() - df['delay_days_clipped'].min())

# Inverse trust (1 - Trust_Score)
df['inverse_trust'] = 1 - df['Trust_Score']

# Vagueness is already normalized
df['vagueness_norm_severity'] = df['vagueness_norm'] / 100  # Convert from 0-100 to 0-1

print("Components normalized")

# Method 1: Equal weights
print("\n2. Computing severity index (equal weights)...")
df['Severity_Index_EW'] = (
    0.40 * df['delay_norm'] +
    0.40 * df['inverse_trust'] +
    0.20 * df['vagueness_norm_severity']
)

print("Equal-weighted Severity Index calculated")
print(f"  Mean: {df['Severity_Index_EW'].mean():.3f}")
print(f"  Std: {df['Severity_Index_EW'].std():.3f}")

# Method 2: Data-driven weights using standardization
print("\n3. Computing severity index (data-driven weights)...")

# Standardize components
severity_components = df[['delay_norm', 'inverse_trust', 'vagueness_norm_severity']].dropna()
scaler = StandardScaler()
components_scaled = scaler.fit_transform(severity_components)

# Calculate variance-weighted combination
variances = severity_components.var()
total_var = variances.sum()
weights = variances / total_var

print("\nDATA-DRIVEN WEIGHTS:")
print("-" * 80)
print(f"  Delay_Days:     {weights['delay_norm']:.3f}")
print(f"  Inverse_Trust:  {weights['inverse_trust']:.3f}")
print(f"  Vagueness:      {weights['vagueness_norm_severity']:.3f}")

# Apply data-driven weights
df['Severity_Index_DD'] = (
    weights['delay_norm'] * df['delay_norm'] +
    weights['inverse_trust'] * df['inverse_trust'] +
    weights['vagueness_norm_severity'] * df['vagueness_norm_severity']
)

print("\nData-driven Severity Index calculated")
print(f"  Mean: {df['Severity_Index_DD'].mean():.3f}")
print(f"  Std: {df['Severity_Index_DD'].std():.3f}")

# Set primary severity index
df['Severity_Index'] = df['Severity_Index_EW']
print("\nUsing Equal-Weighted as primary Severity_Index")

# =============================================================================
# CREATE CATEGORICAL VERSIONS
# =============================================================================

print("\n" + "=" * 80)
print("CREATING CATEGORICAL VARIABLES")
print("=" * 80)

# Tertiles
df['Severity_Tertile'] = pd.qcut(
    df['Severity_Index'], q=3,
    labels=['Low', 'Medium', 'High'], duplicates='drop'
)

print("\nSEVERITY INDEX TERTILES:")
print("-" * 80)
severity_dist = df['Severity_Tertile'].value_counts().sort_index()
for tertile, count in severity_dist.items():
    pct = count / len(df[df['Severity_Tertile'].notna()]) * 100
    mean_sev = df[df['Severity_Tertile'] == tertile]['Severity_Index'].mean()
    mean_delay = df[df['Severity_Tertile'] == tertile]['delay_days'].mean()
    print(f"{tertile:10s}: {count:6,} ({pct:5.1f}%) - Severity: {mean_sev:.3f}, Delay: {mean_delay:.1f} days")

# Binary (High severity)
median_severity = df['Severity_Index'].median()
df['High_Severity'] = (df['Severity_Index'] >= median_severity).astype(int)

# =============================================================================
# VALIDATION & RELATIONSHIPS
# =============================================================================

print("\n" + "=" * 80)
print("VALIDATION CHECKS")
print("=" * 80)

print("\n1. Correlation matrix:")
corr_vars = ['Severity_Index', 'delay_days', 'Trust_Score', 'vagueness_score']
corr_matrix = df[corr_vars].corr()
print(corr_matrix.round(3))

print("\n2. Severity by trust tertile:")
severity_by_trust = df.groupby('Trust_Tertile')['Severity_Index'].agg(['count', 'mean', 'std'])
print(severity_by_trust.round(3))

print("\n3. Mean delay by severity tertile:")
delay_by_severity = df.groupby('Severity_Tertile')['delay_days'].agg(['count', 'mean', 'median', 'std'])
print(delay_by_severity.round(2))

# =============================================================================
# SAVE RESULTS
# =============================================================================

print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

output_file = DATA_PROCESSED / 'data_with_severity.csv'
df.to_csv(output_file, index=False)
print(f"Saved to: {output_file}")

print("\n" + "=" * 80)
print("DELAY SEVERITY INDEX COMPLETE")
print("=" * 80)
print("\nKey variables created:")
print("  Severity_Index (primary)")
print("  Severity_Index_DD (data-driven weights)")
print("  Severity_Tertile (Low/Medium/High)")
print("  High_Severity (binary)")
print("\nNext step: Run python 06_descriptives.py")
print("=" * 80)
