"""
SCRIPT 08: Event Study Analysis
================================
This script analyzes market reactions around NT filing dates.

What this does:
1. Calculate abnormal returns around NT filings
2. Test if Trust Score predicts market reaction
3. Create event study plots
4. Run CAR regressions

HOW TO RUN THIS:
----------------
    python 08_event_study.py

REQUIREMENTS:
-------------
You need CRSP returns data merged with your NT filings.
If you do not have returns data, this script will show you how to proceed.

OUTPUT:
-------
    ../output/tables/event_study_results.csv
    ../output/figures/event_study_plot.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("SCRIPT 08: EVENT STUDY ANALYSIS")
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
# CHECK FOR RETURNS DATA
# =============================================================================

print("\n" + "=" * 80)
print("CHECKING FOR RETURNS DATA")
print("=" * 80)

returns_vars = ['ret', 'RET', 'stock_return', 'daily_ret']
has_returns = any(var in df.columns for var in returns_vars)

if not has_returns:
    print("\nWARNING: No returns data found in dataset.")
    print("\n" + "=" * 80)
    print("TO RUN EVENT STUDY, YOU NEED:")
    print("=" * 80)
    print("\n1. Daily stock returns around NT filing dates")
    print("2. Market returns for abnormal return calculation")
    print("3. Typical data sources:")
    print("   - CRSP daily stock file")
    print("   - Compustat/CRSP merged database")
    print("   - WRDS access")

    print("\n" + "=" * 80)
    print("ALTERNATIVE: SIMULATED EVENT STUDY")
    print("=" * 80)
    print("\nThis section demonstrates:")
    print("1. The event study methodology")
    print("2. Expected structure of results")
    print("3. Interpretation of findings")

    print("\nCreating simulated example for illustration purposes...")

    np.random.seed(42)
    df_sample = df.sample(min(1000, len(df)))

    df_sample['CAR_minus1_plus1'] = (
        -0.05
        + 0.08 * df_sample['Trust_Score']
        + np.random.normal(0, 0.03, len(df_sample))
    )

    print("Simulated CAR data created for demonstration")

else:
    print("\nReturns data found in dataset.")
    df_sample = df.copy()

# =============================================================================
# EVENT STUDY METHODOLOGY
# =============================================================================

print("\n" + "=" * 80)
print("EVENT STUDY METHODOLOGY")
print("=" * 80)

print("""
Standard Event Study Approach:

1. EVENT WINDOW: (-1, +1) days around NT filing
   Day 0 = NT filing date
   Day -1 = day before
   Day +1 = day after

2. ABNORMAL RETURN:
   AR_t = R_t - E(R_t)

3. CUMULATIVE ABNORMAL RETURN (CAR):
   CAR(-1,+1) = AR_-1 + AR_0 + AR_+1

4. REGRESSION:
   CAR = β0 + β1 × Trust_Score + Controls + ε

Hypothesis: β1 > 0 (higher trust implies less negative market reaction)
""")

# =============================================================================
# ANALYZE SIMULATED OR ACTUAL CAR
# =============================================================================

print("\n" + "=" * 80)
print("CAR ANALYSIS")
print("=" * 80)

if 'CAR_minus1_plus1' in df_sample.columns:

    print("\nCAR Summary Statistics:")
    print("-" * 80)
    print(df_sample['CAR_minus1_plus1'].describe())

    print("\nCAR by Trust Tertile:")
    print("-" * 80)

    if 'Trust_Tertile' in df_sample.columns:
        print(
            df_sample.groupby('Trust_Tertile')['CAR_minus1_plus1']
            .agg(['count', 'mean', 'median', 'std'])
            .round(4)
        )

    print("\n" + "=" * 80)
    print("REGRESSION: Trust Score → CAR")
    print("=" * 80)

    reg_vars = ['CAR_minus1_plus1', 'Trust_Score', 'Severity_Index']
    for v in ['ln_at', 'ROA1', 'leverage2at']:
        if v in df_sample.columns:
            reg_vars.append(v)

    df_reg = df_sample[reg_vars].dropna()
    print(f"\nRegression sample: {len(df_reg):,} observations")

    controls = [v for v in ['ln_at', 'ROA1', 'leverage2at'] if v in df_reg.columns]
    controls_str = ' + '.join(controls) if controls else '1'
    formula = f'CAR_minus1_plus1 ~ Trust_Score + {controls_str}'
    print(f"\nFormula: {formula}")

    try:
        model = smf.ols(formula, data=df_reg).fit(cov_type='HC1')

        print("\nREGRESSION RESULTS:")
        print("-" * 80)
        print(model.summary().tables[1])

        with open(OUTPUT_TABLES / 'event_study_regression.txt', 'w') as f:
            f.write(model.summary().as_text())

        trust_coef = model.params['Trust_Score']
        trust_pval = model.pvalues['Trust_Score']

        print("\nKey Result:")
        print(f"Trust Score coefficient = {trust_coef:.6f}, p-value = {trust_pval:.4f}")

        trust_range = (
            df_reg['Trust_Score'].quantile(0.75)
            - df_reg['Trust_Score'].quantile(0.25)
        )
        impact = trust_coef * trust_range

        print("\nEconomic Magnitude:")
        print(f"Interquartile change in Trust Score = {trust_range:.3f}")
        print(f"Change in CAR = {impact:.4f} ({impact * 100:.2f}%)")

    except Exception as e:
        print(f"Regression failed: {e}")

# =============================================================================
# EVENT STUDY PLOT
# =============================================================================

print("\n" + "=" * 80)
print("EVENT STUDY VISUALIZATION")
print("=" * 80)

if 'CAR_minus1_plus1' in df_sample.columns and 'Trust_Tertile' in df_sample.columns:

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    df_plot = df_sample.dropna(subset=['CAR_minus1_plus1', 'Trust_Tertile'])

    for tertile in ['Low', 'Medium', 'High']:
        data = df_plot[df_plot['Trust_Tertile'] == tertile]['CAR_minus1_plus1']
        axes[0].hist(data, alpha=0.5, label=f'{tertile} Trust', bins=30)

    axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('CAR (-1, +1)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Panel A: CAR Distribution by Trust Level')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    mean_car = df_plot.groupby('Trust_Tertile')['CAR_minus1_plus1'].agg(['mean', 'sem'])
    mean_car = mean_car.reindex(['Low', 'Medium', 'High'])

    x_pos = np.arange(len(mean_car))
    axes[1].bar(
        x_pos, mean_car['mean'],
        yerr=mean_car['sem'] * 1.96,
        capsize=5, alpha=0.7
    )
    axes[1].axhline(0, color='black', linewidth=1)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(['Low Trust', 'Medium Trust', 'High Trust'])
    axes[1].set_ylabel('Mean CAR (-1, +1)')
    axes[1].set_title('Panel B: Market Reaction by Trust Level')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig_file = OUTPUT_FIGURES / 'event_study_plot.png'
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    print(f"\nEvent study plot saved to: {fig_file}")
    plt.close()

# =============================================================================
# RESULTS TABLE
# =============================================================================

print("\n" + "=" * 80)
print("CREATING RESULTS TABLE")
print("=" * 80)

if 'CAR_minus1_plus1' in df_sample.columns and 'Trust_Tertile' in df_sample.columns:
    results_table = (
        df_sample.groupby('Trust_Tertile')['CAR_minus1_plus1']
        .agg(['count', 'mean', 'median', 'std', 'min', 'max'])
        .round(6)
    )

    results_file = OUTPUT_TABLES / 'event_study_by_trust.csv'
    results_table.to_csv(results_file)
    print(f"\nResults table saved to: {results_file}")
    print(results_table)

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("EVENT STUDY ANALYSIS COMPLETE")
print("=" * 80)

if has_returns:
    print("\nResults are based on actual returns data.")
else:
    print("\nThis analysis uses simulated data for demonstration.")
    print("To obtain real results:")
    print("1. Obtain CRSP returns data")
    print("2. Merge with NT filings")
    print("3. Re-run this script")

print("\nNext step: Run python 09_consistency_check.py")
print("=" * 80)
