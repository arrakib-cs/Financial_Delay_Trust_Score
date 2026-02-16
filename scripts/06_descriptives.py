"""
SCRIPT 06: Descriptive Statistics
==================================
This script creates all descriptive statistics tables for the paper.

Tables created:
1. Summary statistics (Table 1)
2. Correlation matrix
3. Trust score by year
4. Trust score by industry
5. Variable definitions

HOW TO RUN THIS:
----------------
    python 06_descriptives.py

OUTPUT:
-------
    ../output/tables/table1_summary_stats.csv
    ../output/tables/correlation_matrix.csv
    ../output/figures/trust_distribution.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("=" * 80)
print("SCRIPT 06: DESCRIPTIVE STATISTICS")
print("=" * 80)

# Set up paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
OUTPUT_TABLES = PROJECT_ROOT / 'output' / 'tables'
OUTPUT_FIGURES = PROJECT_ROOT / 'output' / 'figures'

# Create output directories
OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)

# Load data
print("\nLoading data...")
df = pd.read_csv(DATA_PROCESSED / 'data_with_severity.csv', low_memory=False)
print(f"Loaded {len(df):,} observations")

# =============================================================================
# TABLE 1: SUMMARY STATISTICS
# =============================================================================

print("\n" + "=" * 80)
print("TABLE 1: SUMMARY STATISTICS")
print("=" * 80)

# Define variable groups
vars_main = {
    'Dependent Variables': ['delay_days', 'long_delay'],
    'Trust Variables': ['Trust_Score', 'CREDIBILITY', 'CONSISTENCY', 'TRANSPARENCY',
                       'TIMELINESS', 'INTEGRITY', 'Severity_Index'],
    'Text Features': ['word_count', 'vagueness_score', 'commitment_score', 'specificity_score'],
    'Firm Controls': ['at', 'MKCAP', 'ROA1', 'leverage2at', 'tobQ', 'firm_age'],
    'Other': ['big4', 'is_10k']
}

# Create summary statistics
summary_list = []

for category, vars_list in vars_main.items():
    print(f"\n{category}:")
    print("-" * 80)
    
    for var in vars_list:
        if var in df.columns:
            stats = {
                'Category': category,
                'Variable': var,
                'N': df[var].count(),
                'Mean': df[var].mean(),
                'Median': df[var].median(),
                'Std': df[var].std(),
                'Min': df[var].min(),
                'P25': df[var].quantile(0.25),
                'P75': df[var].quantile(0.75),
                'Max': df[var].max()
            }
            summary_list.append(stats)
            print(f"  {var:20s}: N={stats['N']:6,}, Mean={stats['Mean']:8.2f}, Std={stats['Std']:8.2f}")

summary_df = pd.DataFrame(summary_list)

# Save Table 1
table1_file = OUTPUT_TABLES / 'table1_summary_stats.csv'
summary_df.to_csv(table1_file, index=False)
print(f"\nSaved Table 1 to: {table1_file}")

# =============================================================================
# TABLE 2: CORRELATION MATRIX
# =============================================================================

print("\n" + "=" * 80)
print("TABLE 2: CORRELATION MATRIX")
print("=" * 80)

# Key variables for correlation
corr_vars = [
    'Trust_Score', 'Severity_Index', 'delay_days',
    'CREDIBILITY', 'CONSISTENCY', 'TRANSPARENCY', 'TIMELINESS', 'INTEGRITY',
    'at', 'MKCAP', 'ROA1', 'leverage2at'
]

# Filter to available variables
corr_vars_available = [v for v in corr_vars if v in df.columns]

# Compute correlation
corr_matrix = df[corr_vars_available].corr()

print("\nCorrelation Matrix (first 5x5):")
print(corr_matrix.iloc[:5, :5].round(3))

# Save correlation matrix
corr_file = OUTPUT_TABLES / 'correlation_matrix.csv'
corr_matrix.to_csv(corr_file)
print(f"\nSaved correlation matrix to: {corr_file}")

# =============================================================================
# TABLE 3: TRUST SCORE BY YEAR
# =============================================================================

print("\n" + "=" * 80)
print("TABLE 3: TRUST SCORE BY YEAR")
print("=" * 80)

trust_by_year = df.groupby('year').agg({
    'Trust_Score': ['count', 'mean', 'median', 'std'],
    'delay_days': ['mean', 'median'],
    'Severity_Index': ['mean', 'std']
}).round(3)

print("\nRecent years:")
print(trust_by_year.tail(10))

trust_year_file = OUTPUT_TABLES / 'trust_by_year.csv'
trust_by_year.to_csv(trust_year_file)
print(f"\nSaved to: {trust_year_file}")

# =============================================================================
# TABLE 4: TRUST SCORE BY INDUSTRY
# =============================================================================

print("\n" + "=" * 80)
print("TABLE 4: TRUST SCORE BY INDUSTRY")
print("=" * 80)

if 'ff12' in df.columns:
    trust_by_industry = df.groupby('ff12').agg({
        'Trust_Score': ['count', 'mean', 'std'],
        'delay_days': ['mean', 'median'],
        'Severity_Index': 'mean'
    }).round(3)
    
    trust_by_industry = trust_by_industry.sort_values(('Trust_Score', 'mean'), ascending=False)
    print("\nTop 10 industries by trust:")
    print(trust_by_industry.head(10))
    
    trust_industry_file = OUTPUT_TABLES / 'trust_by_industry.csv'
    trust_by_industry.to_csv(trust_industry_file)
    print(f"\nSaved to: {trust_industry_file}")
else:
    print("Industry variable (ff12) not found")

# =============================================================================
# TABLE 5: TRUST SCORE BY FILING TYPE
# =============================================================================

print("\n" + "=" * 80)
print("TABLE 5: TRUST SCORE BY FILING TYPE")
print("=" * 80)

trust_by_type = df.groupby('is_10k').agg({
    'Trust_Score': ['count', 'mean', 'std'],
    'delay_days': ['mean', 'median'],
    'Severity_Index': 'mean',
    'word_count': 'mean'
}).round(3)

trust_by_type.index = ['10-Q', '10-K']
print(trust_by_type)

trust_type_file = OUTPUT_TABLES / 'trust_by_filing_type.csv'
trust_by_type.to_csv(trust_type_file)
print(f"\nSaved to: {trust_type_file}")

# =============================================================================
# FIGURE 1: TRUST SCORE DISTRIBUTION
# =============================================================================

print("\n" + "=" * 80)
print("FIGURE 1: TRUST SCORE DISTRIBUTION")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(df['Trust_Score'].dropna(), bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].axvline(df['Trust_Score'].mean(), color='red', linestyle='--',
                   label=f'Mean = {df["Trust_Score"].mean():.3f}')
axes[0, 0].axvline(df['Trust_Score'].median(), color='blue', linestyle='--',
                   label=f'Median = {df["Trust_Score"].median():.3f}')
axes[0, 0].set_xlabel('Trust Score')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Panel A: Distribution of Trust Score')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

axes[0, 1].hist(df['Severity_Index'].dropna(), bins=50, edgecolor='black',
                alpha=0.7, color='orange')
axes[0, 1].axvline(df['Severity_Index'].mean(), color='red', linestyle='--',
                   label=f'Mean = {df["Severity_Index"].mean():.3f}')
axes[0, 1].set_xlabel('Severity Index')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Panel B: Distribution of Severity Index')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

df['trust_bin_plot'] = pd.qcut(df['Trust_Score'], q=20, duplicates='drop')
trust_delay = df.groupby('trust_bin_plot', observed=True).agg({
    'Trust_Score': 'mean',
    'delay_days': 'mean'
}).reset_index()

axes[1, 0].scatter(trust_delay['Trust_Score'], trust_delay['delay_days'], s=100, alpha=0.6)
axes[1, 0].set_xlabel('Trust Score (binned)')
axes[1, 0].set_ylabel('Mean Delay (days)')
axes[1, 0].set_title('Panel C: Trust Score vs Delay Length')
axes[1, 0].grid(alpha=0.3)

z = np.polyfit(trust_delay['Trust_Score'], trust_delay['delay_days'], 1)
p = np.poly1d(z)
axes[1, 0].plot(trust_delay['Trust_Score'], p(trust_delay['Trust_Score']),
                "r--", alpha=0.8, label=f'Slope = {z[0]:.2f}')
axes[1, 0].legend()

components_by_year = df.groupby('year')[['CREDIBILITY', 'CONSISTENCY', 'TRANSPARENCY',
                                         'TIMELINESS', 'INTEGRITY']].mean()

for comp in ['CREDIBILITY', 'CONSISTENCY', 'TRANSPARENCY', 'TIMELINESS', 'INTEGRITY']:
    axes[1, 1].plot(components_by_year.index, components_by_year[comp],
                    marker='o', label=comp)

axes[1, 1].set_xlabel('Year')
axes[1, 1].set_ylabel('Component Score')
axes[1, 1].set_title('Panel D: Trust Components Over Time')
axes[1, 1].legend(loc='best', fontsize=8)
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()

fig1_file = OUTPUT_FIGURES / 'trust_distribution.png'
plt.savefig(fig1_file, dpi=300, bbox_inches='tight')
print(f"\nSaved Figure 1 to: {fig1_file}")
plt.close()

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("DESCRIPTIVE STATISTICS COMPLETE")
print("=" * 80)

print("\nTables created:")
print("  Table 1: Summary statistics")
print("  Table 2: Correlation matrix")
print("  Table 3: Trust by year")
print("  Table 4: Trust by industry")
print("  Table 5: Trust by filing type")

print("\nFigures created:")
print("  Figure 1: Trust score distribution")
print("  Figure 2: Component correlations")

print("\nNext step: Run python 07_panel_regressions.py")
print("=" * 80)
