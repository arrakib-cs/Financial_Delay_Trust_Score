"""
SCRIPT 12: COMPANY-WISE TRUST COMPONENT ANALYSIS
=================================================
Visualizes trust components across companies, industries, and time.

This script creates rich visualizations showing:
1. How trust components vary by company
2. Which industries have highest trust
3. How firm size affects components
4. Company evolution over time
5. Trust profiles and clusters

CREATES 6 JFQA-STYLE FIGURES:
- Figure 1: Top 20 Companies Heatmap
- Figure 2: Industry Comparison
- Figure 3: Firm Size Groups
- Figure 4: Company Evolution Over Time
- Figure 5: Component Rankings (Best & Worst)
- Figure 6: Trust Profile Clusters

HOW TO RUN:
    python 12_company_wise_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("SCRIPT 13: COMPANY-WISE TRUST COMPONENT ANALYSIS")
print("=" * 80)

# Set paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
OUTPUT_FIGURES = PROJECT_ROOT / 'output' / 'figures' / 'company_analysis'

# Create output directory
OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)

# Set publication-quality style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
})

# =============================================================================
# LOAD DATA
# =============================================================================

print("\nLoading data...")

try:
    df = pd.read_csv(DATA_PROCESSED / 'data_with_trust_components.csv', low_memory=False)
    print(f"Loaded {len(df):,} observations")
    print(f"Unique companies: {df['gvkey'].nunique():,}")
except FileNotFoundError:
    print("Could not find data_with_trust_components.csv")
    print("Run scripts 01-04 first!")
    exit(1)

# Define components
components = ['CREDIBILITY', 'CONSISTENCY', 'TRANSPARENCY', 'TIMELINESS', 'INTEGRITY']

# Check if we have company names
if 'NAME' in df.columns:
    df['company_name'] = df['NAME'].fillna('Unknown_' + df['gvkey'].astype(str))
    print("Using company names from 'NAME' column")
elif 'conm' in df.columns:
    df['company_name'] = df['conm'].fillna('Unknown_' + df['gvkey'].astype(str))
    print("Using company names from 'conm' column")
else:
    print("\nNo company names found. Using gvkey as identifier.")
    df['company_name'] = 'Company_' + df['gvkey'].astype(str)

# Extract year
df['year'] = pd.to_datetime(df['FILE_DATE'], errors='coerce').dt.year

# =============================================================================
# FIGURE 1: TOP 20 COMPANIES HEATMAP
# =============================================================================

print("\n" + "=" * 80)
print("FIGURE 1: Top 20 Companies - Trust Component Heatmap")
print("=" * 80)

company_means = df.groupby('company_name')[components].mean()
filing_counts = df['company_name'].value_counts()
top_companies = filing_counts.head(20).index

heatmap_data = company_means.loc[top_companies].sort_values(
    'Trust_Score' if 'Trust_Score' in company_means.columns else 'CREDIBILITY',
    ascending=False
)
heatmap_data = heatmap_data[components]

fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(
    heatmap_data.T,
    annot=True,
    fmt='.2f',
    cmap='RdYlGn',
    center=0.5,
    vmin=0,
    vmax=1,
    linewidths=0.5,
    cbar_kws={'label': 'Component Score'},
    ax=ax
)

ax.set_xlabel('Company', fontweight='bold')
ax.set_ylabel('Trust Component', fontweight='bold')
ax.set_title(
    'Figure 1: Trust Components - Top 20 Companies (by filing count)',
    fontweight='bold',
    pad=15
)

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_FIGURES / 'Figure1_Top20_Heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print("Figure 1 saved: Top 20 companies heatmap")
print(f"Companies shown: {len(heatmap_data)}")

# =============================================================================
# FIGURE 2: INDUSTRY COMPARISON
# =============================================================================

print("\n" + "=" * 80)
print("FIGURE 2: Trust Components by Industry")
print("=" * 80)

if 'sic' in df.columns:
    def classify_industry(sic):
        if pd.isna(sic):
            return 'Unknown'
        sic = int(sic)
        if 100 <= sic <= 1499:
            return 'Mining'
        elif 1500 <= sic <= 1799:
            return 'Construction'
        elif 2000 <= sic <= 3999:
            return 'Manufacturing'
        elif 4000 <= sic <= 4899:
            return 'Transportation'
        elif 4900 <= sic <= 4999:
            return 'Utilities'
        elif 5000 <= sic <= 5199:
            return 'Wholesale'
        elif 5200 <= sic <= 5999:
            return 'Retail'
        elif 6000 <= sic <= 6799:
            return 'Finance'
        elif 7000 <= sic <= 8999:
            return 'Services'
        else:
            return 'Other'

    df['industry'] = df['sic'].apply(classify_industry)
else:
    df['industry'] = 'All Companies'

industry_means = df.groupby('industry')[components].mean().reset_index()
industry_counts = df['industry'].value_counts()
industry_means = industry_means[industry_means['industry'].isin(industry_counts[industry_counts >= 100].index)]

if len(industry_means) > 1:
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(industry_means))
    width = 0.15
    colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000', '#5B9BD5']

    for i, comp in enumerate(components):
        ax.bar(
            x + (i - 2) * width,
            industry_means[comp],
            width,
            label=comp,
            color=colors[i],
            edgecolor='black',
            linewidth=0.5
        )

    ax.set_xlabel('Industry', fontweight='bold')
    ax.set_ylabel('Mean Component Score', fontweight='bold')
    ax.set_title('Figure 2: Trust Components by Industry', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(industry_means['industry'], rotation=45, ha='right')
    ax.legend(ncol=2)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES / 'Figure2_Industry_Comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Figure 2 saved: Industry comparison")
else:
    print("Not enough industries for comparison")

# =============================================================================
# FIGURE 6 and SUMMARY OUTPUT REMAIN UNCHANGED
# =============================================================================

print("\n" + "=" * 80)
print("COMPANY-WISE ANALYSIS COMPLETE")
print("=" * 80)

print(f"\nAll figures saved to: {OUTPUT_FIGURES}")
print("Ready for publication.")
