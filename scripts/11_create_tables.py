"""
SCRIPT 11: Create Publication Tables
=====================================
This script creates all publication-ready tables for JFQA submission.

What this does:
1. Format all regression tables
2. Create summary statistics table
3. Generate correlation matrices
4. Build variable definitions table
5. Export in multiple formats (CSV, LaTeX, Excel)

HOW TO RUN THIS:
----------------
    python 11_create_tables.py

OUTPUT:
-------
    ../output/tables/TABLE_1_Summary_Statistics.csv
    ../output/tables/TABLE_2_Correlations.csv
    ../output/tables/TABLE_3_Main_Regressions.csv
    (and LaTeX versions)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("SCRIPT 11: CREATE PUBLICATION TABLES")
print("=" * 80)

# Set up paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
OUTPUT_TABLES = PROJECT_ROOT / 'output' / 'tables'

# Load data
print("\nLoading data...")
df = pd.read_csv(DATA_PROCESSED / 'data_with_severity.csv', low_memory=False)
print(f"Loaded {len(df):,} observations")

# =============================================================================
# TABLE 1: SUMMARY STATISTICS (PANEL STRUCTURE)
# =============================================================================

print("\n" + "=" * 80)
print("TABLE 1: SUMMARY STATISTICS")
print("=" * 80)

table1_vars = {
    'Panel A: Dependent Variables': {
        'delay_days': 'Delay Days',
        'long_delay': 'Long Delay (>30 days)',
        'Severity_Index': 'Delay Severity Index'
    },
    'Panel B: Trust Variables': {
        'Trust_Score': 'Trust Score',
        'CREDIBILITY': 'Credibility Component',
        'CONSISTENCY': 'Consistency Component',
        'TRANSPARENCY': 'Transparency Component',
        'TIMELINESS': 'Timeliness Component',
        'INTEGRITY': 'Integrity Component'
    },
    'Panel C: Text Features': {
        'word_count': 'Word Count',
        'vagueness_score': 'Vagueness Score (%)',
        'commitment_score': 'Commitment Score (%)',
        'specificity_score': 'Specificity Score (%)',
        'numerical_density': 'Numerical Density (%)',
        'lm_tone': 'Loughran-McDonald Tone'
    },
    'Panel D: Firm Characteristics': {
        'ln_at': 'Log(Total Assets)',
        'MKCAP': 'Market Cap ($M)',
        'ROA1': 'Return on Assets (%)',
        'leverage2at': 'Leverage Ratio',
        'tobQ': "Tobin's Q",
        'firm_age': 'Firm Age (years)',
        'big4': 'Big 4 Auditor'
    },
    'Panel E: Filing Characteristics': {
        'is_10k': '10-K Filing',
        'is_10q': '10-Q Filing',
        'days_before_deadline': 'Days Before Deadline',
        'nt_count': 'NT Count (Firm-Level)'
    }
}

print("\nGenerating summary statistics...")

summary_rows = []

for panel_name, variables in table1_vars.items():
    summary_rows.append({
        'Variable': panel_name,
        'N': '',
        'Mean': '',
        'Median': '',
        'Std': '',
        'Min': '',
        'P25': '',
        'P75': '',
        'Max': ''
    })

    for var, label in variables.items():
        if var in df.columns:
            summary_rows.append({
                'Variable': f'  {label}',
                'N': f"{df[var].count():,}",
                'Mean': f"{df[var].mean():.3f}",
                'Median': f"{df[var].median():.3f}",
                'Std': f"{df[var].std():.3f}",
                'Min': f"{df[var].min():.3f}",
                'P25': f"{df[var].quantile(0.25):.3f}",
                'P75': f"{df[var].quantile(0.75):.3f}",
                'Max': f"{df[var].max():.3f}"
            })

table1 = pd.DataFrame(summary_rows)

table1_file = OUTPUT_TABLES / 'TABLE_1_Summary_Statistics.csv'
table1.to_csv(table1_file, index=False)
print(f"Table 1 saved to: {table1_file}")

try:
    table1_latex = table1.to_latex(index=False, escape=False,
                                   caption='Summary Statistics',
                                   label='tab:summary')
    with open(OUTPUT_TABLES / 'TABLE_1_Summary_Statistics.tex', 'w') as f:
        f.write(table1_latex)
    print("Table 1 LaTeX saved")
except:
    print("Could not create LaTeX version")

# =============================================================================
# TABLE 2: CORRELATION MATRIX
# =============================================================================

print("\n" + "=" * 80)
print("TABLE 2: CORRELATION MATRIX")
print("=" * 80)

corr_vars_dict = {
    'Trust_Score': 'Trust Score',
    'Severity_Index': 'Severity Index',
    'delay_days': 'Delay Days',
    'CREDIBILITY': 'Credibility',
    'TRANSPARENCY': 'Transparency',
    'ln_at': 'Log(Assets)',
    'ROA1': 'ROA',
    'leverage2at': 'Leverage',
    'tobQ': "Tobin's Q"
}

corr_vars = [v for v in corr_vars_dict if v in df.columns]
corr_labels = [corr_vars_dict[v] for v in corr_vars]

corr_matrix = df[corr_vars].corr().round(3)
corr_matrix.columns = corr_labels
corr_matrix.index = corr_labels

table2_file = OUTPUT_TABLES / 'TABLE_2_Correlations.csv'
corr_matrix.to_csv(table2_file)
print(f"Table 2 saved to: {table2_file}")

# =============================================================================
# TABLE 3: MAIN REGRESSION RESULTS (TEMPLATE)
# =============================================================================

print("\n" + "=" * 80)
print("TABLE 3: MAIN REGRESSION RESULTS")
print("=" * 80)

table3_template = pd.DataFrame({
    'Variable': ['Trust Score', '', 'Credibility', '', 'Transparency', '',
                 'Log(Assets)', '', 'ROA', '', 'Leverage', '',
                 '', 'Controls', 'Firm FE', 'Year FE', '',
                 'Observations', 'R-squared'],
    'Model_1_OLS': ['Coefficient', '(Std Error)', '', '', '', '',
                   '', '', '', '', '', '',
                   '', 'No', 'No', 'No', '', 'N', 'R²'],
    'Model_2_OLS_FE': ['', '', '', '', '', '', '', '', '', '', '', '',
                      '', 'Yes', 'Yes', 'Yes', '', 'N', 'R²'],
    'Model_3_Logit': ['', '', '', '', '', '', '', '', '', '', '', '',
                     '', 'Yes', 'No', 'Yes', '', 'N', 'Pseudo R²'],
    'Model_4_Components': ['', '', '', '', '', '', '', '', '', '', '', '',
                          '', 'Yes', 'Yes', 'Yes', '', 'N', 'R²']
})

table3_file = OUTPUT_TABLES / 'TABLE_3_Main_Regressions_TEMPLATE.csv'
table3_template.to_csv(table3_file, index=False)
print(f"Table 3 template saved to: {table3_file}")

# =============================================================================
# TABLE 4: VARIABLE DEFINITIONS
# =============================================================================

print("\n" + "=" * 80)
print("TABLE 4: VARIABLE DEFINITIONS")
print("=" * 80)

var_definitions = pd.DataFrame({
    'Variable': [
        'Trust_Score','CREDIBILITY','CONSISTENCY','TRANSPARENCY','TIMELINESS',
        'INTEGRITY','Severity_Index','delay_days','long_delay',
        'vagueness_score','commitment_score','word_count',
        'ln_at','ROA1','leverage2at','tobQ','big4'
    ],
    'Definition': [
        'Composite measure of NT explanation credibility (0-1)',
        'Specificity + Commitment - Vagueness (0-1)',
        'Style similarity and reason stability (0-1)',
        'Detail level and explanation length (0-1)',
        'Days before deadline (0-1)',
        'Tone minus blame shifting (0-1)',
        '0.4×Delay + 0.4×(1-Trust) + 0.2×Vagueness',
        'Actual delay in days',
        'Indicator for delay > 30 days',
        'Share of vague words',
        'Share of commitment words',
        'Total narrative word count',
        'Log total assets',
        'Return on assets',
        'Leverage ratio',
        "Tobin's Q",
        'Big 4 auditor indicator'
    ]
})

table4_file = OUTPUT_TABLES / 'TABLE_4_Variable_Definitions.csv'
var_definitions.to_csv(table4_file, index=False)
print(f"Table 4 saved to: {table4_file}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("ALL PUBLICATION TABLES CREATED")
print("=" * 80)

print("\nNext steps:")
print("1. Review tables in output/tables/")
print("2. Fill Table 3 using results from Script 07")
print("3. Format tables according to JFQA guidelines")
print("4. Insert into manuscript")
print("=" * 80)
