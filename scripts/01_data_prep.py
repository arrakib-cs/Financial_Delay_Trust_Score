# Load and clean raw NT filing data
"""
SCRIPT 01: Data Preparation
============================
This script loads your raw data and prepares it for analysis.

What this does:
1. Loads the merged CSV file
2. Creates key variables
3. Handles missing values
4. Filters to valid observations
5. Saves clean dataset

HOW TO RUN THIS:
----------------
    python 01_data_prep.py

OUTPUT:
-------
    ../data/processed/clean_data.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("SCRIPT 01: DATA PREPARATION")
print("=" * 80)

# Set up paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'

# Load data
print("\nLoading raw data...")
df = pd.read_csv(DATA_RAW / 'merged_in_.csv', low_memory=False)
print(f"Loaded {len(df):,} observations")
print(f"Columns: {len(df.columns)}")

# Initial data summary
print("\n" + "=" * 80)
print("RAW DATA SUMMARY")
print("=" * 80)
print(f"Total NT filings: {df['nt_filed'].sum():,}")
print(f"NT 10-Q filings: {df['nt_10q'].sum():,}")
print(f"NT 10-K filings: {df['nt_10k'].sum():,}")
print(f"Years covered: {df['year'].min()} to {df['year'].max()}")

# Data cleaning steps
print("\n" + "=" * 80)
print("DATA CLEANING")
print("=" * 80)

# Step 1: Keep only NT filings
print("\n1. Filtering to NT filings only...")
df_clean = df[df['nt_filed'] == 1].copy()
print(f"   Kept {len(df_clean):,} NT filings (removed {len(df) - len(df_clean):,} non-NT)")

# Step 2: Remove observations without narrative text
print("\n2. Removing observations without narrative text...")
initial_count = len(df_clean)
df_clean = df_clean[df_clean['NARRATIVE_TEXT'].notna()].copy()
print(f"   Kept {len(df_clean):,} observations (removed {initial_count - len(df_clean):,} missing narratives)")

# Step 3: Create clean narrative text variable
print("\n3. Creating clean narrative text...")
df_clean['narrative_clean'] = df_clean['NARRATIVE_TEXT'].astype(str)
print("   Created narrative_clean variable")

# Step 4: Parse dates
print("\n4. Parsing date variables...")
date_vars = ['FILE_DATE', 'MR_NT_FILE_DATE', 'StatDeadline', 'datadate']
for var in date_vars:
    if var in df_clean.columns:
        try:
            df_clean[var] = pd.to_datetime(df_clean[var], errors='coerce')
            print(f"   Parsed {var}")
        except:
            print(f"   Could not parse {var}")

# Step 5: Create key research variables
print("\n5. Creating research variables...")

# Binary: Is this a 10-K filing?
df_clean['is_10k'] = df_clean['nt_10k'].fillna(0).astype(int)

# Binary: Is this a 10-Q filing?
df_clean['is_10q'] = df_clean['nt_10q'].fillna(0).astype(int)

# Binary: Big 4 auditor
df_clean['big4'] = df_clean['big4'].fillna(0).astype(int)

# Delay in days (ensure numeric)
df_clean['delay_days'] = pd.to_numeric(df_clean['DelayDays'], errors='coerce')

# Binary: Long delay (>30 days)
df_clean['long_delay'] = (df_clean['delay_days'] > 30).astype(int)

# Log transformations for skewed variables
for var in ['at', 'sale', 'MKCAP']:
    if var in df_clean.columns:
        df_clean[f'ln_{var.lower()}'] = np.log(df_clean[var].replace(0, np.nan))

print("   Created binary variables (is_10k, is_10q, big4)")
print("   Created delay_days and long_delay")
print("   Created log transformations")

# Step 6: Handle missing values in control variables
print("\n6. Handling missing values in controls...")
control_vars = ['at', 'sale', 'MKCAP', 'ROA1', 'leverage2at', 'tobQ', 'firm_age']
missing_before = {}
for var in control_vars:
    if var in df_clean.columns:
        missing_before[var] = df_clean[var].isna().sum()

# For now, we'll keep observations with missing controls
# We'll handle them in the regression scripts with .dropna()
print("   Note: Missing values will be handled in regression scripts")
for var, count in missing_before.items():
    pct = count / len(df_clean) * 100
    print(f"   {var:20s}: {count:6,} missing ({pct:5.1f}%)")

# Step 7: Remove extreme outliers in delay
print("\n7. Checking for extreme delays...")
delay_extreme = df_clean[df_clean['delay_days'] > 1000]
if len(delay_extreme) > 0:
    print(f"   Found {len(delay_extreme)} observations with delay > 1000 days")
    print("   These may be data errors. Flagging but not removing.")
    df_clean['extreme_delay'] = (df_clean['delay_days'] > 1000).astype(int)
else:
    print("   No extreme delays found")

# Step 8: Sort by firm and date
print("\n8. Sorting data...")
df_clean = df_clean.sort_values(['gvkey', 'FILE_DATE'])
print("   Sorted by gvkey and FILE_DATE")

# Variable summary
print("\n" + "=" * 80)
print("KEY VARIABLE SUMMARY")
print("=" * 80)

summary_stats = pd.DataFrame({
    'N': df_clean[['delay_days', 'at', 'MKCAP', 'ROA1', 'leverage2at']].count(),
    'Mean': df_clean[['delay_days', 'at', 'MKCAP', 'ROA1', 'leverage2at']].mean(),
    'Median': df_clean[['delay_days', 'at', 'MKCAP', 'ROA1', 'leverage2at']].median(),
    'Std': df_clean[['delay_days', 'at', 'MKCAP', 'ROA1', 'leverage2at']].std(),
    'Min': df_clean[['delay_days', 'at', 'MKCAP', 'ROA1', 'leverage2at']].min(),
    'Max': df_clean[['delay_days', 'at', 'MKCAP', 'ROA1', 'leverage2at']].max()
})

print(summary_stats.round(2))

# Save cleaned data
print("\n" + "=" * 80)
print("SAVING CLEAN DATA")
print("=" * 80)

output_file = DATA_PROCESSED / 'clean_data.csv'
df_clean.to_csv(output_file, index=False)
print(f"Saved to: {output_file}")
print(f"Final dataset: {len(df_clean):,} observations")

# Create data dictionary
print("\n" + "=" * 80)
print("CREATING DATA DICTIONARY")
print("=" * 80)

data_dict = pd.DataFrame({
    'Variable': df_clean.columns,
    'Type': df_clean.dtypes.astype(str),
    'Missing_N': df_clean.isna().sum(),
    'Missing_Pct': (df_clean.isna().sum() / len(df_clean) * 100).round(2)
})

dict_file = DATA_PROCESSED / 'data_dictionary.csv'
data_dict.to_csv(dict_file, index=False)
print(f"Data dictionary saved to: {dict_file}")

print("\n" + "=" * 80)
print("DATA PREPARATION COMPLETE")
print("=" * 80)
print("\nNext step: Run python 02_text_preprocessing.py")
print("=" * 80)
