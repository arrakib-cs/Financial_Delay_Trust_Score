"""
ADD Rd2at AND Capx2at TO PROCESSED DATASET
===========================================
Good news: rd2at and capx2at already exist in your raw file!
They were just lowercase so the previous script missed them.

This script:
1. Reads rd2at and capx2at from merged_in_.csv
2. Merges them into data_with_severity.csv
3. Saves the updated file

HOW TO RUN:
-----------
    python scripts/add_new_variables.py

THEN RUN:
---------
    python scripts/10_ml_validation.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ADDING Rd2at AND Capx2at TO PROCESSED DATASET")
print("=" * 80)

# Paths
PROJECT_ROOT   = Path(__file__).parent.parent
DATA_RAW       = PROJECT_ROOT / 'data' / 'raw'
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'

RAW_FILE       = DATA_RAW / 'merged_in_.csv'
PROCESSED_FILE = DATA_PROCESSED / 'data_with_severity.csv'

# -------------------------------------------------------------------
# STEP 1: Load raw data
# -------------------------------------------------------------------
print("\nSTEP 1: Loading raw data...")
df_raw = pd.read_csv(RAW_FILE, low_memory=False)
print(f"  Loaded {len(df_raw):,} rows")

# Rename lowercase to standard capitalized names
df_raw['Rd2at']   = pd.to_numeric(df_raw['rd2at'],   errors='coerce')
df_raw['Capx2at'] = pd.to_numeric(df_raw['capx2at'], errors='coerce')

# Winsorize at 1st and 99th percentile to remove extreme outliers
for col in ['Rd2at', 'Capx2at']:
    p01 = df_raw[col].quantile(0.01)
    p99 = df_raw[col].quantile(0.99)
    df_raw[col] = df_raw[col].clip(lower=p01, upper=p99)
    print(f"  {col}: mean={df_raw[col].mean():.4f}, "
          f"min={df_raw[col].min():.4f}, max={df_raw[col].max():.4f}, "
          f"missing={df_raw[col].isna().sum():,}")

# -------------------------------------------------------------------
# STEP 2: Load processed data
# -------------------------------------------------------------------
print("\nSTEP 2: Loading processed data...")
df_proc = pd.read_csv(PROCESSED_FILE, low_memory=False)
print(f"  Loaded {len(df_proc):,} rows, {len(df_proc.columns)} columns")

# Drop old versions if they exist
for col in ['Rd2at', 'Capx2at', 'rd2at', 'capx2at']:
    if col in df_proc.columns:
        df_proc.drop(columns=[col], inplace=True)
        print(f"  Removed old column: {col}")

# -------------------------------------------------------------------
# STEP 3: Merge
# -------------------------------------------------------------------
print("\nSTEP 3: Merging new variables...")

# Use gvkey + year as merge keys (most reliable)
merge_keys = ['gvkey', 'year']
df_to_merge = df_raw[merge_keys + ['Rd2at', 'Capx2at']].drop_duplicates(subset=merge_keys)

before = len(df_proc)
df_proc = df_proc.merge(df_to_merge, on=merge_keys, how='left')
after  = len(df_proc)

print(f"  Rows before: {before:,}  |  Rows after: {after:,}")

for col in ['Rd2at', 'Capx2at']:
    filled = df_proc[col].notna().sum()
    pct    = filled / len(df_proc) * 100
    print(f"  {col}: {filled:,} non-missing ({pct:.1f}% coverage)")

# -------------------------------------------------------------------
# STEP 4: Save
# -------------------------------------------------------------------
print(f"\nSTEP 4: Saving updated file...")
df_proc.to_csv(PROCESSED_FILE, index=False)
print(f"  Saved: {PROCESSED_FILE}")

# -------------------------------------------------------------------
# DONE
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("DONE! Rd2at and Capx2at added successfully.")
print("=" * 80)
print("\nNext step:")
print("  python scripts/10_ml_validation.py")
print("\nThe ML model will now run with 9 variables:")
print("  Trust_Score, ln_at, ROA1, leverage2at, tobQ, big4, is_10k, Rd2at, Capx2at")