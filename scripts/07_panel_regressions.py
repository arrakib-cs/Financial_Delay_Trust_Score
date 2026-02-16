# Main econometric models
"""
SCRIPT 07: Panel Regressions
=============================
This script runs the main econometric models for the paper.

Models:
1. OLS: Trust → Delay Days (with FE)
2. Logit: Trust → Long Delay
3. Tobit: Trust → Delay (censored at 0)
4. Component decomposition

HOW TO RUN THIS:
----------------
    python 07_panel_regressions.py

OUTPUT:
-------
    ../output/tables/regression_results.csv
    ../output/tables/model1_trust_delay.txt
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.discrete_model import Logit
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("SCRIPT 07: PANEL REGRESSIONS")
print("=" * 80)

# Set up paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
OUTPUT_TABLES = PROJECT_ROOT / 'output' / 'tables'

# Load data
print("\nLoading data...")
df = pd.read_csv(DATA_PROCESSED / 'data_with_severity.csv', low_memory=False)
print(f"Loaded {len(df):,} observations")

# Prepare data for regression
print("\n" + "=" * 80)
print("PREPARING DATA FOR REGRESSION")
print("=" * 80)

# Create lagged controls (if not already present)
control_vars = ['at', 'ROA1', 'leverage2at', 'tobQ', 'firm_age']
available_controls = [v for v in control_vars if v in df.columns]

# Log transform assets if needed
if 'at' in df.columns and 'ln_at' not in df.columns:
    df['ln_at'] = np.log(df['at'].replace(0, np.nan))
    available_controls = ['ln_at'] + [v for v in available_controls if v != 'at']

print(f"Using {len(available_controls)} control variables: {available_controls}")

# Create regression sample (drop missing in key variables)
reg_vars = ['Trust_Score', 'delay_days', 'long_delay'] + available_controls + ['gvkey', 'year', 'big4', 'is_10k']
reg_vars = [v for v in reg_vars if v in df.columns]

df_reg = df[reg_vars].dropna()
print(f"Regression sample: {len(df_reg):,} observations")

# =============================================================================
# MODEL 1: TRUST → DELAY DAYS (OLS WITH FIXED EFFECTS)
# =============================================================================

print("\n" + "=" * 80)
print("MODEL 1: Trust Score → Delay Days (OLS)")
print("=" * 80)

# Build formula
controls_str = ' + '.join(available_controls)
formula1 = f'delay_days ~ Trust_Score + {controls_str} + big4 + is_10k + C(year) + C(gvkey)'

print(f"\nFormula: {formula1}")
print("This may take a few minutes with firm fixed effects...")

try:
    model1 = smf.ols(formula1, data=df_reg).fit(
        cov_type='cluster', cov_kwds={'groups': df_reg['gvkey']}
    )

    print("\nMODEL 1 RESULTS:")
    print("-" * 80)
    print(model1.summary().tables[1])

    with open(OUTPUT_TABLES / 'model1_trust_delay.txt', 'w') as f:
        f.write(model1.summary().as_text())

    print("Model 1 saved to: model1_trust_delay.txt")

    trust_coef = model1.params['Trust_Score']
    trust_se = model1.bse['Trust_Score']
    trust_pval = model1.pvalues['Trust_Score']

    print("\nKey Result: Trust Score Coefficient")
    print(f"Beta = {trust_coef:.4f} (SE = {trust_se:.4f}, p = {trust_pval:.4f})")

    if trust_pval < 0.01:
        print("Significant at 1 percent level")
    elif trust_pval < 0.05:
        print("Significant at 5 percent level")
    elif trust_pval < 0.10:
        print("Significant at 10 percent level")

except Exception as e:
    print(f"Model 1 failed: {e}")
    print("Re-estimating without firm fixed effects...")

    formula1_simple = f'delay_days ~ Trust_Score + {controls_str} + big4 + is_10k + C(year)'
    model1 = smf.ols(formula1_simple, data=df_reg).fit(cov_type='HC1')
    print(model1.summary().tables[1])

# =============================================================================
# MODEL 2: TRUST → LONG DELAY (LOGIT)
# =============================================================================

print("\n" + "=" * 80)
print("MODEL 2: Trust Score → Long Delay (Logit)")
print("=" * 80)

formula2 = f'long_delay ~ Trust_Score + {controls_str} + big4 + is_10k'
print(f"\nFormula: {formula2}")

try:
    model2 = smf.logit(formula2, data=df_reg).fit(disp=0)

    print("\nMODEL 2 RESULTS:")
    print("-" * 80)
    print(model2.summary().tables[1])

    with open(OUTPUT_TABLES / 'model2_trust_longdelay.txt', 'w') as f:
        f.write(model2.summary().as_text())

    print("Model 2 saved to: model2_trust_longdelay.txt")

    margeff = model2.get_margeff()
    print("\nMarginal Effects:")
    print(margeff.summary())

except Exception as e:
    print(f"Model 2 failed: {e}")

# =============================================================================
# MODEL 3: COMPONENT DECOMPOSITION
# =============================================================================

print("\n" + "=" * 80)
print("MODEL 3: Individual Trust Components → Delay")
print("=" * 80)

components = ['CREDIBILITY', 'CONSISTENCY', 'TRANSPARENCY', 'TIMELINESS', 'INTEGRITY']
component_results = []

for comp in components:
    if comp in df_reg.columns:
        formula_comp = f'delay_days ~ {comp} + {controls_str} + big4 + is_10k'

        try:
            model_comp = smf.ols(formula_comp, data=df_reg).fit(cov_type='HC1')

            result = {
                'Component': comp,
                'Coefficient': model_comp.params[comp],
                'Std_Error': model_comp.bse[comp],
                'T_Stat': model_comp.tvalues[comp],
                'P_Value': model_comp.pvalues[comp],
                'R_Squared': model_comp.rsquared,
                'N': int(model_comp.nobs)
            }

            component_results.append(result)

            sig = ""
            if result['P_Value'] < 0.01:
                sig = "***"
            elif result['P_Value'] < 0.05:
                sig = "**"
            elif result['P_Value'] < 0.10:
                sig = "*"

            print(f"{comp:15s}: Beta = {result['Coefficient']:7.3f} "
                  f"(SE = {result['Std_Error']:6.3f}) {sig}")

        except Exception as e:
            print(f"{comp:15s}: Estimation failed - {e}")

if component_results:
    comp_df = pd.DataFrame(component_results)
    comp_file = OUTPUT_TABLES / 'component_regressions.csv'
    comp_df.to_csv(comp_file, index=False)
    print(f"Component results saved to: {comp_file}")

# =============================================================================
# MODEL 4: ALL COMPONENTS TOGETHER
# =============================================================================

print("\n" + "=" * 80)
print("MODEL 4: All Components Together → Delay")
print("=" * 80)

components_str = ' + '.join(components)
formula4 = f'delay_days ~ {components_str} + {controls_str} + big4 + is_10k'
print(f"\nFormula: {formula4}")

try:
    model4 = smf.ols(formula4, data=df_reg).fit(cov_type='HC1')

    print("\nMODEL 4 RESULTS:")
    print("-" * 80)
    print(model4.summary().tables[1])

    with open(OUTPUT_TABLES / 'model4_all_components.txt', 'w') as f:
        f.write(model4.summary().as_text())

    print("Model 4 saved to: model4_all_components.txt")

except Exception as e:
    print(f"Model 4 failed: {e}")

# =============================================================================
# ROBUSTNESS: SEVERITY INDEX
# =============================================================================

print("\n" + "=" * 80)
print("ROBUSTNESS: Severity Index → Delay")
print("=" * 80)

if 'Severity_Index' in df_reg.columns:
    formula_sev = f'delay_days ~ Severity_Index + {controls_str} + big4 + is_10k'

    try:
        model_sev = smf.ols(formula_sev, data=df_reg).fit(cov_type='HC1')

        print("\nSEVERITY MODEL RESULTS:")
        print("-" * 80)
        print(model_sev.summary().tables[1])

        with open(OUTPUT_TABLES / 'model_severity.txt', 'w') as f:
            f.write(model_sev.summary().as_text())

        print("Severity model saved")

    except Exception as e:
        print(f"Severity model failed: {e}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("PANEL REGRESSIONS COMPLETE")
print("=" * 80)

print("\nKey models estimated:")
print("  Model 1: Trust → Delay (OLS with fixed effects)")
print("  Model 2: Trust → Long Delay (Logit)")
print("  Model 3: Individual trust components")
print("  Model 4: All components jointly")
print("  Robustness: Severity Index")

print("\nNext step: Review regression tables or run 08_event_study.py")
print("=" * 80)
