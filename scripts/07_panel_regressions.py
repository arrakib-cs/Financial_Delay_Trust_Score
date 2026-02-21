# Main econometric models
"""
SCRIPT 07: Panel Regressions
=============================
This script runs the main econometric models for the paper.

Models:
1. OLS: Trust → Delay Days (with FE)
2. Logit: Trust → Long Delay
3. Component decomposition (individual)
4. All components together
5. Robustness: Severity Index

HOW TO RUN THIS:
----------------
    python scripts/07_panel_regressions.py

OUTPUT:
-------
    output/tables/model1_trust_delay.txt
    output/tables/model2_trust_longdelay.txt
    output/tables/component_regressions.csv
    output/tables/model4_all_components.txt
    output/tables/model_severity.txt

FIX NOTES (vs original):
--------------------------
- Uppercase column names (CREDIBILITY etc.) now wrapped in Q() for statsmodels
- firm_age fallback: uses lag_firm_age if firm_age missing
- at used directly (not ln_at) to match script 07 original controls
- Severity_Index column check made robust
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("SCRIPT 07: PANEL REGRESSIONS")
print("=" * 80)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
OUTPUT_TABLES  = PROJECT_ROOT / 'output' / 'tables'
OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading data...")
df = pd.read_csv(DATA_PROCESSED / 'data_with_severity.csv', low_memory=False)
print(f"Loaded {len(df):,} observations, {len(df.columns)} columns")

# ─────────────────────────────────────────────────────────────────────────────
# PREPARE VARIABLES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("PREPARING VARIABLES")
print("=" * 80)

# firm_age: use lag_firm_age if firm_age missing
if 'firm_age' not in df.columns and 'lag_firm_age' in df.columns:
    df['firm_age'] = df['lag_firm_age']
    print("  NOTE: firm_age not found — using lag_firm_age as substitute")

# Winsorize continuous variables at 1st/99th percentile
for col in ['at', 'ROA1', 'leverage2at', 'tobQ', 'firm_age']:
    if col in df.columns:
        p01 = df[col].quantile(0.01)
        p99 = df[col].quantile(0.99)
        df[col] = df[col].clip(p01, p99)

# Check trust components: must be uppercase in data (CREDIBILITY etc.)
components = ['CREDIBILITY', 'CONSISTENCY', 'TRANSPARENCY', 'TIMELINESS', 'INTEGRITY']
available_components = [c for c in components if c in df.columns]
print(f"  Trust components found: {available_components}")

# Core control variables
control_vars  = ['at', 'ROA1', 'leverage2at', 'tobQ', 'firm_age']
avail_controls = [v for v in control_vars if v in df.columns]
print(f"  Control variables: {avail_controls}")

# Required columns
reg_base = ['Trust_Score', 'delay_days', 'long_delay'] + avail_controls + ['gvkey', 'year', 'big4', 'is_10k']
reg_base = [v for v in reg_base if v in df.columns]

df_reg = df[list(set(reg_base + available_components))].dropna()
print(f"  Regression sample after listwise deletion: {len(df_reg):,} observations")

controls_str = ' + '.join(avail_controls)   # plain controls string for formulas


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1: OLS — Trust → Delay Days (with Firm + Year FE)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("MODEL 1: Trust Score → Delay Days (OLS with Fixed Effects)")
print("=" * 80)

formula1 = f'delay_days ~ Trust_Score + {controls_str} + big4 + is_10k + C(year) + C(gvkey)'
print(f"\nFormula: {formula1}")
print("(Firm fixed effects included — this may take 1-2 minutes...)")

try:
    model1 = smf.ols(formula1, data=df_reg).fit(
        cov_type='cluster',
        cov_kwds={'groups': df_reg['gvkey']}
    )

    print("\nMODEL 1 RESULTS:")
    print("-" * 80)
    # Print only the non-FE rows for readability
    coef_table = model1.summary().tables[1]
    print(coef_table)

    with open(OUTPUT_TABLES / 'model1_trust_delay.txt', 'w') as f:
        f.write(model1.summary().as_text())
    print("Model 1 saved to: model1_trust_delay.txt")

    trust_coef  = model1.params['Trust_Score']
    trust_se    = model1.bse['Trust_Score']
    trust_pval  = model1.pvalues['Trust_Score']
    trust_r2    = model1.rsquared

    print(f"\nKey Result: Trust Score Coefficient")
    print(f"Beta = {trust_coef:.4f} (SE = {trust_se:.4f}, p = {trust_pval:.4f})")
    print(f"R-squared = {trust_r2:.4f}")

    if   trust_pval < 0.01: print("Significant at 1% level ***")
    elif trust_pval < 0.05: print("Significant at 5% level **")
    elif trust_pval < 0.10: print("Significant at 10% level *")
    else:                   print("Not significant (p > 0.10)")

except Exception as e:
    print(f"Model 1 with firm FE failed: {e}")
    print("Falling back to year FE only...")
    formula1_simple = f'delay_days ~ Trust_Score + {controls_str} + big4 + is_10k + C(year)'
    model1 = smf.ols(formula1_simple, data=df_reg).fit(cov_type='HC1')
    print(model1.summary().tables[1])


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2: LOGIT — Trust → Long Delay
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3: INDIVIDUAL COMPONENTS → Delay Days
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("MODEL 3: Individual Trust Components → Delay Days")
print("=" * 80)

component_results = []

for comp in available_components:
    # Use Q('COMPONENT') to handle uppercase names safely in statsmodels
    formula_comp = f'delay_days ~ Q("{comp}") + {controls_str} + big4 + is_10k + C(year)'

    try:
        model_comp = smf.ols(formula_comp, data=df_reg).fit(cov_type='HC1')

        # statsmodels names the param as Q("COMP") — extract it
        param_key = f'Q("{comp}")'
        coef = model_comp.params[param_key]
        se   = model_comp.bse[param_key]
        tval = model_comp.tvalues[param_key]
        pval = model_comp.pvalues[param_key]

        result = {
            'Component': comp,
            'Coefficient': coef,
            'Std_Error': se,
            'T_Stat': tval,
            'P_Value': pval,
            'R_Squared': model_comp.rsquared,
            'N': int(model_comp.nobs)
        }
        component_results.append(result)

        sig = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.10 else ""))
        print(f"{comp:15s}: Beta = {coef:8.4f}  SE = {se:7.4f}  t = {tval:6.3f}  p = {pval:.4f}  {sig}")

    except Exception as e:
        print(f"{comp:15s}: FAILED — {e}")

if component_results:
    comp_df = pd.DataFrame(component_results)
    comp_file = OUTPUT_TABLES / 'component_regressions.csv'
    comp_df.to_csv(comp_file, index=False)
    print(f"\nComponent results saved to: {comp_file}")

    print("\n--- COMPONENT SUMMARY ---")
    print(f"{'Component':<15} {'Beta':>8} {'SE':>8} {'p':>8} {'Sig':>4} {'R2':>7}")
    print("-" * 55)
    for r in component_results:
        sig = "***" if r['P_Value'] < 0.01 else ("**" if r['P_Value'] < 0.05 else ("*" if r['P_Value'] < 0.10 else "n.s."))
        print(f"{r['Component']:<15} {r['Coefficient']:>8.4f} {r['Std_Error']:>8.4f} {r['P_Value']:>8.4f} {sig:>4} {r['R_Squared']:>7.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 4: ALL COMPONENTS TOGETHER → Delay Days
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("MODEL 4: All Components Together → Delay Days")
print("=" * 80)

if available_components:
    # Use Q() for all uppercase component names
    comp_formula_parts = ' + '.join([f'Q("{c}")' for c in available_components])
    formula4 = f'delay_days ~ {comp_formula_parts} + {controls_str} + big4 + is_10k + C(year)'
    print(f"\nFormula: {formula4}")

    try:
        model4 = smf.ols(formula4, data=df_reg).fit(cov_type='HC1')

        print("\nMODEL 4 RESULTS:")
        print("-" * 80)
        print(model4.summary().tables[1])

        with open(OUTPUT_TABLES / 'model4_all_components.txt', 'w') as f:
            f.write(model4.summary().as_text())
        print("Model 4 saved to: model4_all_components.txt")

        print("\n--- MODEL 4 COMPONENT COEFFICIENTS ---")
        print(f"{'Component':<15} {'Beta':>8} {'SE':>8} {'p':>8} {'Sig':>4}")
        print("-" * 47)
        for comp in available_components:
            param_key = f'Q("{comp}")'
            coef = model4.params[param_key]
            se   = model4.bse[param_key]
            pval = model4.pvalues[param_key]
            sig  = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.10 else "n.s."))
            print(f"{comp:<15} {coef:>8.4f} {se:>8.4f} {pval:>8.4f} {sig:>4}")

        print(f"\nR-squared: {model4.rsquared:.4f}  |  N: {int(model4.nobs):,}")

    except Exception as e:
        print(f"Model 4 failed: {e}")
else:
    print("No components found in dataset — skipping Model 4.")


# ─────────────────────────────────────────────────────────────────────────────
# ROBUSTNESS: SEVERITY INDEX → Delay Days
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("ROBUSTNESS: Severity Index → Delay Days")
print("=" * 80)

# Detect Severity_Index column (may have different capitalization)
sev_col = next((c for c in df.columns if c.lower() == 'severity_index'), None)

if sev_col:
    formula_sev = f'delay_days ~ {sev_col} + {controls_str} + big4 + is_10k + C(year)'
    print(f"\nFormula: {formula_sev}")

    try:
        model_sev = smf.ols(formula_sev, data=df_reg).fit(cov_type='HC1')

        print("\nSEVERITY MODEL RESULTS:")
        print("-" * 80)
        print(model_sev.summary().tables[1])

        with open(OUTPUT_TABLES / 'model_severity.txt', 'w') as f:
            f.write(model_sev.summary().as_text())
        print("Severity model saved to: model_severity.txt")

        sev_coef = model_sev.params[sev_col]
        sev_pval = model_sev.pvalues[sev_col]
        sig      = "***" if sev_pval < 0.01 else ("**" if sev_pval < 0.05 else ("*" if sev_pval < 0.10 else "n.s."))
        print(f"\nSeverity_Index: Beta = {sev_coef:.4f}, p = {sev_pval:.4f}  {sig}")

    except Exception as e:
        print(f"Severity model failed: {e}")
else:
    print("Severity_Index column not found in dataset — skipping robustness check.")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE (copy-paste ready)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("CLEAN SUMMARY — COPY THESE RESULTS INTO YOUR DOCUMENT")
print("=" * 80)

try:
    print("\n=== MODEL 1: OLS (delay_days) ===")
    key_vars = ['Trust_Score', 'at', 'ROA1', 'leverage2at', 'tobQ', 'firm_age', 'big4', 'is_10k']
    print(f"{'Variable':<20} {'Coeff':>10} {'Std.Err':>10} {'t':>8} {'p-value':>10} {'Sig':>5}")
    print("-" * 70)
    for v in key_vars:
        if v in model1.params:
            c = model1.params[v]
            s = model1.bse[v]
            t = model1.tvalues[v]
            p = model1.pvalues[v]
            sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
            print(f"{v:<20} {c:>10.4f} {s:>10.4f} {t:>8.3f} {p:>10.4f} {sig:>5}")
    print(f"R-squared: {model1.rsquared:.4f}  |  N: {int(model1.nobs):,}")
except:
    pass

try:
    print("\n=== MODEL 2: LOGIT (long_delay) ===")
    print(f"{'Variable':<20} {'Coeff':>10} {'Std.Err':>10} {'z':>8} {'p-value':>10} {'Sig':>5}")
    print("-" * 70)
    for v in key_vars:
        if v in model2.params:
            c = model2.params[v]
            s = model2.bse[v]
            z = model2.tvalues[v]
            p = model2.pvalues[v]
            sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
            print(f"{v:<20} {c:>10.4f} {s:>10.4f} {z:>8.3f} {p:>10.4f} {sig:>5}")
except:
    pass

print("\n" + "=" * 80)
print("PANEL REGRESSIONS COMPLETE")
print("=" * 80)
print("\nKey models estimated:")
print("  Model 1: Trust → Delay Days (OLS with firm + year FE)")
print("  Model 2: Trust → Long Delay (Logit)")
print("  Model 3: Individual trust components → Delay Days")
print("  Model 4: All components jointly → Delay Days")
print("  Robustness: Severity Index → Delay Days")
print("\nNext step: run 08_event_study.py")
print("=" * 80)