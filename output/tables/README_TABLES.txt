# OUTPUT TABLES

This folder contains all tables for the paper.

## MAIN TABLES (for manuscript)

**TABLE_1_Summary_Statistics.csv**
- Panel A: Dependent Variables
- Panel B: Trust Variables (Trust Score + 5 components)
- Panel C: Text Features
- Panel D: Firm Characteristics
- Panel E: Filing Characteristics

**TABLE_2_Correlations.csv**
- Correlation matrix of key variables
- Includes Trust Score, components, and controls

**TABLE_3_Main_Regressions_TEMPLATE.csv**
- Template for main regression results
- Fill in with output from Script 07
- Format: Coefficients with standard errors in parentheses
- Add significance stars: *** p<0.01, ** p<0.05, * p<0.1

**TABLE_4_Variable_Definitions.csv**
- Complete definitions of all variables
- Sources and construction methods
- Use in paper's Appendix A

## INTERNET APPENDIX TABLES

**IA_TABLE_1_Component_Correlations.csv**
- Correlations between 5 trust components

**IA_TABLE_2_Trust_by_Industry.csv**
- Trust Score summary by Fama-French 12 industries

**IA_TABLE_3_Trust_by_Year.csv**
- Trust Score trends over time (2000-2024)

## ADDITIONAL FILES

**ALL_TABLES.xlsx**
- Combined Excel workbook with all tables
- Each table on separate sheet
- Easy to copy into Word/PowerPoint

**model1_trust_delay.txt**
- Full regression output from Script 07
- Use to populate Table 3

**ml_results.csv**
- Machine learning model comparison
- For Appendix B (ML Validation)

## NOTES FOR PAPER

1. **Table Formatting**: Use these as starting points, then format for JFQA style
2. **Significance Stars**: Always use *** p<0.01, ** p<0.05, * p<0.1
3. **Standard Errors**: Report in parentheses, clustered by firm
4. **Notes**: Add table notes explaining controls, fixed effects, sample
5. **LaTeX**: .tex versions available for LaTeX users

## HOW TO USE

1. Open CSV files in Excel or R
2. Copy to your manuscript
3. Format according to JFQA guidelines
4. Add table notes and captions
5. Reference in text

For questions about any table, see the script that created it (noted in filename).
