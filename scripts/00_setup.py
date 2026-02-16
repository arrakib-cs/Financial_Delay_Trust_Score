"""
SCRIPT 00: COMPLETE PROJECT SETUP
==================================
This script automatically installs EVERYTHING you need and sets up all data.

WHAT THIS DOES:
1. Installs all Python libraries from requirements.txt
2. Downloads spaCy language model
3. Downloads Loughran-McDonald dictionary
4. Verifies data files are in correct location
5. Creates all project folders
6. Tests that everything works

HOW TO RUN THIS:
----------------
Option 1 (Recommended):
    python 00_setup.py --install-all

Option 2 (Just verify, don't install):
    python 00_setup.py

This is the ONLY script you need to run for complete setup!
"""

import sys
import os
import subprocess
from pathlib import Path
import argparse

print("=" * 80)
print("NT DISCLOSURE TRUST PROJECT - COMPLETE SETUP")
print("=" * 80)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Setup NT Disclosure Trust Project')
parser.add_argument('--install-all', action='store_true', 
                   help='Automatically install all dependencies')
args = parser.parse_args()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
DATA_DICT = PROJECT_ROOT / 'data' / 'dictionaries'
REQUIREMENTS_FILE = PROJECT_ROOT / 'requirements.txt'

# =============================================================================
# STEP 1: CHECK PYTHON VERSION
# =============================================================================

print("\n" + "=" * 80)
print("STEP 1: CHECKING PYTHON VERSION")
print("=" * 80)

print(f"\nPython version: {sys.version}")
if sys.version_info >= (3, 8):
    print("Python version is compatible (3.8+)")
else:
    print("WARNING: Python 3.8+ recommended")

# =============================================================================
# STEP 2: CREATE PROJECT FOLDERS
# =============================================================================

print("\n" + "=" * 80)
print("STEP 2: CREATING PROJECT FOLDERS")
print("=" * 80)

folders = [
    'data/raw',
    'data/processed',
    'data/dictionaries',
    'scripts',
    'output/tables',
    'output/figures',
    'output/diagnostics',
    'notebooks'
]

print("\nCreating project structure...")
for folder in folders:
    folder_path = PROJECT_ROOT / folder
    folder_path.mkdir(parents=True, exist_ok=True)
    print(f"{folder}")

print("\nAll project folders created!")

# =============================================================================
# STEP 3: INSTALL PYTHON LIBRARIES
# =============================================================================

print("\n" + "=" * 80)
print("STEP 3: INSTALLING PYTHON LIBRARIES")
print("=" * 80)

if args.install_all:
    print("\nInstalling all libraries from requirements.txt...")
    print("This will take 5-10 minutes. Please wait...\n")
    
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-r', str(REQUIREMENTS_FILE)],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("All libraries installed successfully!")
        else:
            print("Some libraries may have failed to install")
            print("Error output:")
            print(result.stderr[:500])
            
    except Exception as e:
        print(f"Error installing libraries: {e}")
        print("\nManual installation command:")
        print(f"    pip install -r requirements.txt")
else:
    print("\nSkipping automatic installation (use --install-all to install)")
    print("\nTo install manually, run:")
    print(f"    pip install -r requirements.txt")

print("\n" + "=" * 80)
print("CHECKING INSTALLED LIBRARIES")
print("=" * 80)

required_libs = {
    'pandas': 'Data manipulation',
    'numpy': 'Numerical computing',
    'scipy': 'Scientific computing',
    'spacy': 'NLP text processing',
    'nltk': 'Natural language toolkit',
    'statsmodels': 'Econometric models',
    'sklearn': 'Machine learning',
    'xgboost': 'Gradient boosting',
    'matplotlib': 'Plotting',
    'seaborn': 'Statistical plots',
    'requests': 'Download files',
    'openpyxl': 'Excel support'
}

missing_libs = []
installed_versions = {}

for lib, purpose in required_libs.items():
    try:
        if lib == 'sklearn':
            mod = __import__('sklearn')
        else:
            mod = __import__(lib)
        
        version = getattr(mod, '__version__', 'unknown')
        installed_versions[lib] = version
        print(f"{lib:20s} v{version:12s} - {purpose}")
    except ImportError:
        print(f"{lib:20s} NOT INSTALLED - {purpose}")
        missing_libs.append(lib)

if missing_libs:
    print(f"\n{len(missing_libs)} libraries missing: {', '.join(missing_libs)}")
    print("\nInstall them with:")
    print(f"    pip install {' '.join(missing_libs)}")
else:
    print("\nAll required libraries are installed!")

# =============================================================================
# STEP 4: DOWNLOAD SPACY MODEL
# =============================================================================

print("\n" + "=" * 80)
print("STEP 4: DOWNLOADING SPACY LANGUAGE MODEL")
print("=" * 80)

if args.install_all:
    print("\nDownloading spaCy English model (en_core_web_sm)...")
    print("This will take 1-2 minutes...\n")
    
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0 or 'already satisfied' in result.stdout.lower():
            print("spaCy model installed successfully!")
        else:
            print("spaCy model installation may have failed")
            print("\nManual installation:")
            print("    python -m spacy download en_core_web_sm")
    except Exception as e:
        print(f"Error: {e}")
        print("\nManual installation:")
        print("    python -m spacy download en_core_web_sm")
else:
    print("\nSkipping spaCy model download (use --install-all)")
    print("\nTo download manually:")
    print("    python -m spacy download en_core_web_sm")

try:
    import spacy
    nlp = spacy.load('en_core_web_sm')
    print("\nspaCy model (en_core_web_sm) is working!")
except:
    print("\nspaCy model not found or not working")
    print("Install it with: python -m spacy download en_core_web_sm")

# =============================================================================
# STEP 5: DOWNLOAD LOUGHRAN-MCDONALD DICTIONARY
# =============================================================================

print("\n" + "=" * 80)
print("STEP 5: DOWNLOADING LOUGHRAN-MCDONALD DICTIONARY")
print("=" * 80)

LM_URL = 'https://raw.githubusercontent.com/hanle0/loughran-mcdonald-sentiment/master/LM_MasterDictionary.csv'
LM_FILE = DATA_DICT / 'LM_MasterDictionary.csv'

if LM_FILE.exists():
    print("\nLoughran-McDonald dictionary already exists")
    print(f"Location: {LM_FILE}")
else:
    print("\nDownloading Loughran-McDonald dictionary...")
    
    try:
        import requests
        response = requests.get(LM_URL, timeout=30)
        
        if response.status_code == 200:
            with open(LM_FILE, 'wb') as f:
                f.write(response.content)
            print("Loughran-McDonald dictionary downloaded successfully!")
            print(f"Saved to: {LM_FILE}")
        else:
            print(f"Download failed (status code: {response.status_code})")
    except Exception as e:
        print(f"Error downloading dictionary: {e}")

# =============================================================================
# STEP 6: VERIFY DATA FILES
# =============================================================================

print("\n" + "=" * 80)
print("STEP 6: VERIFYING DATA FILES")
print("=" * 80)

DATA_FILE = DATA_RAW / 'merged_in_.csv'

print("\nChecking for your data file...")
print(f"Looking for: {DATA_FILE}")

if DATA_FILE.exists():
    print("\nData file found!")
else:
    print("\nData file NOT FOUND!")

# =============================================================================
# STEP 7: TEST IMPORTS
# =============================================================================

print("\n" + "=" * 80)
print("STEP 7: TESTING KEY FUNCTIONALITY")
print("=" * 80)

print("\nTesting key imports and functionality...")

tests_passed = 0
tests_total = 5

try:
    import pandas as pd
    print("Test 1: pandas working")
    tests_passed += 1
except Exception as e:
    print(f"Test 1 failed: {e}")

try:
    import numpy as np
    print("Test 2: numpy working")
    tests_passed += 1
except Exception as e:
    print(f"Test 2 failed: {e}")

try:
    import spacy
    print("Test 3: spaCy working")
    tests_passed += 1
except Exception as e:
    print(f"Test 3 failed: {e}")

try:
    import statsmodels.api as sm
    print("Test 4: statsmodels working")
    tests_passed += 1
except Exception as e:
    print(f"Test 4 failed: {e}")

try:
    from sklearn.linear_model import LogisticRegression
    print("Test 5: scikit-learn working")
    tests_passed += 1
except Exception as e:
    print(f"Test 5 failed: {e}")

print(f"\nTests passed: {tests_passed}/{tests_total}")

print("\n" + "=" * 80)
print("SETUP SUMMARY")
print("=" * 80)

print("\nSetup process completed.")
print("=" * 80)
