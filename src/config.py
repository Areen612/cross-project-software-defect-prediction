
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = PROJECT_ROOT / "resources" / "ds.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs"

# Target configuration
SOURCE_COLUMN = "bugs"
TARGET_COLUMN = "buggy"

# Metadata columns (not predictors)
METADATA_COLUMNS = ["Project", "Version", "Class"]

# Approved primary predictors
FEATURES = [
    "noc", "cbo", "rfc", "lcom", "ca", "ce",
    "npm", "lcom3", "loc", "dam", "moa", "mfa",
    "cam", "ic", "cbm", "amc", "nr", "ndc",
]

# Excluded features (not used in modeling)
EXCLUDED_FEATURES = [
    "nml", "isOld", "ndpv", "ner", "max(cc)", "avg(cc)",
]

# Random Forest configuration
RANDOM_SEED = 42
N_ESTIMATORS = 200
MAX_FEATURES = "sqrt"
CLASS_WEIGHT = None
N_JOBS = -1
