from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
GOLD_FEATURES_DIR = DATA_DIR / "gold_features" # analytics layer - please find clustered data attached here
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

for directory in (RAW_DIR, PROCESSED_DIR, GOLD_FEATURES_DIR, MODELS_DIR, OUTPUTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

ZENODO_RECORD = "17352136"
BOUNDARY_FILENAME = "peru_field_boundaries.gpkg"
BOUNDARY_PATH = RAW_DIR / BOUNDARY_FILENAME
BOUNDARY_TABLE = "FIELDS_v2_REFINED_YEARLYCLASS_97_V3"
STAC_COLLECTION = "sentinel-2-l2a"
STAC_ITEM = "S2A_MSIL2A_20240328T152651_R025_T17MRM_20240330T060154"
ACQUISITION_DATE = "2024-03-28"
FIELDS = {"oat_176": 590, "oat_107": 355}
BANDS = ("B04", "B08", "B11", "SCL")
PIXEL_SIZE_M = 10
SEED = 42
MIN_CLUSTER_FRACTION = 0.05
