# deriving NDVI & NDMI indices via SWRI // mn3392g
import numpy as np
import pandas as pd
from . import config


def create_gold_features():
    outputs = {}
    for field in config.FIELDS:
        frame = pd.read_parquet(config.PROCESSED_DIR / f"{field}_silver.parquet")
        frame["ndvi"] = (frame.nir - frame.red) / (frame.nir + frame.red)
        frame["ndmi"] = (frame.nir - frame.swir) / (frame.nir + frame.swir)
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["ndvi", "ndmi"])
        frame = frame[frame.ndvi.between(-1, 1) & frame.ndmi.between(-1, 1)]
        if len(frame) < 25:
            raise ValueError(f"{field}: too few valid index observations")
        output = config.GOLD_FEATURES_DIR / f"{field}_gold.parquet"
        frame.to_parquet(output, index=False)
        frame.to_csv(output.with_suffix(".csv"), index=False)
        outputs[field] = output
    return outputs
