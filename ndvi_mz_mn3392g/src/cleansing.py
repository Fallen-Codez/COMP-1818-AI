import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.features import rasterize
from shapely import transform as shape_transform
from . import config
from .ingestion import field_geometry


def clean_silver():
    outputs = {}
    for field, fid in config.FIELDS.items():
        source = config.RAW_DIR / f"{field}_s2_l2a_20240328.tif"

        # silver level clean up
        with rasterio.open(source) as ds:
            bands = ds.read()
            geometry, _ = field_geometry(fid)
            projected = shape_transform(
                geometry, Transformer.from_crs(32718, ds.crs, always_xy=True).transform,
                interleaved=False,
            )
            inside = rasterize([(projected, 1)], out_shape=(ds.height, ds.width),
                               transform=ds.transform, dtype="uint8").astype(bool)
            
            # SCL 4=vegetation, 5=bare soil; reject shadows, water, cloud and cirrus.
            
            valid = inside & np.isin(bands[3], [4, 5]) & np.all(bands[:3] > 1000, axis=0)
            rows, cols = np.nonzero(valid)

            if len(rows) < 25:
                raise ValueError(f"{field}: only {len(rows)} valid pixels; inspect cloud mask/acquisition")
            
            xx, yy = rasterio.transform.xy(ds.transform, rows, cols, offset="center")
            
            frame = pd.DataFrame({
                "field": field, "fid": fid, "date": config.ACQUISITION_DATE,
                "crs": ds.crs.to_string(),
                "row": rows, "col": cols, "x": xx, "y": yy, "scl": bands[3, rows, cols],
                # ESA processing baseline >=04.00 BOA_ADD_OFFSET=-1000, scale=10000.
                "red": (bands[0, rows, cols].astype(float) - 1000) / 10000,
                "nir": (bands[1, rows, cols].astype(float) - 1000) / 10000,
                "swir": (bands[2, rows, cols].astype(float) - 1000) / 10000,
            })
            output = config.PROCESSED_DIR / f"{field}_silver.parquet"
            frame.to_parquet(output, index=False)
            frame.to_csv(output.with_suffix(".csv"), index=False)
            outputs[field] = output
            print(f"{field}: {len(frame)} valid of {int(inside.sum())} polygon pixels; "
                  f"SCL distribution {dict(zip(*np.unique(bands[3][inside], return_counts=True)))}")
    return outputs
