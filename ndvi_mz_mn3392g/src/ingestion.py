import hashlib
import json
import sqlite3
from contextlib import ExitStack
import numpy as np
import rasterio
import requests
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds
from shapely import wkb, transform as shape_transform
from . import config


def _get_json(url, **kwargs):
    response = requests.get(url, timeout=60, **kwargs)
    response.raise_for_status()
    return response.json()


def download_boundaries():
    record = _get_json(f"https://zenodo.org/api/records/{config.ZENODO_RECORD}")
    source = next(f for f in record["files"] if f["key"].startswith("FIELDS_"))
    if config.BOUNDARY_PATH.exists():
        digest = hashlib.md5(config.BOUNDARY_PATH.read_bytes()).hexdigest()
        if digest == source["checksum"].split(":")[-1]:
            return config.BOUNDARY_PATH
    response = requests.get(source["links"]["self"], timeout=180)
    response.raise_for_status()
    digest = hashlib.md5(response.content).hexdigest()
    assert digest == source["checksum"].split(":")[-1], "Boundary checksum mismatch"
    config.BOUNDARY_PATH.write_bytes(response.content)
    return config.BOUNDARY_PATH


def field_geometry(fid):
    with sqlite3.connect(config.BOUNDARY_PATH) as conn:
        row = conn.execute(
            f'SELECT CROP, year_class, S_DATE_C, H_DATE_C, geom FROM "{config.BOUNDARY_TABLE}" WHERE fid=?',
            (fid,),
        ).fetchone()
    if row is None:
        raise ValueError(f"No boundary with fid={fid}")

    geometry = wkb.loads(row[4][40:])
    return geometry, dict(crop=row[0], year_class=row[1], sowing=row[2], harvest=row[3])


def download_dataset():
    """Create two real, field-cropped source rasters, repeatably and idempotently."""
    outputs = [config.RAW_DIR / f"{name}_s2_l2a_20240328.tif" for name in config.FIELDS]
    manifest_path = config.RAW_DIR / "source_manifest.json"
    if config.BOUNDARY_PATH.exists() and manifest_path.exists() and all(p.exists() for p in outputs):
        manifest = json.loads(manifest_path.read_text())
        checksums = manifest.get("raw_sha256", {})
        if (manifest.get("item_id") == config.STAC_ITEM
                and hashlib.md5(config.BOUNDARY_PATH.read_bytes()).hexdigest() == manifest.get("boundary_md5")
                and all(hashlib.sha256(p.read_bytes()).hexdigest() == checksums.get(p.name)
                        for p in outputs)):
            return outputs
    download_boundaries()
    item = _get_json(
        f"https://planetarycomputer.microsoft.com/api/stac/v1/collections/"
        f"{config.STAC_COLLECTION}/items/{config.STAC_ITEM}"
    )
    if item["properties"]["datetime"][:10] != config.ACQUISITION_DATE:
        raise ValueError("STAC item date changed")
    if not all(p.exists() for p in outputs):
        signed = {
            band: _get_json("https://planetarycomputer.microsoft.com/api/sas/v1/sign",
                            params={"href": item["assets"][band]["href"]})["href"]
            for band in config.BANDS
        }

        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
            with ExitStack() as stack:
                sources = {band: stack.enter_context(rasterio.open(href)) for band, href in signed.items()}
                red = sources["B04"]
                for (name, fid), destination in zip(config.FIELDS.items(), outputs):
                    if destination.exists():
                        continue
                    geom, info = field_geometry(fid)
                    projected = shape_transform(
                        geom, Transformer.from_crs(32718, red.crs, always_xy=True).transform,
                        interleaved=False,
                    )
                    window = from_bounds(*projected.bounds, transform=red.transform)
                    window = window.round_offsets().round_lengths()
                    transform = red.window_transform(window)
                    h, w = int(window.height), int(window.width)
                    stack_data = np.zeros((4, h, w), dtype=np.uint16)
                    stack_data[0] = red.read(1, window=window)
                    for index, band in enumerate(("B08", "B11", "SCL"), start=1):
                        with WarpedVRT(
                            sources[band], crs=red.crs, transform=transform,
                            width=w, height=h,
                            resampling=Resampling.nearest if band == "SCL" else Resampling.bilinear,
                        ) as vrt:
                            stack_data[index] = vrt.read(1)
                    profile = red.profile.copy()
                    profile.update(driver="GTiff", width=w, height=h, count=4,
                                   transform=transform, dtype="uint16", compress="deflate")
                    with rasterio.open(destination, "w", **profile) as out:
                        out.write(stack_data)
                        out.descriptions = config.BANDS
                        out.update_tags(item_id=config.STAC_ITEM, field_fid=fid,
                                        crop=info["crop"], year_class=info["year_class"])
    manifest = {
        "boundary_doi": f"10.5281/zenodo.{config.ZENODO_RECORD}",
        "boundary_md5": hashlib.md5(config.BOUNDARY_PATH.read_bytes()).hexdigest(),
        "sentinel_collection": config.STAC_COLLECTION,
        "item_id": config.STAC_ITEM, "datetime": item["properties"]["datetime"],
        "scene_cloud_percent": item["properties"]["eo:cloud_cover"],
        "bands": list(config.BANDS), "fields": config.FIELDS,
        "raw_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs},
        "source_asset_hrefs": {b: item["assets"][b]["href"] for b in config.BANDS},
        "notes": "B11 is bilinear-resampled from 20 m to B04's 10 m grid; SCL is nearest-resampled.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return outputs
