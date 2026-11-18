import json
import joblib
import numpy as np
import pandas as pd
import rasterio
from src import config
from src.ingestion import download_dataset
from src.cleansing import clean_silver
from src.feature_engineering import create_gold_features
from src.clustering import fit_field
from src.evaluation import agreement, metrics, profiles
from src.visualization import (distributions, field_output, index_maps,
                               matched_zone_maps, quality_plot, raw_preview,
                               spatial_plot, tuning_plot)


def run_analysis():
    """Fit and evaluate one field at a time; keep combined summaries at root."""
    all_metrics, all_profiles, all_agreements, all_pixels = [], [], [], []
    selected, field_results = {}, {}
    for field in config.FIELDS:
        destination = field_output(field)
        frame = pd.read_parquet(config.GOLD_FEATURES_DIR / f"{field}_gold.parquet")
        result = fit_field(frame)
        field_results[field] = (frame, result)
        selected[field] = result["selected"]
        model_dir = config.MODELS_DIR / field
        model_dir.mkdir(parents=True, exist_ok=True)
        for stage, model in result["kmeans_models"].items():
            joblib.dump({"scaler": result["scaler"] if stage == "tuned" else None,
                         "model": model, "features": ["ndvi", "ndmi"],
                         "field": field, "acquisition_date": config.ACQUISITION_DATE,
                         "scaling": "fieldwise StandardScaler" if stage == "tuned" else "none"},
                        model_dir / f"kmeans_{stage}.joblib")
        result["grids"]["kmeans"].to_csv(destination / "03_kmeans_search.csv", index=False)
        field_metrics, field_profiles, field_pixels = [], [], []
        for name, labels in result["labels"].items():
            field_metrics.append(metrics(frame, result["x"], labels, name))
            field_profiles.append(profiles(frame, labels, name))
            output = frame[["field", "fid", "date", "crs", "row", "col",
                            "x", "y", "ndvi", "ndmi"]].copy()
            output["model"] = name
            output["cluster"] = labels.astype(int)
            output["uncertainty"] = result["uncertainty"].get(name, float("nan"))
            field_pixels.append(output)
        metrics_frame = pd.DataFrame(field_metrics)
        profiles_frame = pd.concat(field_profiles, ignore_index=True)
        agreement_frame = agreement(frame, result["labels"])
        pixels_frame = pd.concat(field_pixels, ignore_index=True)
        metrics_frame.to_csv(destination / "07_evaluation_metrics.csv", index=False)
        profiles_frame.to_csv(destination / "08_results_zone_profiles.csv", index=False)
        agreement_frame.to_csv(destination / "09_evaluation_agreement.csv", index=False)
        pixels_frame.to_parquet(destination / "10_clustered_pixels.parquet", index=False)
        pixels_frame.to_csv(destination / "10_clustered_pixels.csv", index=False)
        all_metrics.extend(field_metrics)
        all_profiles.append(profiles_frame)
        all_agreements.append(agreement_frame)
        all_pixels.append(pixels_frame)
        index_maps(frame, field)
        tuning_plot(result, field)
        distributions(frame, result, profiles_frame, field)
        source = config.RAW_DIR / f"{field}_s2_l2a_20240328.tif"
        with rasterio.open(source) as ds:
            profile = ds.profile.copy()
            profile.update(count=1, dtype="uint8", nodata=0)
            for stage in ("baseline", "tuned"):
                grid = np.zeros((ds.height, ds.width), dtype="uint8")
                grid[frame.row, frame.col] = result["labels"][f"kmeans_{stage}"] + 1
                k = 4 if stage == "baseline" else 3
                with rasterio.open(destination / f"05_{stage}_k{k}_zones.tif", "w", **profile) as dst:
                    dst.write(grid, 1)
    matched_zone_maps(field_results, "baseline")
    matched_zone_maps(field_results, "tuned")
    metrics_table = pd.DataFrame(all_metrics)
    metrics_table.to_csv(config.OUTPUTS_DIR / "07_cluster_metrics.csv", index=False)
    pd.concat(all_profiles, ignore_index=True).to_csv(
        config.OUTPUTS_DIR / "08_zone_profiles.csv", index=False)
    pd.concat(all_agreements, ignore_index=True).to_csv(
        config.OUTPUTS_DIR / "09_model_agreement.csv", index=False)
    combined = pd.concat(all_pixels, ignore_index=True)
    combined.to_parquet(config.OUTPUTS_DIR / "10_clustered_pixels.parquet", index=False)
    combined.to_csv(config.OUTPUTS_DIR / "10_clustered_pixels.csv", index=False)
    (config.OUTPUTS_DIR / "03_selected_parameters.json").write_text(
        json.dumps(selected, indent=2, default=lambda value: value.item()
                   if isinstance(value, np.generic) else str(value)))
    quality_plot(metrics_table)
    spatial_plot(metrics_table)
    return metrics_table


def main():
    download_dataset()
    for field in config.FIELDS:
        raw_preview(field)
    clean_silver()
    create_gold_features()
    return run_analysis()


if __name__ == "__main__":
    print(main().round(3).to_string(index=False))
