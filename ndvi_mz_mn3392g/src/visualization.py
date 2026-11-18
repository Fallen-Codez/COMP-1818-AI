# matplot lib for assignment compatibility // mn3392g
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import rasterio
from . import config


def field_output(field):
    path = config.OUTPUTS_DIR / field
    path.mkdir(parents=True, exist_ok=True)
    return path


def index_maps(frame, field):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, column, cmap in zip(axes, ("ndvi", "ndmi"), ("YlGn", "BrBG")):
        grid = np.full((frame.row.max() + 1, frame.col.max() + 1), np.nan)
        grid[frame.row, frame.col] = frame[column]
        im = ax.imshow(grid, cmap=cmap, vmin=-1 if column == "ndmi" else 0,
                       vmax=1, interpolation="nearest")
        ax.set_title(f"{field} | {column.upper()}")
        ax.set_xlabel("Raster column (10 m)"); ax.set_ylabel("Raster row (10 m)")
        fig.colorbar(im, ax=ax, label=column.upper(), shrink=.75)
    path = field_output(field) / "02_indices.png"
    fig.savefig(path, dpi=180); plt.close(fig)
    return path


def raw_preview(field):
    source = config.RAW_DIR / f"{field}_s2_l2a_20240328.tif"
    with rasterio.open(source) as ds:
        red, nir, swir, scl = ds.read()
    valid = np.isin(scl, [4, 5]) & (red > 1000) & (nir > 1000) & (swir > 1000)
    channels = []
    for band in (nir, red, swir):
        good = band[valid]
        lo, hi = np.percentile(good, [2, 98]) if len(good) else (0, 1)
        channels.append(np.clip((band.astype(float) - lo) / max(hi - lo, 1), 0, 1))
    rgb = np.dstack(channels)
    rgb[~valid] = 1
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].imshow(rgb, interpolation="nearest")
    axes[0].set_title("False colour: NIR / red / SWIR")
    im = axes[1].imshow(np.ma.masked_where(~valid, nir), cmap="gray", interpolation="nearest")
    axes[1].set_title("Original NIR band (digital numbers)")
    fig.colorbar(im, ax=axes[1], shrink=.8, label="NIR DN")
    for ax in axes:
        ax.set_xlabel("Raster column (10 m)"); ax.set_ylabel("Raster row (10 m)")
    fig.suptitle(f"{field}: raw satellite crop, 2024-03-28 (display stretch only)")
    path = field_output(field) / "01_raw_preview.png"
    fig.savefig(path, dpi=180); plt.close(fig)
    return path


def tuning_plot(result, field):
    #Method figure: inertia elbow and silhouette, with their different choices
    grid = result["grids"]["kmeans"]
    summary = grid[grid.feasible].groupby("k", as_index=False).agg(
        inertia=("inertia", "min"), silhouette=("silhouette", "max"),
        elbow_distance=("elbow_distance", "max"))
    selected = result["selected"]["tuned_kmeans"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    axes[0].plot(summary.k, summary.inertia, "o-", color="#566a9e")
    axes[1].plot(summary.k, summary.silhouette, "o-", color="#63a39b")
    axes[0].set(xlabel="Number of zones k", ylabel="Minimum standardized inertia",
                title=f"Inertia elbow = {selected['elbow_k']}")
    axes[1].set(xlabel="Number of zones k", ylabel="Maximum silhouette",
                title=f"Highest silhouette at k = {selected['silhouette_best_k']}")
    for ax in axes:
        ax.axvline(selected["elbow_k"], ls="--", c="#c25136", label="Elbow-selected k=3")
        ax.set_xticks(summary.k)
        ax.legend(fontsize=8)
    fig.suptitle(f"{field}: K-Means selection on standardized NDVI and NDMI")
    path = field_output(field) / "03_method_kmeans_elbow.png"
    fig.savefig(path, dpi=180); plt.close(fig)
    return path


def distributions(frame, result, profiles, field):

    labels = result["labels"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    tuned = profiles[(profiles.field == field) & (profiles.model == "kmeans_tuned")]
    for _, zone in tuned.iterrows():
        points = frame[np.asarray(labels["kmeans_tuned"]) == zone.cluster]
        axes[0].scatter(points.ndvi, points.ndmi, s=13, alpha=.7,
                        label=f"Zone {int(zone.cluster)} ({len(points)} pixels)")
    axes[0].set(xlabel="NDVI", ylabel="NDMI", title="Tuned K-Means: feature distribution")
    axes[0].legend(fontsize=7)
    models = ["fcm_baseline", "kmeans_baseline", "gmm_baseline",
              "fcm_tuned", "kmeans_tuned", "gmm_tuned"]
    shares = []
    for name in models:
        assigned = np.asarray(labels[name])
        ranks = sorted(np.unique(assigned), key=lambda v: frame.ndvi.to_numpy()[assigned == v].mean())
        shares.append([np.mean(assigned == v) for v in ranks] + [0] * (4 - len(ranks)))
    colours = ["#566a9e", "#63a39b", "#e1c46c", "#d97958"]
    bottom = np.zeros(len(models))
    for rank in range(4):
        values = np.asarray(shares)[:, rank]
        axes[1].bar(models, values, bottom=bottom, color=colours[rank],
                    label=f"NDVI rank {rank + 1}")
        bottom += values
    axes[1].set(ylabel="Fraction of valid pixels", title="Relative zone sizes", ylim=(0, 1))
    axes[1].tick_params(axis="x", labelrotation=45)
    axes[1].legend(fontsize=7)
    path = field_output(field) / "06_results_distributions.png"
    fig.savefig(path, dpi=180); plt.close(fig)
    return path


def matched_zone_maps(field_results, phase):

    baseline = phase == "baseline"
    k = 4 if baseline else 3
    names = {"FCM": f"fcm_{phase}", "K-Means": f"kmeans_{phase}",
             "GMM": f"gmm_{phase}"}
    colours = ["#566a9e", "#63a39b", "#e1c46c", "#d97958"][:k]
    cmap = ListedColormap(colours)
    fig, axes = plt.subplots(len(field_results), 3, figsize=(14, 8))
    for row, (field, (frame, result)) in enumerate(field_results.items()):
        for col, (name, key) in enumerate(names.items()):
            ax = axes[row, col]
            labels = result["labels"][key]
            groups = sorted(set(labels), key=lambda v: frame.ndvi.to_numpy()[labels == v].mean())
            ranks = {group: rank for rank, group in enumerate(groups)}
            grid = np.full((int(frame.row.max()) + 1, int(frame.col.max()) + 1), np.nan)
            grid[frame.row, frame.col] = [ranks[value] for value in labels]
            ax.imshow(np.ma.masked_invalid(grid), cmap=cmap,
                      norm=BoundaryNorm(np.arange(-.5, k + .5), k),
                      interpolation="nearest")
            ax.set_title(f"{field} | {name} ({len(groups)} zones)", fontsize=10)
            ax.set_xlabel("Raster column (10 m)"); ax.set_ylabel("Raster row (10 m)")
    fig.suptitle(f"{'Starting' if baseline else 'Tuned'} configurations: {k} zones; "
                 "colours rank mean NDVI & NDMI within each model")
    fig.legend(handles=[Patch(facecolor=colour, label=f"NDVI rank {i+1}")
                        for i, colour in enumerate(colours)], loc="lower center",
               ncol=k, fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, .94), h_pad=3)
    path = config.OUTPUTS_DIR / ("04a_results_baseline_k4.png" if baseline
                                 else "04b_results_tuned_k3.png")
    fig.savefig(path, dpi=180); plt.close(fig)
    return path


def _metric_bars(metrics_table, columns, titles, output_name):

    fig, axes = plt.subplots(2, len(columns), figsize=(5 * len(columns), 8),
                             constrained_layout=True)
    models = ["fcm", "kmeans", "gmm"]
    for row, field in enumerate(config.FIELDS):
        subset = metrics_table[metrics_table.field == field].set_index("model")
        for col, (metric, title) in enumerate(zip(columns, titles)):
            ax = axes[row, col]
            for shift, stage, colour in [(-.19, "baseline", "#566a9e"),
                                         (.19, "tuned", "#d97958")]:
                values = [subset.loc[f"{model}_{stage}", metric] for model in models]
                ax.bar(np.arange(3) + shift, values, width=.38,
                       color=colour, label=f"{stage}: k={4 if stage=='baseline' else 3}")
                for pos, value in enumerate(values):
                    ax.annotate(f"{value:.2f}" if metric != "components" else f"{value:.0f}",
                                (pos + shift, value), xytext=(0, 3),
                                textcoords="offset points", ha="center", fontsize=7)
            ax.set_xticks(np.arange(3), ["FCM", "K-Means", "GMM"])
            ax.set_title(f"{field} | {title}")
            ax.legend(fontsize=8)
            ax.margins(y=.18)
    path = config.OUTPUTS_DIR / output_name
    fig.savefig(path, dpi=180); plt.close(fig)
    return path


def quality_plot(metrics_table):
    return _metric_bars(metrics_table,
                        ["silhouette", "davies_bouldin", "calinski_harabasz"],
                        ["Silhouette ↑", "Davies–Bouldin ↓", "Calinski–Harabasz ↑"],
                        "07a_evaluation_cluster_quality.png")


def spatial_plot(metrics_table):
    return _metric_bars(metrics_table,
                        ["neighbour_agreement", "components"],
                        ["Same-zone neighbour fraction ↑", "Connected components ↓"],
                        "07b_evaluation_spatial_coherence.png")
