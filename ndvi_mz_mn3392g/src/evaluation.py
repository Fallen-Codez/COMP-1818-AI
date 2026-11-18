import numpy as np
import pandas as pd
from scipy.ndimage import label as connected_components
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from . import config


def metrics(frame, x, labels, model):
    labels = np.asarray(labels)
    covered = np.ones(len(labels), dtype=bool)
    group_count = len(np.unique(labels))
    grid = np.full((int(frame.row.max()) + 1, int(frame.col.max()) + 1), -99, dtype=int)
    grid[frame.row.to_numpy(), frame.col.to_numpy()] = labels
    components = sum(connected_components(grid == lab)[1] for lab in np.unique(labels))
    edges = []
    for axis in (0, 1):
        a = np.take(grid, range(grid.shape[axis] - 1), axis=axis)
        b = np.take(grid, range(1, grid.shape[axis]), axis=axis)
        valid = (a != -99) & (b != -99)
        if valid.any():
            edges.extend((a[valid] == b[valid]).tolist())
    row = dict(field=frame.field.iloc[0], model=model, pixels=len(x),
               k=group_count, coverage=1.0, components=int(components),
               neighbour_agreement=float(np.mean(edges)) if edges else np.nan)
    if group_count >= 2 and len(labels) > group_count:
        row.update(silhouette=silhouette_score(x[covered], labels[covered]),
                   davies_bouldin=davies_bouldin_score(x[covered], labels[covered]),
                   calinski_harabasz=calinski_harabasz_score(x[covered], labels[covered]))
    else:
        row.update(silhouette=np.nan, davies_bouldin=np.nan, calinski_harabasz=np.nan)
    return row


def profiles(frame, labels, model):
    d = frame.assign(cluster=labels)
    summary = d.groupby("cluster").agg(pixels=("cluster", "size"),
                                        ndvi_mean=("ndvi", "mean"), ndmi_mean=("ndmi", "mean"),
                                        ndvi_sd=("ndvi", "std"), ndmi_sd=("ndmi", "std"))
    summary["hectares_approx"] = summary.pixels * config.PIXEL_SIZE_M ** 2 / 10000
    summary["fraction_covered"] = summary.pixels / len(frame)
    summary["field"] = frame.field.iloc[0]
    summary["model"] = model

    med_vi, med_mi = frame.ndvi.median(), frame.ndmi.median()
    summary["relative_pattern"] = [
        ("high NDVI" if vi >= med_vi else "low NDVI") + " / " +
        ("high NDMI" if mi >= med_mi else "low NDMI")
        for vi, mi in zip(summary.ndvi_mean, summary.ndmi_mean)
    ]
    return summary.reset_index()


def agreement(frame, results):
    rows = []
    for name in results:
        if name == "kmeans_tuned":
            continue
        labels = results[name]
        rows.append(dict(field=frame.field.iloc[0], comparator=name,
                         ari_vs_kmeans=float(adjusted_rand_score(
                             results["kmeans_tuned"], labels)) if len(labels) > 1 else np.nan,
                         compared_fraction=1.0))
    return pd.DataFrame(rows)
