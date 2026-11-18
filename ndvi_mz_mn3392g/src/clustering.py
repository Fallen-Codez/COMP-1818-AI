import numpy as np
import pandas as pd
import skfuzzy as fuzz
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from . import config


def kmeans_search(x):
    #Search k=2..5 on standardized indices and select the normalized elbow.

    # Silhouette and the 5% minimum zone fraction remain

    rows, candidates = [], []
    
    for k in range(2, min(6, len(x) // 15 + 1)):
        for init in ("k-means++", "random"): # cap search space
            for n_init in (10, 30):
                
                model = KMeans(n_clusters=k, init=init, n_init=n_init,
                               max_iter=500, random_state=config.SEED).fit(x)
                
                counts = np.bincount(model.labels_, minlength=k)
                
                row = dict(k=k, init=init, n_init=n_init,
                           silhouette=silhouette_score(x, model.labels_),
                           inertia=model.inertia_, iterations=model.n_iter_,
                           min_zone_fraction=counts.min() / len(x),
                           feasible=counts.min() / len(x) >= config.MIN_CLUSTER_FRACTION)
                
                rows.append(row)
                candidates.append((row, model))
    
    
    grid = pd.DataFrame(rows)
    valid = grid[grid.feasible]
    
    if valid.empty:
        raise ValueError("No K-Means candidate satisfies the minimum zone fraction")
    
    by_k = valid.groupby("k", as_index=False).agg(inertia=("inertia", "min"),
                                                  silhouette=("silhouette", "max"))
    
    if len(by_k) < 3:
        raise ValueError("At least three feasible k values are required for the elbow")
    
    k_norm = (by_k.k - by_k.k.min()) / (by_k.k.max() - by_k.k.min())
    inertia_range = by_k.inertia.max() - by_k.inertia.min()
    inertia_norm = (by_k.inertia - by_k.inertia.min()) / inertia_range
    by_k["elbow_distance"] = (1 - k_norm) - inertia_norm
    by_k.loc[by_k.index[[0, -1]], "elbow_distance"] = 0.0
    elbow_k = int(by_k.loc[by_k.elbow_distance.idxmax(), "k"])
    grid = grid.merge(by_k[["k", "elbow_distance"]], on="k", how="left")
    contenders = [(row, model) for row, model in candidates
                  if row["feasible"] and row["k"] == elbow_k]
    best_row, best_model = max(contenders, key=lambda pair: (
        pair[0]["silhouette"], -pair[0]["inertia"],
        pair[0]["init"] == "k-means++", pair[0]["n_init"]))
    selection = dict(best_row, elbow_k=elbow_k,
                     silhouette_best_k=int(by_k.loc[by_k.silhouette.idxmax(), "k"]),
                     elbow_distance=float(by_k.loc[by_k.k == elbow_k, "elbow_distance"].iloc[0]))
    return grid, selection, best_model


def fuzzy_fit(x, k, m=2.0):
    centres, membership, _, _, _, _, fpc = fuzz.cluster.cmeans(x.T, c=k, m=m, error=1e-5, maxiter=500, seed=config.SEED)
    return centres, membership.T, fpc


def fuzzy_search(x, k):
    rows = []
    for m in (1.6, 2.0, 2.4):
        centres, membership, fpc = fuzzy_fit(x, k, m)
        labels = membership.argmax(axis=1)
        rows.append(dict(m=m, fpc=fpc, silhouette=silhouette_score(x, labels),
                         mean_max_membership=membership.max(axis=1).mean()))
    return pd.DataFrame(rows), max(rows, key=lambda r: r["fpc"])["m"]


def gmm_search(x, k):
    rows, fits = [], []
    for covariance in ("full", "diag", "tied"):
        for reg in (1e-6, 1e-3):
            model = GaussianMixture(n_components=k, covariance_type=covariance,
                                    reg_covar=reg, n_init=5, random_state=config.SEED).fit(x)
            row = dict(covariance_type=covariance, reg_covar=reg,
                       bic=model.bic(x), aic=model.aic(x), converged=model.converged_)
            rows.append(row)
            fits.append((row, model))
    return pd.DataFrame(rows), min(fits, key=lambda rm: rm[0]["bic"])[1]


def fit_field(frame):
    raw = frame[["ndvi", "ndmi"]].to_numpy() # inverse transform to log results
    
    scaler = StandardScaler().fit(raw)

    #doing it seperately for now to avoid any issues further down the pipeline
    x = scaler.transform(raw)
    
    kgrid, selected_k, tuned_km = kmeans_search(x)
    if selected_k["k"] != 3:
        raise ValueError(f"The elbow selected k={selected_k['k']}; review the three-zone design")
    baseline_km = KMeans(n_clusters=4, init="random", n_init=1,
                         max_iter=300, random_state=config.SEED).fit(raw)

    _, baseline_u, baseline_fpc = fuzzy_fit(raw, k=4, m=2.0)
    fgrid, tuned_m = fuzzy_search(x, k=3)
    _, tuned_u, tuned_fpc = fuzzy_fit(x, k=3, m=tuned_m)
    baseline_ggrid, baseline_gmm = gmm_search(raw, k=4)
    tuned_ggrid, tuned_gmm = gmm_search(x, k=3)

    labels = {
        "fcm_baseline": baseline_u.argmax(axis=1),
        "kmeans_baseline": baseline_km.labels_,
        "gmm_baseline": baseline_gmm.predict(raw),
        "fcm_tuned": tuned_u.argmax(axis=1),
        "kmeans_tuned": tuned_km.labels_,
        "gmm_tuned": tuned_gmm.predict(x),
    }
    uncertainty = {
        "fcm_baseline": 1 - baseline_u.max(axis=1),
        "fcm_tuned": 1 - tuned_u.max(axis=1),
        "gmm_baseline": 1 - baseline_gmm.predict_proba(raw).max(axis=1),
        "gmm_tuned": 1 - tuned_gmm.predict_proba(x).max(axis=1),
    }
    selected = {
        "baseline_kmeans": dict(k=4, init="random", n_init=1, max_iter=300,
                                random_state=config.SEED, scaling="none",
                                iterations=baseline_km.n_iter_),
        "tuned_kmeans": dict(selected_k, max_iter=500, random_state=config.SEED,
                             scaling="fieldwise StandardScaler"),
        "baseline_fcm": dict(k=4, m=2.0, fpc=baseline_fpc, scaling="none"),
        "tuned_fcm": dict(k=3, m=tuned_m, fpc=tuned_fpc,
                          scaling="fieldwise StandardScaler"),
        "baseline_gmm": dict(k=4, covariance_type=baseline_gmm.covariance_type,
                             reg_covar=baseline_gmm.reg_covar,
                             bic=baseline_gmm.bic(raw), scaling="none"),
        "tuned_gmm": dict(k=3, covariance_type=tuned_gmm.covariance_type,
                          reg_covar=tuned_gmm.reg_covar,
                          bic=tuned_gmm.bic(x), scaling="fieldwise StandardScaler"),
        "evaluation_space": "same standardized NDVI/NDMI for all labels",
    }
    return dict(x=x, raw=raw, scaler=scaler, labels=labels,
                kmeans_models={"baseline": baseline_km, "tuned": tuned_km},
                uncertainty=uncertainty,
                grids={"kmeans": kgrid, "fcm": fgrid,
                       "gmm_baseline": baseline_ggrid, "gmm_tuned": tuned_ggrid},
                selected=selected)
