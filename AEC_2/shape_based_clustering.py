import argparse
import os
import shutil
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataclasses import dataclass as _dataclass


@dataclass
class CFG:
    n_points: int = 256
    smooth_kernel: int = 9
    min_clusters: int = 3
    max_clusters: int = 8
    sample_per_cluster: int = 12
    random_state: int = 42
    line_mask_min_pixels: int = 150
    normalize_mode: str = "zscore"
    copy_images: bool = True
    max_images: int = 0
    silhouette_sample_size: int = 600
    copy_all_cluster_images: bool = True
    image_dir: str = r"C:\Users\user\Desktop\Study\data\AEC\강남\AEC"
    out_dir: str = rf"C:\Users\user\Desktop\Study\result\Case1\강남\{normalize_mode}"


@_dataclass
class KMeansResult:
    n_clusters: int
    cluster_centers_: np.ndarray
    inertia_: float


def parse_args() -> CFG:
    parser = argparse.ArgumentParser(
        description="Cluster AEC graph PNG files by curve shape."
    )
    parser.add_argument("--image-dir", default=CFG.image_dir)
    parser.add_argument("--out-dir", default=CFG.out_dir)
    parser.add_argument("--n-points", type=int, default=CFG.n_points)
    parser.add_argument("--smooth-kernel", type=int, default=CFG.smooth_kernel)
    parser.add_argument("--min-clusters", type=int, default=CFG.min_clusters)
    parser.add_argument("--max-clusters", type=int, default=CFG.max_clusters)

    parser.add_argument("--sample-per-cluster", type=int, default=CFG.sample_per_cluster)
    parser.add_argument("--random-state", type=int, default=CFG.random_state)
    parser.add_argument("--line-mask-min-pixels", type=int, default=CFG.line_mask_min_pixels)
    parser.add_argument(
        "--normalize-mode",
        choices=["zscore", "minmax", "none"],
        default=CFG.normalize_mode,
        help="How to normalize the extracted curve profile before clustering.",
    )
    parser.add_argument(
        "--no-copy-images",
        action="store_true",
        help="Do not copy example images into cluster folders.",
    )
    parser.add_argument(
        "--copy-all-cluster-images",
        action="store_true",
        default=CFG.copy_all_cluster_images,
        help="Copy every image into its cluster folder instead of only representative samples.",
    )
    parser.add_argument(
        "--representatives-only",
        action="store_false",
        dest="copy_all_cluster_images",
        help="Copy only representative images into cluster folders.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=CFG.max_images,
        help="If > 0, only process the first N PNG files. Useful for fast testing.",
    )
    parser.add_argument(
        "--silhouette-sample-size",
        type=int,
        default=CFG.silhouette_sample_size,
        help="Maximum number of samples used when estimating silhouette score.",
    )
    args = parser.parse_args()

    cfg = CFG(
        image_dir=args.image_dir,
        out_dir=args.out_dir,
        n_points=args.n_points,
        smooth_kernel=args.smooth_kernel,
        min_clusters=args.min_clusters,
        max_clusters=args.max_clusters,
        sample_per_cluster=args.sample_per_cluster,
        random_state=args.random_state,
        line_mask_min_pixels=args.line_mask_min_pixels,
        normalize_mode=args.normalize_mode,
        copy_images=not args.no_copy_images,
        max_images=args.max_images,
        silhouette_sample_size=args.silhouette_sample_size,
        copy_all_cluster_images=args.copy_all_cluster_images,
    )
    return cfg


def standardize_features(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (X - mean) / std


def pairwise_distances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    X_sq = np.sum(X * X, axis=1, keepdims=True)
    Y_sq = np.sum(Y * Y, axis=1, keepdims=True).T
    dist_sq = np.maximum(X_sq + Y_sq - 2.0 * (X @ Y.T), 0.0)
    return np.sqrt(dist_sq).astype(np.float32)


def fit_kmeans(X: np.ndarray, n_clusters: int, random_state: int, n_init: int = 20) -> Tuple[np.ndarray, KMeansResult]:
    if n_clusters < 1 or n_clusters > len(X):
        raise ValueError("Invalid n_clusters value.")

    best_labels = None
    best_centers = None
    best_inertia = None
    base_rng = np.random.default_rng(random_state)
    seeds = base_rng.integers(0, 1_000_000_000, size=n_init)

    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        center_indices = rng.choice(len(X), size=n_clusters, replace=False)
        centers = X[center_indices].copy()

        for _ in range(100):
            distances = pairwise_distances(X, centers)
            labels = np.argmin(distances, axis=1)

            new_centers = centers.copy()
            for cluster_id in range(n_clusters):
                members = X[labels == cluster_id]
                if len(members) == 0:
                    new_centers[cluster_id] = X[rng.integers(0, len(X))]
                else:
                    new_centers[cluster_id] = members.mean(axis=0)

            shift = float(np.linalg.norm(new_centers - centers))
            centers = new_centers
            if shift < 1e-4:
                break

        final_distances = pairwise_distances(X, centers)
        labels = np.argmin(final_distances, axis=1)
        inertia = float(np.sum((X - centers[labels]) ** 2))

        if best_inertia is None or inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()

    result = KMeansResult(
        n_clusters=n_clusters,
        cluster_centers_=best_centers,
        inertia_=float(best_inertia),
    )
    return best_labels.astype(np.int32), result


def silhouette_score(X: np.ndarray, labels: np.ndarray, sample_size: int, random_state: int) -> float:
    n_samples = len(X)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return -1.0

    if sample_size > 0 and n_samples > sample_size:
        rng = np.random.default_rng(random_state)
        selected = np.sort(rng.choice(n_samples, size=sample_size, replace=False))
        X = X[selected]
        labels = labels[selected]
        n_samples = len(X)

    distances = pairwise_distances(X, X)
    sil_values = np.zeros(n_samples, dtype=np.float32)

    for i in range(n_samples):
        same_cluster = labels == labels[i]
        same_cluster[i] = False

        if np.any(same_cluster):
            a = float(distances[i, same_cluster].mean())
        else:
            sil_values[i] = 0.0
            continue

        b = np.inf
        for other_label in unique_labels:
            if other_label == labels[i]:
                continue
            other_cluster = labels == other_label
            if np.any(other_cluster):
                b = min(b, float(distances[i, other_cluster].mean()))

        if max(a, b) < 1e-6:
            sil_values[i] = 0.0
        else:
            sil_values[i] = (b - a) / max(a, b)

    return float(sil_values.mean())


def pca_2d(X: np.ndarray) -> np.ndarray:
    if X.ndim != 2 or len(X) == 0:
        raise ValueError("PCA input must be a non-empty 2D array.")
    centered = X - X.mean(axis=0, keepdims=True)
    U, S, _ = np.linalg.svd(centered, full_matrices=False)
    coords = U[:, :2] * S[:2]
    if coords.shape[1] == 1:
        coords = np.concatenate([coords, np.zeros((len(coords), 1), dtype=coords.dtype)], axis=1)
    return coords.astype(np.float32)


def extract_blue_curve_mask(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    lower = np.array([85, 25, 80], dtype=np.uint8)
    upper = np.array([130, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def mask_to_profile(mask: np.ndarray, n_points: int) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("No blue curve pixels detected.")

    x_min, x_max = int(xs.min()), int(xs.max())
    if x_max <= x_min:
        raise ValueError("Invalid curve width detected.")

    width = x_max - x_min + 1
    curve_y = np.full(width, np.nan, dtype=np.float32)

    for x in range(x_min, x_max + 1):
        col_ys = ys[xs == x]
        if len(col_ys) > 0:
            curve_y[x - x_min] = float(np.median(col_ys))

    valid = np.where(~np.isnan(curve_y))[0]
    if len(valid) < 2:
        raise ValueError("Too few valid curve points detected.")

    full_index = np.arange(width, dtype=np.float32)
    curve_y = np.interp(full_index, valid.astype(np.float32), curve_y[valid])

    curve_y = -curve_y
    resample_x = np.linspace(0, width - 1, n_points, dtype=np.float32)
    profile = np.interp(resample_x, full_index, curve_y).astype(np.float32)
    return profile


def smooth_1d(values: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size == 1:
        return values.copy()

    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    padded = np.pad(values, (kernel_size // 2, kernel_size // 2), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed.astype(np.float32)


def normalize_profile(profile: np.ndarray, mode: str) -> np.ndarray:
    profile = profile.astype(np.float32)
    if mode == "none":
        return profile

    if mode == "minmax":
        lo = float(np.min(profile))
        hi = float(np.max(profile))
        if hi - lo < 1e-6:
            return np.zeros_like(profile)
        return (profile - lo) / (hi - lo)

    mean = float(np.mean(profile))
    std = float(np.std(profile))
    if std < 1e-6:
        return np.zeros_like(profile)
    return (profile - mean) / std


def extract_shape_feature(image_path: str, cfg: CFG) -> np.ndarray:
    image_bgr = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Image could not be read.")

    mask = extract_blue_curve_mask(image_bgr)
    if int(mask.sum() / 255) < cfg.line_mask_min_pixels:
        raise ValueError("Detected line mask is too small.")

    profile = mask_to_profile(mask, cfg.n_points)
    profile = smooth_1d(profile, cfg.smooth_kernel)
    profile = normalize_profile(profile, cfg.normalize_mode)
    return profile


def load_features(cfg: CFG) -> Tuple[pd.DataFrame, np.ndarray, List[dict]]:
    records = []
    features = []
    failures = []
    image_paths = []

    for name in sorted(os.listdir(cfg.image_dir)):
        if name.lower().endswith(".png"):
            image_paths.append(os.path.join(cfg.image_dir, name))
    if not image_paths:
        raise FileNotFoundError(f"No PNG files found in: {cfg.image_dir}")

    for idx, image_path in enumerate(image_paths, start=1):
        image_name = os.path.basename(image_path)
        patient_id = os.path.splitext(image_name)[0]
        try:
            feature = extract_shape_feature(image_path, cfg)
            features.append(feature)
            records.append(
                {
                    "patient_id": patient_id,
                    "image_name": image_name,
                    "image_path": image_path,
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "image_name": image_name,
                    "image_path": image_path,
                    "error": str(exc),
                }
            )

    if not features:
        raise RuntimeError("No valid graph profiles were extracted.")

    df = pd.DataFrame(records)
    X = np.vstack(features).astype(np.float32)
    return df, X, failures


def run_clustering(
    X_scaled: np.ndarray,
    cfg: CFG
) -> Tuple[np.ndarray, KMeansResult, pd.DataFrame]:
    rows = []
    best_score = -np.inf
    best_labels = None
    best_model = None

    upper = min(cfg.max_clusters, len(X_scaled) - 1)
    lower = min(cfg.min_clusters, upper)

    if upper < 2:
        raise ValueError("At least 2 valid images are required for clustering.")

    for k in range(lower, upper + 1):
        labels, model = fit_kmeans(
            X_scaled,
            n_clusters=k,
            random_state=cfg.random_state + k,
            n_init=20,
        )

        score = silhouette_score(
            X_scaled,
            labels,
            sample_size=cfg.silhouette_sample_size,
            random_state=cfg.random_state + k,
        )

        rows.append({
            "n_clusters": k,
            "silhouette_score": score,
            "inertia": model.inertia_,
        })

        if score > best_score:
            best_score = score
            best_labels = labels
            best_model = model

    score_df = pd.DataFrame(rows).sort_values(
        by=["silhouette_score", "n_clusters"],
        ascending=[False, True],
    )

    return best_labels, best_model, score_df


def distance_to_center(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> np.ndarray:
    dists = np.zeros(len(X), dtype=np.float32)
    for i in range(len(X)):
        dists[i] = float(np.linalg.norm(X[i] - centers[labels[i]]))
    return dists


def save_cluster_plots(df: pd.DataFrame, X: np.ndarray, labels: np.ndarray, cfg: CFG) -> None:
    plot_dir = os.path.join(cfg.out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    unique_labels = sorted(df["cluster_id"].unique())
    x_axis = np.linspace(0.0, 1.0, X.shape[1])

    for cluster_id in unique_labels:
        idx = np.where(labels == cluster_id)[0]
        curves = X[idx]
        mean_curve = curves.mean(axis=0)

        plt.figure(figsize=(11, 4.5))
        for curve in curves[: min(len(curves), 80)]:
            plt.plot(x_axis, curve, color="steelblue", alpha=0.08, linewidth=1)
        plt.plot(x_axis, mean_curve, color="crimson", linewidth=2.5, label="cluster mean")
        plt.title(f"Cluster {cluster_id} shape profiles (n={len(curves)})")
        plt.xlabel("Normalized instance number")
        plt.ylabel(f"Curve profile ({cfg.normalize_mode})")
        plt.grid(alpha=0.2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"cluster_{cluster_id}_profiles.png"), dpi=200)
        plt.close()


def save_pca_plot(X: np.ndarray, labels: np.ndarray, cfg: CFG) -> None:
    if len(X) < 2:
        return

    coords = pca_2d(X)
    plt.figure(figsize=(8, 6))
    for cluster_id in sorted(np.unique(labels)):
        idx = labels == cluster_id
        plt.scatter(coords[idx, 0], coords[idx, 1], s=18, alpha=0.7, label=f"Cluster {cluster_id}")
    plt.title("AEC graph shape clustering (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "pca_clusters.png"), dpi=200)
    plt.close()


def export_cluster_examples(df: pd.DataFrame, cfg: CFG) -> None:
    example_root = os.path.join(cfg.out_dir, "cluster_examples")
    os.makedirs(example_root, exist_ok=True)

    for cluster_id, cluster_df in df.groupby("cluster_id"):
        cluster_dir = os.path.join(example_root, f"cluster_{cluster_id}")
        os.makedirs(cluster_dir, exist_ok=True)

        cluster_df = cluster_df.sort_values("distance_to_center").copy()
        cluster_df[["patient_id", "image_name", "image_path", "distance_to_center"]].to_csv(
            os.path.join(cluster_dir, "all_members.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        top_df = cluster_df.sort_values("distance_to_center").head(cfg.sample_per_cluster)
        top_df[["patient_id", "image_name", "distance_to_center"]].to_csv(
            os.path.join(cluster_dir, "representatives.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        if cfg.copy_images:
            copy_df = cluster_df if cfg.copy_all_cluster_images else top_df
            for _, row in copy_df.iterrows():
                src = row["image_path"]
                dst = os.path.join(cluster_dir, row["image_name"])
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)


def save_outputs(
    df: pd.DataFrame,
    X_scaled: np.ndarray,
    labels: np.ndarray,
    model: KMeansResult,
    score_df: pd.DataFrame,
    failures: List[dict],
    cfg: CFG,
) -> None:
    df = df.copy()
    df["cluster_id"] = labels
    df["distance_to_center"] = distance_to_center(X_scaled, labels, model.cluster_centers_)

    summary_df = (
        df.groupby("cluster_id")
        .agg(
            count=("patient_id", "count"),
            mean_distance=("distance_to_center", "mean"),
            median_distance=("distance_to_center", "median"),
        )
        .reset_index()
        .sort_values("cluster_id")
    )
    if len(score_df) > 0:
        score_df.to_csv(
            os.path.join(cfg.out_dir, "cluster_selection_scores.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    if failures:
        pd.DataFrame(failures).to_csv(
            os.path.join(cfg.out_dir, "failed_images.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    save_cluster_plots(df, X_scaled, labels, cfg)
    save_pca_plot(X_scaled, labels, cfg)
    export_cluster_examples(df, cfg)


def main():
    cfg = parse_args()
    os.makedirs(cfg.out_dir, exist_ok=True)

    df, X_raw, failures = load_features(cfg)
    X_scaled = standardize_features(X_raw)
    labels, model, score_df = run_clustering(X_scaled, cfg)

    save_outputs(df, X_scaled, labels, model, score_df, failures, cfg)

if __name__ == "__main__":
    main()
