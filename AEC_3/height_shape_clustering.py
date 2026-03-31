import argparse
import os
import shutil
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class CFG:
    image_dir: str = r"C:\Users\user\Desktop\Study\data\AEC\강남\AEC_2"
    out_dir: str = r"C:\Users\user\Desktop\Study\result\AEC_3\강남"
    n_clusters: int = 2
    n_points: int = 64
    smooth_kernel: int = 9
    line_mask_min_pixels: int = 150
    max_images: int = 0
    random_state: int = 42
    shape_weight: float = 1.0
    height_weight: float = 3.0
    sample_per_cluster: int = 20
    copy_images: bool = True
    copy_all_cluster_images: bool = True


@dataclass
class KMeansResult:
    n_clusters: int
    cluster_centers_: np.ndarray
    inertia_: float


def parse_args() -> CFG:
    parser = argparse.ArgumentParser(
        description="Cluster AEC graph PNG images using both vertical height and curve shape."
    )
    parser.add_argument("--image-dir", default=CFG.image_dir)
    parser.add_argument("--out-dir", default=CFG.out_dir)
    parser.add_argument("--n-clusters", type=int, default=CFG.n_clusters)
    parser.add_argument("--n-points", type=int, default=CFG.n_points)
    parser.add_argument("--smooth-kernel", type=int, default=CFG.smooth_kernel)
    parser.add_argument("--line-mask-min-pixels", type=int, default=CFG.line_mask_min_pixels)
    parser.add_argument("--max-images", type=int, default=CFG.max_images)
    parser.add_argument("--random-state", type=int, default=CFG.random_state)
    parser.add_argument("--shape-weight", type=float, default=CFG.shape_weight)
    parser.add_argument("--height-weight", type=float, default=CFG.height_weight)
    parser.add_argument("--sample-per-cluster", type=int, default=CFG.sample_per_cluster)
    parser.add_argument("--no-copy-images", action="store_true")
    parser.add_argument(
        "--representatives-only",
        action="store_false",
        dest="copy_all_cluster_images",
        help="Copy only representative images into each cluster folder.",
    )
    parser.add_argument(
        "--copy-all-cluster-images",
        action="store_true",
        default=CFG.copy_all_cluster_images,
        help="Copy every image into each cluster folder.",
    )
    args = parser.parse_args()

    return CFG(
        image_dir=args.image_dir,
        out_dir=args.out_dir,
        n_clusters=args.n_clusters,
        n_points=args.n_points,
        smooth_kernel=args.smooth_kernel,
        line_mask_min_pixels=args.line_mask_min_pixels,
        max_images=args.max_images,
        random_state=args.random_state,
        shape_weight=args.shape_weight,
        height_weight=args.height_weight,
        sample_per_cluster=args.sample_per_cluster,
        copy_images=not args.no_copy_images,
        copy_all_cluster_images=args.copy_all_cluster_images,
    )


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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


def fit_kmeans(
    X: np.ndarray,
    n_clusters: int,
    random_state: int,
    n_init: int = 20,
) -> Tuple[np.ndarray, KMeansResult]:
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


def pca_2d(X: np.ndarray) -> np.ndarray:
    if X.ndim != 2 or len(X) == 0:
        raise ValueError("PCA input must be a non-empty 2D array.")

    centered = X - X.mean(axis=0, keepdims=True)
    U, S, _ = np.linalg.svd(centered, full_matrices=False)
    coords = U[:, :2] * S[:2]
    if coords.shape[1] == 1:
        coords = np.concatenate(
            [coords, np.zeros((len(coords), 1), dtype=coords.dtype)],
            axis=1,
        )
    return coords.astype(np.float32)


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


def extract_blue_curve_mask(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    lower = np.array([85, 25, 80], dtype=np.uint8)
    upper = np.array([130, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def mask_to_absolute_profile(mask: np.ndarray, n_points: int) -> np.ndarray:
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

    height = max(mask.shape[0] - 1, 1)
    curve_height = 1.0 - (curve_y / float(height))
    resample_x = np.linspace(0, width - 1, n_points, dtype=np.float32)
    profile = np.interp(resample_x, full_index, curve_height).astype(np.float32)
    return profile


def extract_summary_features(profile: np.ndarray) -> Dict[str, float]:
    x_axis = np.linspace(0.0, 1.0, len(profile), dtype=np.float32)
    peak_idx = int(np.argmax(profile))
    trough_idx = int(np.argmin(profile))
    slopes = np.diff(profile)
    curvature = np.diff(profile, n=2) if len(profile) >= 3 else np.array([], dtype=np.float32)

    midpoint = len(profile) // 2
    left_half = profile[:midpoint] if midpoint > 0 else profile
    right_half = profile[midpoint:] if midpoint > 0 else profile

    return {
        "mean_height": float(np.mean(profile)),
        "median_height": float(np.median(profile)),
        "std_height": float(np.std(profile)),
        "peak_height": float(np.max(profile)),
        "trough_height": float(np.min(profile)),
        "height_range": float(np.max(profile) - np.min(profile)),
        "peak_position": float(x_axis[peak_idx]),
        "trough_position": float(x_axis[trough_idx]),
        "mean_slope": float(np.mean(slopes)) if len(slopes) > 0 else 0.0,
        "mean_abs_slope": float(np.mean(np.abs(slopes))) if len(slopes) > 0 else 0.0,
        "mean_abs_curvature": float(np.mean(np.abs(curvature))) if len(curvature) > 0 else 0.0,
        "left_half_mean_height": float(np.mean(left_half)),
        "right_half_mean_height": float(np.mean(right_half)),
    }


def build_feature_vector(profile: np.ndarray, cfg: CFG) -> Tuple[np.ndarray, Dict[str, float]]:
    summary = extract_summary_features(profile)

    shape_profile = profile - float(np.mean(profile))
    shape_std = float(np.std(shape_profile))
    if shape_std >= 1e-6:
        shape_profile = shape_profile / shape_std
    else:
        shape_profile = np.zeros_like(shape_profile)

    summary_vector = np.array(
        [
            summary["mean_height"],
            summary["median_height"],
            summary["std_height"],
            summary["peak_height"],
            summary["trough_height"],
            summary["height_range"],
            summary["peak_position"],
            summary["trough_position"],
            summary["mean_slope"],
            summary["mean_abs_slope"],
            summary["mean_abs_curvature"],
            summary["left_half_mean_height"],
            summary["right_half_mean_height"],
        ],
        dtype=np.float32,
    )

    feature_vector = np.concatenate(
        [
            shape_profile * float(cfg.shape_weight),
            summary_vector * float(cfg.height_weight),
        ]
    ).astype(np.float32)

    return feature_vector, summary


def extract_image_feature(image_path: str, cfg: CFG) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    image_bgr = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Image could not be read.")

    mask = extract_blue_curve_mask(image_bgr)
    if int(mask.sum() / 255) < cfg.line_mask_min_pixels:
        raise ValueError("Detected line mask is too small.")

    profile = mask_to_absolute_profile(mask, cfg.n_points)
    profile = smooth_1d(profile, cfg.smooth_kernel)
    feature_vector, summary = build_feature_vector(profile, cfg)
    return feature_vector, profile, summary


def load_features(cfg: CFG) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[dict]]:
    image_paths = []
    for name in sorted(os.listdir(cfg.image_dir)):
        if name.lower().endswith(".png"):
            full_path = os.path.join(cfg.image_dir, name)
            if os.path.getsize(full_path) > 0:
                image_paths.append(full_path)

    if cfg.max_images > 0:
        image_paths = image_paths[: cfg.max_images]

    if not image_paths:
        raise FileNotFoundError(f"No valid PNG files found in: {cfg.image_dir}")

    records: List[dict] = []
    features: List[np.ndarray] = []
    profiles: List[np.ndarray] = []
    failures: List[dict] = []

    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        patient_id = os.path.splitext(image_name)[0]

        try:
            feature_vector, profile, summary = extract_image_feature(image_path, cfg)
            features.append(feature_vector)
            profiles.append(profile)
            records.append(
                {
                    "patient_id": patient_id,
                    "image_name": image_name,
                    "image_path": image_path,
                    **summary,
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
    P = np.vstack(profiles).astype(np.float32)
    return df, X, P, failures


def distance_to_center(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> np.ndarray:
    dists = np.zeros(len(X), dtype=np.float32)
    for i in range(len(X)):
        dists[i] = float(np.linalg.norm(X[i] - centers[labels[i]]))
    return dists


def reorder_clusters_by_height(
    df: pd.DataFrame,
    labels: np.ndarray,
    centers: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    temp_df = df.copy()
    temp_df["cluster_id"] = labels
    cluster_height = (
        temp_df.groupby("cluster_id")["mean_height"]
        .mean()
        .sort_values(ascending=False)
    )

    mapping = {old_id: new_id for new_id, old_id in enumerate(cluster_height.index.tolist())}
    new_labels = np.array([mapping[int(label)] for label in labels], dtype=np.int32)

    new_centers = np.zeros_like(centers)
    cluster_rows = []
    for old_id, new_id in mapping.items():
        new_centers[new_id] = centers[old_id]
        cluster_rows.append(
            {
                "original_cluster_id": int(old_id),
                "cluster_id": int(new_id),
                "cluster_name": cluster_name_from_rank(new_id, len(mapping)),
                "height_hint": height_hint_from_rank(new_id, len(mapping)),
                "mean_height_rank": int(new_id + 1),
                "cluster_mean_height": float(cluster_height.loc[old_id]),
            }
        )

    cluster_map_df = pd.DataFrame(cluster_rows).sort_values("cluster_id").reset_index(drop=True)
    return new_labels, new_centers, cluster_map_df


def cluster_name_from_rank(rank: int, n_clusters: int) -> str:
    if n_clusters == 2:
        return "high_position_group" if rank == 0 else "low_position_group"
    return f"height_rank_{rank + 1:02d}"


def height_hint_from_rank(rank: int, n_clusters: int) -> str:
    if n_clusters == 2:
        return "male_like" if rank == 0 else "female_like"
    if rank == 0:
        return "highest_group"
    if rank == n_clusters - 1:
        return "lowest_group"
    return "middle_group"


def summarize_clusters(df: pd.DataFrame) -> pd.DataFrame:
    summary_df = (
        df.groupby(["cluster_id", "cluster_name", "height_hint"])
        .agg(
            count=("patient_id", "count"),
            cluster_mean_height=("mean_height", "mean"),
            cluster_median_height=("mean_height", "median"),
            cluster_mean_peak=("peak_height", "mean"),
            cluster_mean_range=("height_range", "mean"),
            mean_distance=("distance_to_center", "mean"),
            median_distance=("distance_to_center", "median"),
        )
        .reset_index()
        .sort_values("cluster_id")
    )
    return summary_df


def export_cluster_examples(df: pd.DataFrame, cfg: CFG) -> None:
    example_root = os.path.join(cfg.out_dir, "cluster_examples")
    ensure_dir(example_root)

    for cluster_id, cluster_df in df.groupby("cluster_id"):
        cluster_dir = os.path.join(example_root, f"cluster_{cluster_id}_{cluster_df['cluster_name'].iloc[0]}")
        ensure_dir(cluster_dir)

        cluster_df = cluster_df.sort_values("distance_to_center").copy()
        cluster_df.to_csv(
            os.path.join(cluster_dir, "all_members.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        top_df = cluster_df.head(cfg.sample_per_cluster)
        top_df.to_csv(
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


def save_cluster_profile_plots(df: pd.DataFrame, profiles: np.ndarray, cfg: CFG) -> None:
    plot_dir = os.path.join(cfg.out_dir, "plots")
    ensure_dir(plot_dir)
    x_axis = np.linspace(0.0, 1.0, profiles.shape[1], dtype=np.float32)

    for cluster_id in sorted(df["cluster_id"].unique()):
        idx = np.where(df["cluster_id"].values == cluster_id)[0]
        cluster_profiles = profiles[idx]
        mean_profile = cluster_profiles.mean(axis=0)

        cluster_name = df.loc[df["cluster_id"] == cluster_id, "cluster_name"].iloc[0]
        height_hint = df.loc[df["cluster_id"] == cluster_id, "height_hint"].iloc[0]

        plt.figure(figsize=(11, 4.8))
        for curve in cluster_profiles[: min(len(cluster_profiles), 120)]:
            plt.plot(x_axis, curve, color="steelblue", alpha=0.08, linewidth=1)
        plt.plot(x_axis, mean_profile, color="crimson", linewidth=2.5, label="cluster mean")
        plt.title(f"Cluster {cluster_id} | {cluster_name} | {height_hint} (n={len(cluster_profiles)})")
        plt.xlabel("Normalized horizontal position")
        plt.ylabel("Normalized vertical height")
        plt.ylim(0.0, 1.0)
        plt.grid(alpha=0.2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(plot_dir, f"cluster_{cluster_id}_{cluster_name}_profiles.png"),
            dpi=200,
        )
        plt.close()


def save_cluster_comparison_plot(df: pd.DataFrame, profiles: np.ndarray, cfg: CFG) -> None:
    plot_dir = os.path.join(cfg.out_dir, "plots")
    ensure_dir(plot_dir)
    x_axis = np.linspace(0.0, 1.0, profiles.shape[1], dtype=np.float32)

    plt.figure(figsize=(11, 5))
    for cluster_id in sorted(df["cluster_id"].unique()):
        idx = np.where(df["cluster_id"].values == cluster_id)[0]
        mean_profile = profiles[idx].mean(axis=0)
        cluster_name = df.loc[df["cluster_id"] == cluster_id, "cluster_name"].iloc[0]
        plt.plot(x_axis, mean_profile, linewidth=2.5, label=f"Cluster {cluster_id}: {cluster_name}")

    plt.title("Cluster mean profiles")
    plt.xlabel("Normalized horizontal position")
    plt.ylabel("Normalized vertical height")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "cluster_mean_profiles.png"), dpi=200)
    plt.close()


def save_height_scatter_plot(df: pd.DataFrame, cfg: CFG) -> None:
    plot_dir = os.path.join(cfg.out_dir, "plots")
    ensure_dir(plot_dir)

    plt.figure(figsize=(8.5, 6))
    for cluster_id in sorted(df["cluster_id"].unique()):
        cluster_df = df[df["cluster_id"] == cluster_id]
        cluster_name = cluster_df["cluster_name"].iloc[0]
        plt.scatter(
            cluster_df["mean_height"],
            cluster_df["height_range"],
            s=24,
            alpha=0.7,
            label=f"Cluster {cluster_id}: {cluster_name}",
        )

    plt.title("Mean height vs. shape range")
    plt.xlabel("Mean vertical height")
    plt.ylabel("Height range")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "height_range_scatter.png"), dpi=200)
    plt.close()


def save_pca_plot(X_scaled: np.ndarray, df: pd.DataFrame, cfg: CFG) -> None:
    if len(X_scaled) < 2:
        return

    coords = pca_2d(X_scaled)
    plt.figure(figsize=(8.5, 6))
    for cluster_id in sorted(df["cluster_id"].unique()):
        idx = df["cluster_id"].values == cluster_id
        cluster_name = df.loc[df["cluster_id"] == cluster_id, "cluster_name"].iloc[0]
        plt.scatter(
            coords[idx, 0],
            coords[idx, 1],
            s=18,
            alpha=0.7,
            label=f"Cluster {cluster_id}: {cluster_name}",
        )

    plt.title("Height + shape clustering (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "pca_clusters.png"), dpi=200)
    plt.close()


def save_outputs(
    df: pd.DataFrame,
    X_scaled: np.ndarray,
    profiles: np.ndarray,
    labels: np.ndarray,
    model: KMeansResult,
    failures: List[dict],
    cfg: CFG,
) -> None:
    df = df.copy()
    df["cluster_id"] = labels
    df["distance_to_center"] = distance_to_center(X_scaled, labels, model.cluster_centers_)
    df["cluster_name"] = df["cluster_id"].map(
        lambda x: cluster_name_from_rank(int(x), cfg.n_clusters)
    )
    df["height_hint"] = df["cluster_id"].map(
        lambda x: height_hint_from_rank(int(x), cfg.n_clusters)
    )

    member_csv_path = os.path.join(cfg.out_dir, "cluster_members.csv")
    df.to_csv(member_csv_path, index=False, encoding="utf-8-sig")

    summary_df = summarize_clusters(df)
    summary_df.to_csv(
        os.path.join(cfg.out_dir, "cluster_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    if failures:
        pd.DataFrame(failures).to_csv(
            os.path.join(cfg.out_dir, "failed_images.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    save_cluster_profile_plots(df, profiles, cfg)
    save_cluster_comparison_plot(df, profiles, cfg)
    save_height_scatter_plot(df, cfg)
    save_pca_plot(X_scaled, df, cfg)
    export_cluster_examples(df, cfg)

    with open(os.path.join(cfg.out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("=== Height + Shape AEC Graph Clustering Summary ===\n\n")
        f.write(f"Image dir: {cfg.image_dir}\n")
        f.write(f"Output dir: {cfg.out_dir}\n")
        f.write(f"n_clusters: {cfg.n_clusters}\n")
        f.write(f"n_points: {cfg.n_points}\n")
        f.write(f"smooth_kernel: {cfg.smooth_kernel}\n")
        f.write(f"shape_weight: {cfg.shape_weight}\n")
        f.write(f"height_weight: {cfg.height_weight}\n\n")
        f.write("Interpretation note:\n")
        f.write("- higher mean_height cluster = graph drawn at a higher vertical position\n")
        f.write("- lower mean_height cluster = graph drawn at a lower vertical position\n")
        f.write("- male_like / female_like is only a height-based heuristic label\n\n")
        f.write(f"Analyzed images: {len(df)}\n")
        f.write(f"Failed images: {len(failures)}\n\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n")


def main() -> None:
    cfg = parse_args()
    ensure_dir(cfg.out_dir)

    df, X_raw, profiles, failures = load_features(cfg)
    X_scaled = standardize_features(X_raw)

    if len(df) < cfg.n_clusters:
        raise ValueError(
            f"Need at least {cfg.n_clusters} valid images, but only {len(df)} were available."
        )

    labels, model = fit_kmeans(
        X_scaled,
        n_clusters=cfg.n_clusters,
        random_state=cfg.random_state,
        n_init=20,
    )

    reordered_labels, reordered_centers, _ = reorder_clusters_by_height(
        df=df,
        labels=labels,
        centers=model.cluster_centers_,
    )
    model.cluster_centers_ = reordered_centers

    save_outputs(
        df=df,
        X_scaled=X_scaled,
        profiles=profiles,
        labels=reordered_labels,
        model=model,
        failures=failures,
        cfg=cfg,
    )

    print("Done")
    print(f"Analyzed images: {len(df)}")
    print(f"Failed images: {len(failures)}")
    print(f"Output dir: {cfg.out_dir}")


if __name__ == "__main__":
    main()
