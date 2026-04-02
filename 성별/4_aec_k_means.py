from __future__ import annotations

import os
from itertools import permutations
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FONT_SCALE = 1.3
SEX_FILTER_LABELS = {
    "F": "female",
    "M": "male",
}

plt.rcParams.update(
    {
        "font.size": 10 * FONT_SCALE,
        "axes.titlesize": 12 * FONT_SCALE,
        "axes.labelsize": 11 * FONT_SCALE,
        "xtick.labelsize": 10 * FONT_SCALE,
        "ytick.labelsize": 10 * FONT_SCALE,
        "legend.fontsize": 10 * FONT_SCALE,
        "legend.title_fontsize": 10 * FONT_SCALE,
        "figure.titlesize": 14 * FONT_SCALE,
    }
)


SITE_NAME = os.environ.get("AEC_SITE_NAME", "강남")
REQUESTED_SEX_FILTER = os.environ.get("AEC_SEX_FILTER", "").strip().upper()
SEX_FILTER = ""
SEX_FILTER_DIR = ""
BASE_DIR = Path(__file__).resolve().parents[2]
SITE_DIR = BASE_DIR / "data" / "AEC" / SITE_NAME
EXCEL_PATH = SITE_DIR / "Result_Filter.xlsx"
IMAGE_DIR = SITE_DIR / "Image_Filter"
OUTPUT_DIR = Path()

AEC_POINTS = 128
N_CLUSTERS = 4
TEST_SIZE = 0.2
RANDOM_STATE = 42
BMI_THRESHOLD = 25.0
CLINICAL_USECOLS = ["PatientID", "PatientSex", "SRC_Report", "BMI"]
SUMMARY_FEATURE_COLS = [
    "aec_mean",
    "aec_std",
    "aec_min",
    "aec_max",
    "aec_range",
    "end_minus_start",
    "mean_abs_slope",
]
MEMBER_COLS = [
    "PatientID",
    "PatientSex",
    "BMI",
    "TAMA",
    "cluster_id",
    "image_path",
    "aec_mean",
    "aec_std",
    "aec_min",
    "aec_max",
    "aec_range",
    "end_minus_start",
    "mean_abs_slope",
    "auc",
    "cv",
    "asymmetry",
    "peak_position",
    "valley_position",
    "sign_changes",
    "high_mA_mean",
    "chest_slope",
    "abdomen_slope",
    "pelvis_slope",
]
TARGET_GROUP_LABELS = {
    0: "tama_lt_mean_bmi_lt_25",
    1: "tama_lt_mean_bmi_ge_25",
    2: "tama_ge_mean_bmi_lt_25",
    3: "tama_ge_mean_bmi_ge_25",
}


def resolve_requested_sex_filters() -> list[str]:
    if not REQUESTED_SEX_FILTER:
        return list(SEX_FILTER_LABELS)
    if REQUESTED_SEX_FILTER not in SEX_FILTER_LABELS:
        valid = ", ".join(SEX_FILTER_LABELS)
        raise ValueError(
            f"Unsupported AEC_SEX_FILTER '{REQUESTED_SEX_FILTER}'. Expected one of: {valid}"
        )
    return [REQUESTED_SEX_FILTER]


def configure_sex_filter(sex_filter: str) -> None:
    global SEX_FILTER, SEX_FILTER_DIR, OUTPUT_DIR

    SEX_FILTER = sex_filter
    SEX_FILTER_DIR = SEX_FILTER_LABELS[sex_filter]
    OUTPUT_DIR = (
        BASE_DIR
        / "result"
        / "성별"
        / SITE_NAME
        / SEX_FILTER_DIR
        / "4_aec_k_means_train_test"
    )


# Data loading and AEC feature helpers
def apply_sex_filter(df: pd.DataFrame) -> pd.DataFrame:
    filtered_df = df[df["PatientSex"] == SEX_FILTER].copy()
    if filtered_df.empty:
        raise ValueError(
            f"No rows found for site '{SITE_NAME}' with sex filter '{SEX_FILTER_DIR}'."
        )
    return filtered_df


def load_clinical_data() -> pd.DataFrame:
    df = pd.read_excel(EXCEL_PATH, usecols=CLINICAL_USECOLS).copy()
    df = df.rename(columns={"SRC_Report": "TAMA"})
    df["PatientID"] = df["PatientID"].astype(str)
    df["PatientSex"] = (
        df["PatientSex"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({"MALE": "M", "FEMALE": "F"})
    )
    df["TAMA"] = pd.to_numeric(df["TAMA"], errors="coerce")
    df["BMI"] = pd.to_numeric(df["BMI"], errors="coerce")
    df = df.dropna(subset=["PatientID", "PatientSex", "TAMA", "BMI"]).copy()
    return apply_sex_filter(df)


def extract_aec(image_path: Path, n_points: int = AEC_POINTS) -> np.ndarray:
    image_bgr = cv2.imdecode(
        np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if image_bgr is None:
        raise ValueError(f"Image could not be read: {image_path}")

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 80, 80], dtype=np.uint8)
    upper_blue = np.array([130, 255, 255], dtype=np.uint8)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((3, 3), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

    y_idx, x_idx = np.where(blue_mask > 0)
    if len(x_idx) == 0:
        raise ValueError(f"Blue line not found: {image_path}")

    unique_x = np.unique(x_idx)
    mean_y = np.array([y_idx[x_idx == x].mean() for x in unique_x], dtype=float)
    target_x = np.linspace(unique_x.min(), unique_x.max(), n_points)
    interpolated_y = np.interp(target_x, unique_x, mean_y)
    return image_bgr.shape[0] - interpolated_y


def compute_aec_features(aec: np.ndarray) -> dict[str, float]:
    n_points = len(aec)
    slope = np.diff(aec)
    first_third = n_points // 3
    second_third = 2 * n_points // 3
    left_half = aec[: n_points // 2]
    right_half = aec[n_points // 2 :]

    return {
        "aec_mean": float(aec.mean()),
        "aec_std": float(aec.std()),
        "aec_min": float(aec.min()),
        "aec_max": float(aec.max()),
        "aec_range": float(aec.max() - aec.min()),
        "start_height": float(aec[0]),
        "mid_height": float(aec[n_points // 2]),
        "end_height": float(aec[-1]),
        "left_mean": float(aec[:first_third].mean()),
        "center_mean": float(aec[first_third:second_third].mean()),
        "right_mean": float(aec[second_third:].mean()),
        "end_minus_start": float(aec[-1] - aec[0]),
        "mean_abs_slope": float(np.abs(slope).mean()) if len(slope) > 0 else 0.0,
        "auc": float(np.trapezoid(aec)),
        "peak_position": float(np.argmax(aec) / n_points),
        "valley_position": float(np.argmin(aec) / n_points),
        "cv": float(aec.std() / aec.mean()) if aec.mean() != 0 else 0.0,
        "asymmetry": float((left_half.mean() - right_half.mean()) / aec.mean())
        if aec.mean() != 0
        else 0.0,
        "sign_changes": int(np.sum(np.diff(np.sign(slope)) != 0))
        if len(slope) > 1
        else 0,
        "high_mA_mean": float(aec[aec >= np.percentile(aec, 75)].mean()),
        "chest_slope": float(slope[: n_points // 4].mean())
        if len(slope[: n_points // 4]) > 0
        else 0.0,
        "abdomen_slope": float(slope[n_points // 4 : n_points // 2].mean())
        if len(slope[n_points // 4 : n_points // 2]) > 0
        else 0.0,
        "pelvis_slope": float(slope[n_points // 2 :].mean())
        if len(slope[n_points // 2 :]) > 0
        else 0.0,
    }


def build_aec_dataset(clinical_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    patient_ids = set(clinical_df["PatientID"])

    for image_path in sorted(IMAGE_DIR.glob("*.png")):
        patient_id = image_path.stem
        if patient_id not in patient_ids:
            continue

        try:
            aec = extract_aec(image_path)
        except Exception as exc:
            failures.append({"PatientID": patient_id, "reason": str(exc)})
            continue

        rows.append(
            {
                "PatientID": patient_id,
                "image_path": str(image_path),
                "aec": aec,
                **compute_aec_features(aec),
            }
        )

    if failures:
        pd.DataFrame(failures).to_csv(
            OUTPUT_DIR / "extraction_failures.csv", index=False, encoding="utf-8-sig"
        )
        print(f"[Warning] {len(failures)} images failed extraction.")

    aec_df = pd.DataFrame(rows)
    return clinical_df.merge(aec_df, on="PatientID", how="inner").reset_index(drop=True)


# Clustering helpers
def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    aecs = np.vstack(df["aec"].to_numpy())
    summary = df[SUMMARY_FEATURE_COLS].to_numpy(dtype=float)
    return np.hstack([aecs, summary])


def split_dataset(
    df: pd.DataFrame,
    feature_matrix: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    train_features = feature_matrix[train_idx]
    test_features = feature_matrix[test_idx]
    return train_df, test_df, train_features, test_features


def safe_silhouette_score(feature_matrix: np.ndarray, labels: np.ndarray) -> float:
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(unique_labels) >= len(feature_matrix):
        return float("nan")
    return float(silhouette_score(feature_matrix, labels))


def safe_silhouette_samples(
    feature_matrix: np.ndarray, labels: np.ndarray
) -> np.ndarray | None:
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(unique_labels) >= len(feature_matrix):
        return None
    return silhouette_samples(feature_matrix, labels)


def compute_squared_distances(
    feature_matrix: np.ndarray, centers: np.ndarray
) -> np.ndarray:
    return np.sum((feature_matrix[:, None, :] - centers[None, :, :]) ** 2, axis=2)


def compute_assignment_inertia(
    feature_matrix: np.ndarray, centers: np.ndarray
) -> float:
    min_sq_distance = np.min(compute_squared_distances(feature_matrix, centers), axis=1)
    return float(min_sq_distance.sum())


def initialize_kmeans_plus_plus(
    feature_matrix: np.ndarray, n_clusters: int, rng: np.random.Generator
) -> np.ndarray:
    centers = np.empty((n_clusters, feature_matrix.shape[1]), dtype=float)
    first_idx = int(rng.integers(len(feature_matrix)))
    centers[0] = feature_matrix[first_idx]

    closest_sq_dist = np.sum((feature_matrix - centers[0]) ** 2, axis=1)
    for center_idx in range(1, n_clusters):
        total_dist = float(closest_sq_dist.sum())
        if total_dist == 0.0:
            selected_idx = int(rng.integers(len(feature_matrix)))
        else:
            prob = closest_sq_dist / total_dist
            selected_idx = int(rng.choice(len(feature_matrix), p=prob))
        centers[center_idx] = feature_matrix[selected_idx]
        new_sq_dist = np.sum((feature_matrix - centers[center_idx]) ** 2, axis=1)
        closest_sq_dist = np.minimum(closest_sq_dist, new_sq_dist)

    return centers


def fit_kmeans_numpy(
    feature_matrix: np.ndarray,
    n_clusters: int,
    random_state: int,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(random_state)
    centers = initialize_kmeans_plus_plus(feature_matrix, n_clusters, rng)

    for _ in range(max_iter):
        distances = compute_squared_distances(feature_matrix, centers)
        labels = np.argmin(distances, axis=1)

        new_centers = centers.copy()
        for cluster_id in range(n_clusters):
            cluster_points = feature_matrix[labels == cluster_id]
            if len(cluster_points) == 0:
                new_centers[cluster_id] = feature_matrix[
                    int(rng.integers(len(feature_matrix)))
                ]
            else:
                new_centers[cluster_id] = cluster_points.mean(axis=0)

        shift = float(np.linalg.norm(new_centers - centers))
        centers = new_centers
        if shift <= tol:
            break

    final_distances = compute_squared_distances(feature_matrix, centers)
    final_labels = np.argmin(final_distances, axis=1)
    final_inertia = float(np.min(final_distances, axis=1).sum())
    return centers, final_labels.astype(int), final_inertia


def assign_reference_groups(df: pd.DataFrame, tama_threshold: float) -> pd.DataFrame:
    result_df = df.copy()
    result_df["tama_group"] = np.where(
        result_df["TAMA"] < tama_threshold, "lt_mean", "ge_mean"
    )
    result_df["bmi_group"] = np.where(
        result_df["BMI"] < BMI_THRESHOLD, "lt_25", "ge_25"
    )
    result_df["y_true"] = (
        (result_df["TAMA"] >= tama_threshold).astype(int) * 2
        + (result_df["BMI"] >= BMI_THRESHOLD).astype(int)
    )
    result_df["y_true_label"] = result_df["y_true"].map(TARGET_GROUP_LABELS)
    return result_df


def find_best_cluster_group_mapping(train_df: pd.DataFrame) -> dict[int, int]:
    cluster_ids = [int(cluster_id) for cluster_id in sorted(train_df["cluster_id"].unique())]
    group_ids = sorted(TARGET_GROUP_LABELS)
    contingency = pd.crosstab(train_df["cluster_id"], train_df["y_true"])

    best_score = -1
    best_mapping: dict[int, int] = {}
    for assigned_group_ids in permutations(group_ids, len(cluster_ids)):
        score = 0
        for cluster_id, group_id in zip(cluster_ids, assigned_group_ids):
            score += int(contingency.get(group_id, pd.Series()).get(cluster_id, 0))
        if score > best_score:
            best_score = score
            best_mapping = {
                int(cluster_id): int(group_id)
                for cluster_id, group_id in zip(cluster_ids, assigned_group_ids)
            }

    return best_mapping


def apply_cluster_group_mapping(
    df: pd.DataFrame, cluster_to_group: dict[int, int]
) -> pd.DataFrame:
    result_df = df.copy()
    result_df["y_pred"] = result_df["cluster_id"].map(cluster_to_group).astype("Int64")
    result_df["y_pred_label"] = result_df["y_pred"].map(TARGET_GROUP_LABELS)
    result_df["y_match"] = result_df["y_true"] == result_df["y_pred"]
    return result_df


def cluster_summary_frame(df: pd.DataFrame) -> pd.DataFrame:
    summary_df = (
        df.groupby("cluster_id")
        .agg(
            member_count=("PatientID", "size"),
            female_count=("PatientSex", lambda s: int((s == "F").sum())),
            male_count=("PatientSex", lambda s: int((s == "M").sum())),
            bmi_mean=("BMI", "mean"),
            bmi_median=("BMI", "median"),
            tama_mean=("TAMA", "mean"),
            tama_median=("TAMA", "median"),
            aec_mean=("aec_mean", "mean"),
            aec_range=("aec_range", "mean"),
            end_minus_start=("end_minus_start", "mean"),
            mean_abs_slope=("mean_abs_slope", "mean"),
        )
        .reset_index()
    )
    summary_df["female_ratio_pct"] = (
        summary_df["female_count"] / summary_df["member_count"] * 100
    )
    summary_df["male_ratio_pct"] = (
        summary_df["male_count"] / summary_df["member_count"] * 100
    )
    return summary_df


# Output helpers
def save_detailed_silhouette_plot(
    df: pd.DataFrame,
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    split_name: str,
    output_path: Path,
) -> None:
    sample_silhouette_values = safe_silhouette_samples(feature_matrix, labels)
    silhouette_avg = safe_silhouette_score(feature_matrix, labels)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    projection_points = df[["aec_mean", "aec_range"]].to_numpy(dtype=float)
    unique_clusters = sorted(np.unique(labels))

    if sample_silhouette_values is None:
        ax1.text(
            0.5,
            0.5,
            "Silhouette score is not available\nfor this split.",
            ha="center",
            va="center",
            fontsize=12 * FONT_SCALE,
        )
        ax1.set_axis_off()
    else:
        ax1.set_xlim([-0.1, 1.0])
        ax1.set_ylim([0, len(feature_matrix) + (len(unique_clusters) + 1) * 10])

        y_lower = 10
        for cluster_id in unique_clusters:
            cluster_values = sample_silhouette_values[labels == cluster_id]
            cluster_values.sort()

            cluster_size = cluster_values.shape[0]
            y_upper = y_lower + cluster_size
            color = cm.nipy_spectral(float(cluster_id) / max(N_CLUSTERS, 1))

            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                cluster_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )
            ax1.text(
                -0.06,
                y_lower + 0.5 * cluster_size,
                str(cluster_id),
                fontsize=10 * FONT_SCALE,
            )
            y_lower = y_upper + 10

        ax1.set_title(f"{split_name} Silhouette Plot")
        ax1.set_xlabel("Silhouette Coefficient")
        ax1.set_ylabel("Cluster Label")
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax1.set_yticks([])
        ax1.set_xticks([-0.1, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    colors = cm.nipy_spectral(labels.astype(float) / max(N_CLUSTERS, 1))
    ax2.scatter(
        projection_points[:, 0],
        projection_points[:, 1],
        marker=".",
        s=30,
        lw=0,
        alpha=0.7,
        c=colors,
        edgecolor="k",
    )

    cluster_centers = np.vstack(
        [
            projection_points[labels == cluster_id].mean(axis=0)
            for cluster_id in unique_clusters
        ]
    )
    ax2.scatter(
        cluster_centers[:, 0],
        cluster_centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )
    for cluster_id, center in zip(unique_clusters, cluster_centers):
        ax2.text(
            center[0],
            center[1],
            str(cluster_id),
            ha="center",
            va="center",
            fontsize=10 * FONT_SCALE,
            weight="bold",
        )

    ax2.set_title(f"{split_name} Clusters in Summary-Feature Space")
    ax2.set_xlabel("AEC Mean")
    ax2.set_ylabel("AEC Range")

    silhouette_text = f"{silhouette_avg:.3f}" if not np.isnan(silhouette_avg) else "N/A"
    fig.suptitle(
        f"{split_name} Silhouette Analysis for KMeans (k = {N_CLUSTERS}, avg = {silhouette_text})",
        fontsize=14 * FONT_SCALE,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_cluster_mean_aecs(
    df: pd.DataFrame, split_name: str, output_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for cluster_id in sorted(df["cluster_id"].unique()):
        cluster_aecs = np.vstack(
            df.loc[df["cluster_id"] == cluster_id, "aec"].to_numpy()
        )
        mean_aec = cluster_aecs.mean(axis=0)
        std_aec = cluster_aecs.std(axis=0)
        x_axis = np.linspace(0, 1, len(mean_aec))
        ax.plot(x_axis, mean_aec, label=f"Cluster {cluster_id} (n={len(cluster_aecs)})")
        ax.fill_between(x_axis, mean_aec - std_aec, mean_aec + std_aec, alpha=0.15)

    ax.set_xlabel("Instance Number (Head → Leg)")
    ax.set_ylabel("X-ray Tube Current")
    ax.set_title(f"{split_name} Cluster Mean AECs")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_csv_with_fallback(df: pd.DataFrame, output_path: Path) -> Path:
    try:
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        return output_path
    except PermissionError:
        fallback_path = output_path.with_name(f"{output_path.stem}{output_path.suffix}")
        df.to_csv(fallback_path, index=False, encoding="utf-8-sig")
        print(
            f"Warning: {output_path.name} is locked. Saved fallback file to {fallback_path.name}"
        )
        return fallback_path


def save_split_outputs(
    df: pd.DataFrame,
    feature_matrix: np.ndarray,
    split_name: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_prefix = split_name.lower()

    save_detailed_silhouette_plot(
        df,
        feature_matrix,
        df["cluster_id"].to_numpy(dtype=int),
        split_name=split_name,
        output_path=output_dir
        / f"{split_prefix}_silhouette_analysis_k{N_CLUSTERS}.png",
    )
    save_cluster_mean_aecs(
        df, split_name, output_dir / f"{split_prefix}_cluster_mean_aecs.png"
    )

    save_csv_with_fallback(
        df[MEMBER_COLS].sort_values(["cluster_id", "PatientID"]),
        output_dir / f"{split_prefix}_cluster_members.csv",
    )
    save_csv_with_fallback(
        cluster_summary_frame(df), output_dir / f"{split_prefix}_cluster_summary.csv"
    )


def save_split_assignment_table(
    train_df: pd.DataFrame, test_df: pd.DataFrame, output_path: Path
) -> None:
    split_df = pd.concat(
        [
            train_df.assign(split="train")[
                [
                    "PatientID",
                    "cluster_id",
                    "y_true",
                    "y_true_label",
                    "y_pred",
                    "y_pred_label",
                    "y_match",
                    "BMI",
                    "TAMA",
                    "image_path",
                    "split",
                ]
            ],
            test_df.assign(split="test")[
                [
                    "PatientID",
                    "cluster_id",
                    "y_true",
                    "y_true_label",
                    "y_pred",
                    "y_pred_label",
                    "y_match",
                    "BMI",
                    "TAMA",
                    "image_path",
                    "split",
                ]
            ],
        ],
        ignore_index=True,
    ).sort_values(["split", "cluster_id", "PatientID"])
    save_csv_with_fallback(split_df, output_path)


def save_target_group_evaluation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cluster_to_group: dict[int, int],
    tama_threshold: float,
    output_dir: Path,
) -> tuple[dict[str, float], dict[str, float]]:
    mapping_rows = []
    for cluster_id in sorted(cluster_to_group):
        group_id = cluster_to_group[cluster_id]
        cluster_mask = train_df["cluster_id"] == cluster_id
        mapping_rows.append(
            {
                "cluster_id": cluster_id,
                "mapped_y": group_id,
                "mapped_y_label": TARGET_GROUP_LABELS[group_id],
                "train_cluster_size": int(cluster_mask.sum()),
                "train_match_count": int(
                    train_df.loc[cluster_mask, "y_true"].eq(group_id).sum()
                ),
            }
        )

    save_csv_with_fallback(
        pd.DataFrame(mapping_rows),
        output_dir / "cluster_to_y_mapping.csv",
    )

    train_crosstab = pd.crosstab(
        train_df["y_true_label"],
        train_df["y_pred_label"],
        dropna=False,
    )
    train_crosstab.index.name = "y_true_label"
    save_csv_with_fallback(
        train_crosstab.reset_index(),
        output_dir / "train_y_confusion_matrix.csv",
    )

    test_crosstab = pd.crosstab(
        test_df["y_true_label"],
        test_df["y_pred_label"],
        dropna=False,
    )
    test_crosstab.index.name = "y_true_label"
    save_csv_with_fallback(
        test_crosstab.reset_index(),
        output_dir / "test_y_confusion_matrix.csv",
    )

    threshold_df = pd.DataFrame(
        [
            {
                "tama_threshold_mean": tama_threshold,
                "bmi_threshold": BMI_THRESHOLD,
            }
        ]
    )
    save_csv_with_fallback(threshold_df, output_dir / "y_thresholds.csv")

    train_group_metrics = {
        "tama_threshold": tama_threshold,
        "bmi_threshold": BMI_THRESHOLD,
        "y_accuracy": float(train_df["y_match"].mean()),
    }
    test_group_metrics = {
        "tama_threshold": tama_threshold,
        "bmi_threshold": BMI_THRESHOLD,
        "y_accuracy": float(test_df["y_match"].mean()),
    }
    return train_group_metrics, test_group_metrics


def save_metrics_report(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_metrics: dict[str, float],
    test_metrics: dict[str, float],
    train_group_metrics: dict[str, float],
    test_group_metrics: dict[str, float],
    output_dir: Path,
) -> None:
    metrics_df = pd.DataFrame([train_metrics, test_metrics])
    save_csv_with_fallback(metrics_df, output_dir / "train_test_metrics.csv")

    lines = [
        "# AEC KMeans Train/Test Evaluation",
        "",
        f"- Site: {SITE_NAME}",
        f"- Cluster count (k): {N_CLUSTERS}",
        f"- Random state: {RANDOM_STATE}",
        f"- Train/Test split: {1 - TEST_SIZE:.0%} / {TEST_SIZE:.0%}",
        f"- Total patients analyzed: {len(train_df) + len(test_df)}",
        "",
        "## Metrics",
        "",
        f"- Train silhouette score: {train_metrics['silhouette']:.4f}"
        if not np.isnan(train_metrics["silhouette"])
        else "- Train silhouette score: N/A",
        f"- Test silhouette score: {test_metrics['silhouette']:.4f}"
        if not np.isnan(test_metrics["silhouette"])
        else "- Test silhouette score: N/A",
        f"- Train inertia: {train_metrics['inertia']:.4f}",
        f"- Test assignment inertia: {test_metrics['inertia']:.4f}",
        f"- TAMA mean threshold for y: {train_group_metrics['tama_threshold']:.4f}",
        f"- BMI threshold for y: {train_group_metrics['bmi_threshold']:.1f}",
        f"- Train y-match accuracy: {train_group_metrics['y_accuracy']:.4f}",
        f"- Test y-match accuracy: {test_group_metrics['y_accuracy']:.4f}",
        "",
        "## Cluster Counts",
        "",
        f"- Train counts: {train_df['cluster_id'].value_counts().sort_index().to_dict()}",
        f"- Test counts: {test_df['cluster_id'].value_counts().sort_index().to_dict()}",
        "",
        "## Y Counts",
        "",
        f"- Train y counts: {train_df['y_true_label'].value_counts().sort_index().to_dict()}",
        f"- Test y counts: {test_df['y_true_label'].value_counts().sort_index().to_dict()}",
    ]
    (output_dir / "analysis_report.md").write_text(
        "\n".join(lines), encoding="utf-8-sig"
    )


# Entry point
def run_single_sex_filter() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    clinical_df = load_clinical_data()
    dataset_df = build_aec_dataset(clinical_df)
    raw_feature_matrix = build_feature_matrix(dataset_df)
    train_df, test_df, train_raw, test_raw = split_dataset(
        dataset_df, raw_feature_matrix
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_raw)
    x_test = scaler.transform(test_raw)

    centers, train_labels, train_inertia = fit_kmeans_numpy(
        x_train,
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
    )
    test_labels = np.argmin(compute_squared_distances(x_test, centers), axis=1)

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["cluster_id"] = train_labels.astype(int)
    test_df["cluster_id"] = test_labels.astype(int)

    tama_threshold = float(train_df["TAMA"].mean())
    train_df = assign_reference_groups(train_df, tama_threshold)
    test_df = assign_reference_groups(test_df, tama_threshold)
    cluster_to_group = find_best_cluster_group_mapping(train_df)
    train_df = apply_cluster_group_mapping(train_df, cluster_to_group)
    test_df = apply_cluster_group_mapping(test_df, cluster_to_group)

    train_metrics = {
        "split": "train",
        "n_samples": float(len(train_df)),
        "silhouette": safe_silhouette_score(x_train, train_labels),
        "inertia": train_inertia,
    }
    test_metrics = {
        "split": "test",
        "n_samples": float(len(test_df)),
        "silhouette": safe_silhouette_score(x_test, test_labels),
        "inertia": compute_assignment_inertia(x_test, centers),
    }
    train_group_metrics, test_group_metrics = save_target_group_evaluation(
        train_df,
        test_df,
        cluster_to_group,
        tama_threshold,
        OUTPUT_DIR,
    )

    split_info = pd.concat(
        [
            train_df.assign(split="train")[
                [
                    "PatientID",
                    "PatientSex",
                    "BMI",
                    "TAMA",
                    "y_true",
                    "y_true_label",
                    "split",
                ]
            ],
            test_df.assign(split="test")[
                [
                    "PatientID",
                    "PatientSex",
                    "BMI",
                    "TAMA",
                    "y_true",
                    "y_true_label",
                    "split",
                ]
            ],
        ],
        ignore_index=True,
    ).sort_values(["split", "PatientID"])
    save_csv_with_fallback(split_info, OUTPUT_DIR / "dataset_split.csv")

    save_split_outputs(
        train_df, x_train, split_name="Train", output_dir=OUTPUT_DIR / "train"
    )
    save_split_outputs(
        test_df, x_test, split_name="Test", output_dir=OUTPUT_DIR / "test"
    )
    save_split_assignment_table(
        train_df, test_df, OUTPUT_DIR / "train_test_cluster_assignments.csv"
    )
    save_metrics_report(
        train_df,
        test_df,
        train_metrics,
        test_metrics,
        train_group_metrics,
        test_group_metrics,
        OUTPUT_DIR,
    )

    print(f"Clinical rows: {len(clinical_df)}")
    print(f"Merged rows: {len(dataset_df)}")
    print(f"Sex filter: {SEX_FILTER_DIR}")
    print(f"Train/Test rows: {len(train_df)} / {len(test_df)}")
    print(
        f"Train silhouette: {train_metrics['silhouette']:.4f}"
        if not np.isnan(train_metrics["silhouette"])
        else "Train silhouette: N/A"
    )
    print(
        f"Test silhouette: {test_metrics['silhouette']:.4f}"
        if not np.isnan(test_metrics["silhouette"])
        else "Test silhouette: N/A"
    )
    print(f"TAMA mean threshold for y: {tama_threshold:.4f}")
    print(f"Test y-match accuracy: {test_group_metrics['y_accuracy']:.4f}")
    print(f"Results saved to: {OUTPUT_DIR}")


def main() -> None:
    for sex_filter in resolve_requested_sex_filters():
        configure_sex_filter(sex_filter)
        run_single_sex_filter()


if __name__ == "__main__":
    main()
