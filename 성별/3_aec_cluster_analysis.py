from __future__ import annotations

import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score
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
CLUSTER_RANGE = range(2, 10)
SILHOUETTE_ANALYSIS_RANGE = [2, 3, 4, 5, 6]
MAX_DETAILED_SILHOUETTE_CLUSTERS = 12
TOP_CLUSTER_RANK_COUNT = 3
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
        / "aec_cluster_analysis"
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
    return clinical_df.merge(aec_df, on="PatientID", how="inner")


# Clustering helpers
def prepare_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    aecs = np.vstack(df["aec"].to_numpy())
    summary = df[SUMMARY_FEATURE_COLS].to_numpy(dtype=float)
    feature_matrix = np.hstack([aecs, summary])
    return StandardScaler().fit_transform(feature_matrix)


def compute_inertia(feature_matrix: np.ndarray, labels: np.ndarray) -> float:
    inertia = 0.0
    for cluster_id in np.unique(labels):
        cluster_points = feature_matrix[labels == cluster_id]
        centroid = cluster_points.mean(axis=0)
        inertia += float(np.sum((cluster_points - centroid) ** 2))
    return inertia


def select_cluster_count(feature_matrix: np.ndarray) -> tuple[int, pd.DataFrame]:
    records: list[dict[str, float]] = []

    for n_clusters in CLUSTER_RANGE:
        labels = AgglomerativeClustering(
            n_clusters=n_clusters, linkage="ward"
        ).fit_predict(feature_matrix)
        records.append(
            {
                "n_clusters": n_clusters,
                "silhouette": round(float(silhouette_score(feature_matrix, labels)), 3),
                "inertia": int(round(compute_inertia(feature_matrix, labels))),
            }
        )

    scores_df = pd.DataFrame(records)
    best_k = int(scores_df.loc[scores_df["silhouette"].idxmax(), "n_clusters"])
    return best_k, scores_df


def get_top_cluster_rankings(
    scores_df: pd.DataFrame, top_n: int = TOP_CLUSTER_RANK_COUNT
) -> pd.DataFrame:
    return (
        scores_df.sort_values(["silhouette", "n_clusters"], ascending=[False, True])
        .head(top_n)
        .reset_index(drop=True)
    )


def cluster_aecs(
    df: pd.DataFrame, feature_matrix: np.ndarray, n_clusters: int
) -> pd.DataFrame:
    labels = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(
        feature_matrix
    )
    result_df = df.copy()
    result_df["cluster_id"] = labels.astype(int)
    return result_df


# Output helpers
def save_silhouette_plot(
    scores_df: pd.DataFrame, output_dir: Path = OUTPUT_DIR
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    axes[0].plot(scores_df["n_clusters"], scores_df["silhouette"], marker="o")
    axes[0].set_ylabel("Silhouette")
    axes[0].set_title("Cluster Count Selection")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].plot(
        scores_df["n_clusters"], scores_df["inertia"], marker="o", color="#ff7f0e"
    )
    axes[1].set_xlabel("Number of Clusters")
    axes[1].set_ylabel("Inertia")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_dir / "silhouette_scores.png", dpi=200)
    plt.close(fig)


def save_detailed_silhouette_plots(
    df: pd.DataFrame,
    feature_matrix: np.ndarray,
    cluster_counts: list[int],
    output_dir: Path = OUTPUT_DIR,
) -> None:
    max_valid_clusters = len(feature_matrix) - 1
    valid_counts = sorted(
        {count for count in cluster_counts if 2 <= count <= max_valid_clusters}
    )
    if not valid_counts:
        return

    silhouette_dir = output_dir / "silhouette_analysis"
    silhouette_dir.mkdir(parents=True, exist_ok=True)
    projection_points = df[["aec_mean", "aec_range"]].to_numpy(dtype=float)

    for n_clusters in valid_counts:
        cluster_labels = AgglomerativeClustering(
            n_clusters=n_clusters, linkage="ward"
        ).fit_predict(feature_matrix)
        silhouette_avg = silhouette_score(feature_matrix, cluster_labels)
        sample_silhouette_values = silhouette_samples(feature_matrix, cluster_labels)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        ax1.set_xlim([-0.1, 1.0])
        ax1.set_ylim([0, len(feature_matrix) + (n_clusters + 1) * 10])

        y_lower = 10
        for cluster_id in range(n_clusters):
            cluster_silhouette_values = sample_silhouette_values[
                cluster_labels == cluster_id
            ]
            cluster_silhouette_values.sort()

            cluster_size = cluster_silhouette_values.shape[0]
            y_upper = y_lower + cluster_size
            color = cm.nipy_spectral(float(cluster_id) / n_clusters)

            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                cluster_silhouette_values,
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

        ax1.set_title("Silhouette Plot by Cluster")
        ax1.set_xlabel("Silhouette Coefficient")
        ax1.set_ylabel("Cluster Label")
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax1.set_yticks([])
        ax1.set_xticks([-0.1, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
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
                projection_points[cluster_labels == cluster_id].mean(axis=0)
                for cluster_id in range(n_clusters)
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
        for cluster_id, center in enumerate(cluster_centers):
            ax2.text(
                center[0],
                center[1],
                str(cluster_id),
                ha="center",
                va="center",
                fontsize=10 * FONT_SCALE,
                weight="bold",
            )

        ax2.set_title("Clustered AEC Data in Summary-Feature Space")
        ax2.set_xlabel("AEC Mean")
        ax2.set_ylabel("AEC Range")

        fig.suptitle(
            f"Silhouette Analysis for Agglomerative Clustering (n_clusters = {n_clusters}, avg = {silhouette_avg:.3f})",
            fontsize=14 * FONT_SCALE,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(silhouette_dir / f"silhouette_analysis_k{n_clusters}.png", dpi=200)
        plt.close(fig)


def save_cluster_mean_aecs(df: pd.DataFrame, output_dir: Path = OUTPUT_DIR) -> None:
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
    ax.set_title("Cluster Mean AECs")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "cluster_mean_aecs.png", dpi=200)
    plt.close(fig)


def save_sex_ratio_plot(df: pd.DataFrame, output_dir: Path = OUTPUT_DIR) -> None:
    ratio_df = (
        df.groupby(["cluster_id", "PatientSex"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["F", "M"], fill_value=0)
    )
    ratio_pct = ratio_df.div(ratio_df.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(ratio_pct))
    for sex_code, color in zip(["F", "M"], ["#ff9896", "#1f77b4"]):
        values = ratio_pct[sex_code].to_numpy()
        ax.bar(
            ratio_pct.index.astype(str),
            values,
            bottom=bottom,
            label=sex_code,
            color=color,
        )
        bottom += values

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Percentage")
    ax.set_title("Sex Ratio by Cluster")
    ax.legend(title="Sex")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(output_dir / "cluster_sex_ratio.png", dpi=200)
    plt.close(fig)


def save_boxplot(
    df: pd.DataFrame,
    column: str,
    title: str,
    output_name: str,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    cluster_ids = sorted(df["cluster_id"].unique())
    values = [
        df.loc[df["cluster_id"] == cluster_id, column].dropna().to_numpy()
        for cluster_id in cluster_ids
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(
        values,
        tick_labels=[str(cluster_id) for cluster_id in cluster_ids],
        patch_artist=True,
    )
    ax.set_xlabel("Cluster")
    ax.set_ylabel(column)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / output_name, dpi=200)
    plt.close(fig)


def remove_output_file(output_dir: Path, output_name: str) -> None:
    output_path = output_dir / output_name
    if output_path.exists():
        output_path.unlink()


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


def save_summary_tables(
    df: pd.DataFrame, scores_df: pd.DataFrame, output_dir: Path = OUTPUT_DIR
) -> None:
    summary_df = (
        df.groupby("cluster_id")
        .agg(
            member_count=("PatientID", "size"),
            female_count=("PatientSex", lambda s: int((s == "F").sum())),
            male_count=("PatientSex", lambda s: int((s == "M").sum())),
            bmi_mean=("BMI", "mean"),
            bmi_std=("BMI", "std"),
            bmi_median=("BMI", "median"),
            tama_mean=("TAMA", "mean"),
            tama_std=("TAMA", "std"),
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

    save_csv_with_fallback(summary_df, output_dir / "cluster_summary.csv")
    save_csv_with_fallback(scores_df, output_dir / "silhouette_scores.csv")


def save_report(
    df: pd.DataFrame,
    scores_df: pd.DataFrame,
    n_clusters: int,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    cluster_summary = (
        df.groupby("cluster_id")
        .agg(
            member_count=("PatientID", "size"),
            female_ratio_pct=("PatientSex", lambda s: float((s == "F").mean() * 100)),
            bmi_mean=("BMI", "mean"),
            bmi_median=("BMI", "median"),
            tama_mean=("TAMA", "mean"),
            tama_median=("TAMA", "median"),
            aec_mean=("aec_mean", "mean"),
            aec_range=("aec_range", "mean"),
            end_minus_start=("end_minus_start", "mean"),
        )
        .reset_index()
    )

    selected_row = scores_df.loc[scores_df["n_clusters"] == n_clusters].iloc[0]
    lines = [
        "# AEC Height/Shape Clustering",
        "",
        f"- Site: {SITE_NAME}",
        f"- Patients analyzed: {len(df)}",
        f"- Selected cluster count: {n_clusters}",
        f"- Silhouette score at selected cluster count: {selected_row['silhouette']:.4f}",
        f"- Inertia at selected cluster count: {selected_row['inertia']:.4f}",
        "- AEC extraction: blue line traced from PNG and resampled to 128 points.",
        "- Clustering input: extracted AEC values + height summary features.",
        "",
        "## Cluster Summary",
        "",
    ]

    for row in cluster_summary.itertuples(index=False):
        lines.extend(
            [
                f"### Cluster {row.cluster_id}",
                f"- Members: {row.member_count}",
                f"- Female ratio: {row.female_ratio_pct:.1f}%",
                f"- BMI mean/median: {row.bmi_mean:.2f} / {row.bmi_median:.2f}",
                f"- TAMA mean/median: {row.tama_mean:.2f} / {row.tama_median:.2f}",
                f"- Mean extracted height: {row.aec_mean:.2f}",
                f"- Mean AEC range: {row.aec_range:.2f}",
                f"- Mean end-minus-start: {row.end_minus_start:.2f}",
                "",
            ]
        )

    (output_dir / "analysis_report.md").write_text(
        "\n".join(lines), encoding="utf-8-sig"
    )


# Entry point
def run_single_sex_filter() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clinical_df = load_clinical_data()
    dataset_df = build_aec_dataset(clinical_df)
    feature_matrix = prepare_feature_matrix(dataset_df)
    n_clusters, scores_df = select_cluster_count(feature_matrix)
    save_silhouette_plot(scores_df, output_dir=OUTPUT_DIR)
    save_csv_with_fallback(scores_df, OUTPUT_DIR / "silhouette_scores.csv")
    top_rankings = get_top_cluster_rankings(scores_df)
    ranking_output = top_rankings.copy()
    ranking_output.insert(0, "rank", np.arange(1, len(ranking_output) + 1))
    save_csv_with_fallback(ranking_output, OUTPUT_DIR / "top_cluster_rankings.csv")

    for row in ranking_output.itertuples(index=False):
        rank_output_dir = (
            OUTPUT_DIR
            / f"rank_{row.rank}_k{row.n_clusters}_silhouette_{row.silhouette:.3f}"
        )
        rank_output_dir.mkdir(parents=True, exist_ok=True)
        clustered_df = cluster_aecs(dataset_df, feature_matrix, int(row.n_clusters))

        save_silhouette_plot(scores_df, output_dir=rank_output_dir)
        save_detailed_silhouette_plots(
            dataset_df,
            feature_matrix,
            [int(row.n_clusters)],
            output_dir=rank_output_dir,
        )
        save_cluster_mean_aecs(clustered_df, output_dir=rank_output_dir)
        remove_output_file(rank_output_dir, "cluster_sex_ratio.png")
        remove_output_file(rank_output_dir, "cluster_bmi_boxplot.png")
        remove_output_file(rank_output_dir, "cluster_tama_boxplot.png")
        remove_output_file(rank_output_dir, "cluster_members.csv")
        save_summary_tables(clustered_df, scores_df, output_dir=rank_output_dir)
        save_report(
            clustered_df, scores_df, int(row.n_clusters), output_dir=rank_output_dir
        )

    print(f"Clinical rows: {len(clinical_df)}")
    print(f"Merged rows: {len(dataset_df)}")
    print(f"Sex filter: {SEX_FILTER_DIR}")
    print(
        f"Top cluster counts: {', '.join(str(k) for k in ranking_output['n_clusters'].tolist())}"
    )
    print(f"Results saved to: {OUTPUT_DIR}")


def main() -> None:
    for sex_filter in resolve_requested_sex_filters():
        configure_sex_filter(sex_filter)
        run_single_sex_filter()


if __name__ == "__main__":
    main()
