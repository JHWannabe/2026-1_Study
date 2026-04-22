from __future__ import annotations

import os
from itertools import combinations
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

FONT_SCALE = 1.3

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

SITE_NAME = os.environ.get("AEC_SITE_NAME", "신촌")
BASE_DIR = Path(__file__).resolve().parents[2]
SITE_DIR = BASE_DIR / "data" / "AEC" / SITE_NAME
EXCEL_PATH = SITE_DIR / "Result_Filter.xlsx"
IMAGE_DIR = SITE_DIR / "Image_Filter"
OUTPUT_DIR = BASE_DIR / "result/0327" / SITE_NAME / "aec_quadrant_shape_analysis"

AEC_POINTS = 128
BMI_CUTOFF = 25.0
SHOW_PLOTS = True
CLINICAL_USECOLS = ["PatientID", "PatientSex", "SRC_Report", "BMI"]

GROUP_ORDER = [
    "BMI_▼_TAMA_▼",
    "BMI_▼_TAMA_△",
    "BMI_△_TAMA_▼",
    "BMI_△_TAMA_△",
]
GROUP_COLORS = {
    "BMI_▼_TAMA_▼": "#1f77b4",
    "BMI_▼_TAMA_△": "#2ca02c",
    "BMI_△_TAMA_▼": "#ff7f0e",
    "BMI_△_TAMA_△": "#d62728",
}

ALL_FEATURE_COLS = [
    "aec_mean",
    "aec_std",
    "aec_range",
    "left_mean",
    "center_mean",
    "right_mean",
    "end_minus_start",
    "mean_abs_slope",
    "auc",
    "peak_position",
    "cv",
    "asymmetry",
    "sign_changes",
    "high_mA_mean",
    "chest_slope",
    "abdomen_slope",
    "pelvis_slope",
]
GROUP_MEMBER_COLS = [
    "PatientID",
    "PatientSex",
    "BMI",
    "TAMA",
    "group",
    "image_path",
    "aec_mean",
    "aec_std",
    "aec_min",
    "aec_max",
    "aec_range",
    "left_mean",
    "center_mean",
    "right_mean",
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


# Data loading and AEC feature helpers
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
    return df.dropna(subset=["PatientID", "PatientSex", "TAMA", "BMI"]).copy()


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

    base = {
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
    }

    left_half = aec[: n_points // 2]
    right_half = aec[n_points // 2 :]
    extra = {
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
    return {**base, **extra}


def build_dataset(clinical_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    patient_ids = set(clinical_df["PatientID"])

    for image_path in sorted(IMAGE_DIR.glob("*.png")):
        patient_id = image_path.stem
        if patient_id not in patient_ids:
            continue

        aec = extract_aec(image_path)
        rows.append(
            {
                "PatientID": patient_id,
                "image_path": str(image_path),
                "aec": aec,
                **compute_aec_features(aec),
            }
        )

    aec_df = pd.DataFrame(rows)
    return clinical_df.merge(aec_df, on="PatientID", how="inner")


def assign_quadrant_groups(df: pd.DataFrame, tama_cutoff: float) -> pd.DataFrame:
    result_df = df.copy()
    bmi_group = np.where(result_df["BMI"] >= BMI_CUTOFF, "BMI_△", "BMI_▼")
    tama_group = np.where(result_df["TAMA"] >= tama_cutoff, "TAMA_△", "TAMA_▼")
    result_df["group"] = bmi_group + "_" + tama_group
    return result_df


def finalize_figure(fig: plt.Figure, output_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    if SHOW_PLOTS and "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close(fig)


# Statistical analysis helpers
def run_statistical_tests(df: pd.DataFrame, output_dir: Path) -> None:
    kruskal_results: list[dict[str, object]] = []
    posthoc_results: list[dict[str, object]] = []
    group_pairs = list(combinations(GROUP_ORDER, 2))

    for feature in ALL_FEATURE_COLS:
        groups_data = [
            df.loc[df["group"] == group, feature].dropna().to_numpy()
            for group in GROUP_ORDER
        ]
        groups_data = [group for group in groups_data if len(group) > 0]
        if len(groups_data) < 2:
            continue

        stat, p_value = stats.kruskal(*groups_data)
        kruskal_results.append(
            {
                "feature": feature,
                "H_stat": round(float(stat), 4),
                "p_value": round(float(p_value), 4),
                "significant": p_value < 0.05,
            }
        )

        if p_value < 0.05:
            n_pairs = len(group_pairs)
            for group_1, group_2 in group_pairs:
                values_1 = df.loc[df["group"] == group_1, feature].dropna().to_numpy()
                values_2 = df.loc[df["group"] == group_2, feature].dropna().to_numpy()
                if len(values_1) == 0 or len(values_2) == 0:
                    continue

                u_stat, raw_p = stats.mannwhitneyu(
                    values_1, values_2, alternative="two-sided"
                )
                corrected_p = min(raw_p * n_pairs, 1.0)
                posthoc_results.append(
                    {
                        "feature": feature,
                        "group1": group_1,
                        "group2": group_2,
                        "U_stat": round(float(u_stat), 2),
                        "p_raw": round(float(raw_p), 4),
                        "p_bonferroni": round(float(corrected_p), 4),
                        "significant": corrected_p < 0.05,
                    }
                )

    pd.DataFrame(kruskal_results).to_csv(
        output_dir / "kruskal_wallis.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(posthoc_results).to_csv(
        output_dir / "posthoc_mannwhitney.csv", index=False, encoding="utf-8-sig"
    )
    print("Statistical tests saved.")


# Output helpers
def save_group_mean_aecs(df: pd.DataFrame, tama_cutoff: float) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for group_name in GROUP_ORDER:
        group_df = df[df["group"] == group_name]
        if group_df.empty:
            continue

        aecs = np.vstack(group_df["aec"].to_numpy())
        mean_aec = aecs.mean(axis=0)
        std_aec = aecs.std(axis=0)
        x_axis = np.linspace(0, 1, len(mean_aec))
        ax.plot(
            x_axis,
            mean_aec,
            label=f"{group_name} (n={len(group_df)})",
            color=GROUP_COLORS[group_name],
        )
        ax.fill_between(
            x_axis,
            mean_aec - std_aec,
            mean_aec + std_aec,
            alpha=0.15,
            color=GROUP_COLORS[group_name],
        )

    ax.set_xlabel("Instance Number (Head → Leg)")
    ax.set_ylabel("X-ray Tube Current")
    ax.set_title(
        f"AEC Shape Comparison by BMI/TAMA Quadrants (TAMA mean={tama_cutoff:.2f})"
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    finalize_figure(fig, OUTPUT_DIR / "group_mean_aecs.png")


def save_group_count_plot(df: pd.DataFrame) -> None:
    counts = df["group"].value_counts().reindex(GROUP_ORDER, fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        counts.index,
        counts.values,
        color=[GROUP_COLORS[group] for group in counts.index],
    )
    ax.set_xlabel("Group")
    ax.set_ylabel("Count")
    ax.set_title("BMI/TAMA Quadrant Group Counts")
    ax.tick_params(axis="x")
    finalize_figure(fig, OUTPUT_DIR / "group_counts.png")


def save_feature_boxplots(df: pd.DataFrame) -> None:
    feature_cols = [
        "aec_mean",
        "aec_range",
        "end_minus_start",
        "mean_abs_slope",
        "left_mean",
        "right_mean",
        "auc",
        "cv",
        "asymmetry",
        "peak_position",
        "chest_slope",
        "abdomen_slope",
        "pelvis_slope",
    ]

    n_cols = 4
    n_rows = (len(feature_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, n_rows * 5))
    axes = axes.ravel()

    for ax, feature in zip(axes, feature_cols):
        data = [
            df.loc[df["group"] == group, feature].dropna().to_numpy()
            for group in GROUP_ORDER
        ]
        boxplot = ax.boxplot(data, tick_labels=GROUP_ORDER, patch_artist=True)
        for patch, group in zip(boxplot["boxes"], GROUP_ORDER):
            patch.set_facecolor(GROUP_COLORS[group])
            patch.set_alpha(0.6)
        ax.set_title(feature)
        ax.tick_params(axis="x")
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    for ax in axes[len(feature_cols) :]:
        ax.set_visible(False)

    finalize_figure(fig, OUTPUT_DIR / "shape_feature_boxplots.png")


def save_group_summary(df: pd.DataFrame, tama_cutoff: float) -> None:
    df[GROUP_MEMBER_COLS].sort_values(["group", "PatientID"]).to_csv(
        OUTPUT_DIR / "group_members.csv", index=False, encoding="utf-8-sig"
    )

    summary_df = (
        df.groupby("group")
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
            left_mean=("left_mean", "mean"),
            center_mean=("center_mean", "mean"),
            right_mean=("right_mean", "mean"),
            end_minus_start=("end_minus_start", "mean"),
            mean_abs_slope=("mean_abs_slope", "mean"),
        )
        .reindex(GROUP_ORDER)
        .reset_index()
    )
    summary_df["female_ratio_pct"] = (
        summary_df["female_count"] / summary_df["member_count"] * 100
    )
    summary_df["male_ratio_pct"] = (
        summary_df["male_count"] / summary_df["member_count"] * 100
    )
    summary_df["tama_cutoff_mean"] = tama_cutoff
    summary_df.to_csv(
        OUTPUT_DIR / "group_summary.csv", index=False, encoding="utf-8-sig"
    )


def save_report(df: pd.DataFrame, tama_cutoff: float) -> None:
    summary_df = (
        df.groupby("group")
        .agg(
            member_count=("PatientID", "size"),
            female_ratio_pct=("PatientSex", lambda s: float((s == "F").mean() * 100)),
            bmi_mean=("BMI", "mean"),
            tama_mean=("TAMA", "mean"),
            aec_mean=("aec_mean", "mean"),
            aec_range=("aec_range", "mean"),
            left_mean=("left_mean", "mean"),
            center_mean=("center_mean", "mean"),
            right_mean=("right_mean", "mean"),
            end_minus_start=("end_minus_start", "mean"),
            mean_abs_slope=("mean_abs_slope", "mean"),
        )
        .reindex(GROUP_ORDER)
        .reset_index()
    )

    lines = [
        "# BMI/TAMA Quadrant AEC Shape Analysis",
        "",
        f"- Site: {SITE_NAME}",
        f"- Patients analyzed: {len(df)}",
        f"- BMI cutoff: {BMI_CUTOFF:.2f}",
        f"- TAMA mean cutoff: {tama_cutoff:.2f}",
        "- AEC extraction: blue line traced from PNG and resampled to 128 points.",
        "",
        "## Group Summary",
        "",
    ]

    for row in summary_df.itertuples(index=False):
        lines.extend(
            [
                f"### {row.group}",
                f"- Members: {row.member_count}",
                f"- Female ratio: {row.female_ratio_pct:.1f}%",
                f"- BMI mean: {row.bmi_mean:.2f}",
                f"- TAMA mean: {row.tama_mean:.2f}",
                f"- Mean extracted height: {row.aec_mean:.2f}",
                f"- Mean AEC range: {row.aec_range:.2f}",
                f"- Left/Center/Right mean: {row.left_mean:.2f} / {row.center_mean:.2f} / {row.right_mean:.2f}",
                f"- Mean end-minus-start: {row.end_minus_start:.2f}",
                f"- Mean absolute slope: {row.mean_abs_slope:.2f}",
                "",
            ]
        )

    (OUTPUT_DIR / "analysis_report.md").write_text(
        "\n".join(lines), encoding="utf-8-sig"
    )


# Entry point
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clinical_df = load_clinical_data()
    dataset_df = build_dataset(clinical_df)
    tama_cutoff = float(dataset_df["TAMA"].mean())
    result_df = assign_quadrant_groups(dataset_df, tama_cutoff)

    run_statistical_tests(result_df, OUTPUT_DIR)
    save_group_mean_aecs(result_df, tama_cutoff)
    save_group_count_plot(result_df)
    save_feature_boxplots(result_df)
    save_group_summary(result_df, tama_cutoff)
    save_report(result_df, tama_cutoff)

    print(f"Clinical rows: {len(clinical_df)}")
    print(f"Merged rows: {len(result_df)}")
    print(f"TAMA mean cutoff: {tama_cutoff:.4f}")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
