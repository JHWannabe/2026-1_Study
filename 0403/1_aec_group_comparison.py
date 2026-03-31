import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FONT_SCALE = 1.3
SEX_FILTER_LABELS = {
    "F": "female",
    "M": "male",
}

plt.rcParams.update(
    {
        "font.size": 10 * FONT_SCALE,
        "axes.titlesize": 14 * FONT_SCALE,
        "axes.labelsize": 12 * FONT_SCALE,
        "xtick.labelsize": 10 * FONT_SCALE,
        "ytick.labelsize": 10 * FONT_SCALE,
        "legend.fontsize": 11 * FONT_SCALE,
    }
)

STUDY_ROOT = Path(__file__).resolve().parents[2]
AEC_ROOT = STUDY_ROOT / "data" / "AEC"
RESULT_ROOT = STUDY_ROOT / "result"
SITE_NAME = os.environ.get("AEC_SITE_NAME", "신촌")
REQUESTED_SEX_FILTER = os.environ.get("AEC_SEX_FILTER", "").strip().upper()
SEX_FILTER = ""
SEX_FILTER_DIR = ""


@dataclass
class CFG:
    excel_path: str = ""
    image_dir: str = ""
    out_dir: str = ""
    n_points: int = 256
    smooth_kernel: int = 9
    line_mask_min_pixels: int = 150
    max_images: int = 0
    y_min: float = 0.0
    y_max: float = 800.0


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
    global SEX_FILTER, SEX_FILTER_DIR

    SEX_FILTER = sex_filter
    SEX_FILTER_DIR = SEX_FILTER_LABELS[sex_filter]


# Configuration helpers
def discover_default_paths(site_name: str = SITE_NAME) -> Tuple[str, str, str]:
    if not AEC_ROOT.exists():
        raise FileNotFoundError(f"AEC root directory not found: {AEC_ROOT}")

    site_dir = AEC_ROOT / site_name
    image_dir = site_dir / "Image_Filter"
    excel_path = site_dir / "Result_Filter.xlsx"
    out_dir = (
        RESULT_ROOT
        / "0403"
        / site_dir.name
        / f"sex_{SEX_FILTER_DIR}"
        / "aec_group_comparison"
    )

    if not site_dir.is_dir():
        raise FileNotFoundError(
            f"Site directory not found for SITE_NAME='{site_name}': {site_dir}"
        )
    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"Image_Filter directory not found for SITE_NAME='{site_name}': {image_dir}"
        )
    if not excel_path.is_file():
        raise FileNotFoundError(
            f"Result_Filter.xlsx not found for SITE_NAME='{site_name}': {excel_path}"
        )

    return str(excel_path), str(image_dir), str(out_dir)


def parse_args() -> CFG:
    default_excel_path, default_image_dir, default_out_dir = discover_default_paths()

    parser = argparse.ArgumentParser(
        description="Compare AEC graphs by sex, SRC_Report median split, and BMI threshold."
    )
    parser.add_argument("--excel-path", default=default_excel_path)
    parser.add_argument("--image-dir", default=default_image_dir)
    parser.add_argument("--out-dir", default=default_out_dir)
    parser.add_argument("--n-points", type=int, default=CFG.n_points)
    parser.add_argument("--smooth-kernel", type=int, default=CFG.smooth_kernel)
    parser.add_argument(
        "--line-mask-min-pixels", type=int, default=CFG.line_mask_min_pixels
    )
    parser.add_argument("--max-images", type=int, default=CFG.max_images)
    parser.add_argument("--y-min", type=float, default=CFG.y_min)
    parser.add_argument("--y-max", type=float, default=CFG.y_max)
    args = parser.parse_args()

    return CFG(
        excel_path=args.excel_path,
        image_dir=args.image_dir,
        out_dir=args.out_dir,
        n_points=args.n_points,
        smooth_kernel=args.smooth_kernel,
        line_mask_min_pixels=args.line_mask_min_pixels,
        max_images=args.max_images,
        y_min=args.y_min,
        y_max=args.y_max,
    )


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# Data loading and AEC extraction helpers
def normalize_patient_sex(series: pd.Series) -> pd.Series:
    return (
        series.astype(str).str.strip().str.upper().replace({"MALE": "M", "FEMALE": "F"})
    )


def apply_sex_filter(df: pd.DataFrame) -> pd.DataFrame:
    filtered_df = df[df["PatientSex"] == SEX_FILTER].copy()
    if filtered_df.empty:
        raise ValueError(
            f"No rows found for site '{SITE_NAME}' with sex filter '{SEX_FILTER_DIR}'."
        )
    return filtered_df


def normalize_patient_id(value) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


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

    lower = np.array([100, 80, 80], dtype=np.uint8)
    upper = np.array([130, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def detect_plot_bounds(image_bgr: np.ndarray) -> Tuple[int, int, int, int]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    dark_mask = gray < 60

    col_counts = dark_mask.sum(axis=0)
    row_counts = dark_mask.sum(axis=1)

    strong_cols = np.where(col_counts > image_bgr.shape[0] * 0.5)[0]
    strong_rows = np.where(row_counts > image_bgr.shape[1] * 0.5)[0]

    if len(strong_cols) < 2 or len(strong_rows) < 2:
        raise ValueError("Plot bounds could not be detected.")

    left = int(strong_cols.min())
    right = int(strong_cols.max())
    top = int(strong_rows.min())
    bottom = int(strong_rows.max())

    if right <= left or bottom <= top:
        raise ValueError("Invalid plot bounds detected.")

    return left, right, top, bottom


def mask_to_raw_aec(
    mask: np.ndarray,
    plot_bounds: Tuple[int, int, int, int],
    n_points: int,
    y_min: float,
    y_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
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

    source_x = np.arange(x_min, x_max + 1, dtype=np.float32)
    curve_y = np.interp(
        source_x,
        source_x[valid],
        curve_y[valid],
    )

    target_source_x = np.linspace(
        float(x_min), float(x_max), n_points, dtype=np.float32
    )
    sampled_y = np.interp(target_source_x, source_x, curve_y).astype(np.float32)
    target_x = np.linspace(0.0, 1.0, n_points, dtype=np.float32)

    _, _, top, bottom = plot_bounds
    plot_height = float(bottom - top)
    if plot_height <= 0:
        raise ValueError("Plot height must be positive.")

    upward_ratio = (float(bottom) - sampled_y) / plot_height
    upward_ratio = np.clip(upward_ratio, 0.0, 1.0)
    raw_aec = y_min + upward_ratio * (y_max - y_min)
    return target_x, raw_aec.astype(np.float32)


def extract_aec(image_path: str, cfg: CFG) -> Tuple[np.ndarray, np.ndarray]:
    image_bgr = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Image could not be read.")

    mask = extract_blue_curve_mask(image_bgr)
    if int(mask.sum() / 255) < cfg.line_mask_min_pixels:
        raise ValueError("Detected line mask is too small.")

    plot_bounds = detect_plot_bounds(image_bgr)
    x_axis, aec = mask_to_raw_aec(
        mask=mask,
        plot_bounds=plot_bounds,
        n_points=cfg.n_points,
        y_min=cfg.y_min,
        y_max=cfg.y_max,
    )
    aec = smooth_1d(aec, cfg.smooth_kernel)
    return x_axis, aec


def extract_summary_features(aec: np.ndarray, x_axis: np.ndarray) -> Dict[str, float]:
    peak_idx = int(np.argmax(aec))
    trough_idx = int(np.argmin(aec))
    slopes = np.diff(aec)
    curvature = np.diff(aec, n=2) if len(aec) >= 3 else np.array([], dtype=np.float32)
    midpoint = len(aec) // 2

    left_part = aec[:midpoint] if midpoint > 0 else aec
    right_part = aec[midpoint:] if midpoint > 0 else aec

    return {
        "aec_mean": float(np.mean(aec)),
        "aec_std": float(np.std(aec)),
        "aec_min": float(np.min(aec)),
        "aec_max": float(np.max(aec)),
        "aec_range": float(np.max(aec) - np.min(aec)),
        "peak_x": float(x_axis[peak_idx]),
        "trough_x": float(x_axis[trough_idx]),
        "left_mean": float(np.mean(left_part)),
        "right_mean": float(np.mean(right_part)),
        "mean_abs_slope": float(np.mean(np.abs(slopes))) if len(slopes) > 0 else 0.0,
        "mean_abs_curvature": float(np.mean(np.abs(curvature)))
        if len(curvature) > 0
        else 0.0,
    }


def load_filtered_dataset(
    cfg: CFG,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[dict], Dict[str, float]]:
    df = pd.read_excel(cfg.excel_path).copy()
    required_cols = ["PatientID", "PatientSex", "SRC_Report", "BMI"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df["PatientID"] = df["PatientID"].apply(normalize_patient_id)
    df["PatientSex"] = normalize_patient_sex(df["PatientSex"])
    df = df[df["PatientID"] != ""].copy()
    df = df[df["PatientSex"].isin(["M", "F"])].copy()
    df = df[df["SRC_Report"].notna() & df["BMI"].notna()].copy()
    df = df.drop_duplicates(subset=["PatientID"]).copy()
    df = apply_sex_filter(df)

    image_dir = Path(cfg.image_dir)
    image_index = {path.stem: str(path) for path in image_dir.glob("*.png")}

    records: List[dict] = []
    x_axes: List[np.ndarray] = []
    aecs: List[np.ndarray] = []
    failures: List[dict] = []

    if cfg.max_images > 0:
        df = df.head(cfg.max_images).copy()

    for _, row in df.iterrows():
        patient_id = row["PatientID"]
        image_path = image_index.get(patient_id)
        if image_path is None:
            failures.append(
                {
                    "PatientID": patient_id,
                    "reason": "Matching PNG not found in Image_Filter",
                }
            )
            continue

        try:
            x_axis, aec = extract_aec(image_path, cfg)
        except Exception as exc:
            failures.append(
                {
                    "PatientID": patient_id,
                    "image_path": image_path,
                    "reason": str(exc),
                }
            )
            continue

        summary = extract_summary_features(aec, x_axis)
        records.append(
            {
                "PatientID": patient_id,
                "PatientSex": row["PatientSex"],
                "SRC_Report": float(row["SRC_Report"]),
                "BMI": float(row["BMI"]),
                "image_path": image_path,
                **summary,
            }
        )
        x_axes.append(x_axis)
        aecs.append(aec)

    if not records:
        raise RuntimeError("No valid matched images were available for analysis.")

    result_df = pd.DataFrame(records)
    aec_matrix = np.vstack(aecs).astype(np.float32)
    x_axis_matrix = np.vstack(x_axes).astype(np.float32)
    common_x_axis = x_axis_matrix.mean(axis=0).astype(np.float32)

    src_median = float(result_df["SRC_Report"].median())
    bmi_threshold = 25.0

    result_df["sex_group"] = result_df["PatientSex"]
    result_df["src_group"] = np.where(
        result_df["SRC_Report"] >= src_median, "SRC_high", "SRC_low"
    )
    result_df["bmi_group"] = np.where(
        result_df["BMI"] >= bmi_threshold, "BMI_ge_25", "BMI_lt_25"
    )

    metadata = {
        "src_report_mean": src_median,
        "bmi_threshold": bmi_threshold,
        "analyzed_count": float(len(result_df)),
    }
    return result_df, common_x_axis, aec_matrix, failures, metadata


# Output helpers
def save_mean_aec_plot(
    df: pd.DataFrame,
    x_axis: np.ndarray,
    aecs: np.ndarray,
    group_col: str,
    title: str,
    ylabel: str,
    out_path: str,
    y_min: float,
    y_max: float,
) -> None:
    plt.figure(figsize=(10, 5))

    for group_name in sorted(df[group_col].unique()):
        idx = df[group_col].values == group_name
        group_aecs = aecs[idx]
        mean_aec = group_aecs.mean(axis=0)
        std_aec = group_aecs.std(axis=0)
        plt.plot(
            x_axis, mean_aec, linewidth=2.5, label=f"{group_name} (n={len(group_aecs)})"
        )
        plt.fill_between(x_axis, mean_aec - std_aec, mean_aec + std_aec, alpha=0.2)

    plt.title(title)
    plt.xlabel("Instance Number (Head → Leg)")
    plt.ylabel(ylabel)
    plt.xlim(0.0, 1.0)
    plt.ylim(y_min, y_max)
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_overlay_plot(
    df: pd.DataFrame,
    x_axis: np.ndarray,
    aecs: np.ndarray,
    group_col: str,
    title: str,
    ylabel: str,
    out_path: str,
    y_min: float,
    y_max: float,
    max_per_group: int = 150,
) -> None:
    plt.figure(figsize=(10, 5))
    colors = ["steelblue", "darkorange", "forestgreen", "crimson"]

    for color_idx, group_name in enumerate(sorted(df[group_col].unique())):
        idx = np.where(df[group_col].values == group_name)[0]
        selected = idx[: min(len(idx), max_per_group)]
        color = colors[color_idx % len(colors)]

        for sample_idx in selected:
            plt.plot(x_axis, aecs[sample_idx], color=color, alpha=0.08, linewidth=1)

        mean_aec = aecs[idx].mean(axis=0)
        plt.plot(
            x_axis,
            mean_aec,
            color=color,
            linewidth=2.5,
            label=f"{group_name} mean (n={len(idx)})",
        )

    plt.title(title)
    plt.xlabel("Instance Number (Head → Leg)")
    plt.ylabel(ylabel)
    plt.xlim(0.0, 1.0)
    plt.ylim(y_min, y_max)
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_difference_plot(
    df: pd.DataFrame,
    x_axis: np.ndarray,
    aecs: np.ndarray,
    group_col: str,
    title: str,
    ylabel: str,
    out_path: str,
) -> None:
    groups = sorted(df[group_col].unique())
    if len(groups) != 2:
        return

    first_group = aecs[df[group_col].values == groups[0]].mean(axis=0)
    second_group = aecs[df[group_col].values == groups[1]].mean(axis=0)
    diff = second_group - first_group

    plt.figure(figsize=(10, 5))
    plt.plot(x_axis, diff, linewidth=2.5)
    plt.axhline(0.0, linestyle="--", linewidth=1, color="black")
    plt.title(title)
    plt.xlabel("Instance Number (Head → Leg)")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_feature_boxplots(df: pd.DataFrame, group_col: str, out_dir: str) -> None:
    feature_cols = [
        "aec_mean",
        "aec_std",
        "aec_range",
        "peak_x",
        "mean_abs_slope",
        "mean_abs_curvature",
    ]
    groups = sorted(df[group_col].unique())

    for feature_col in feature_cols:
        plt.figure(figsize=(6, 4))
        data = [
            df.loc[df[group_col] == group_name, feature_col].dropna().values
            for group_name in groups
        ]
        plt.boxplot(data, tick_labels=groups)
        plt.title(f"{feature_col} by {group_col}")
        plt.ylabel(feature_col)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{feature_col}_boxplot.png"), dpi=200)
        plt.close()


def save_group_stats(df: pd.DataFrame, group_col: str, out_dir: str) -> None:
    summary_cols = [
        "SRC_Report",
        "BMI",
        "aec_mean",
        "aec_std",
        "aec_min",
        "aec_max",
        "aec_range",
        "peak_x",
        "trough_x",
        "left_mean",
        "right_mean",
        "mean_abs_slope",
        "mean_abs_curvature",
    ]

    stats_df = df.groupby(group_col)[summary_cols].agg(
        ["count", "mean", "std", "median"]
    )
    stats_df.to_csv(os.path.join(out_dir, "group_statistics.csv"), encoding="utf-8-sig")

    member_df = df.sort_values([group_col, "PatientID"]).copy()
    member_df.to_csv(
        os.path.join(out_dir, "group_members.csv"), index=False, encoding="utf-8-sig"
    )


def save_group_curve_csv(
    df: pd.DataFrame,
    x_axis: np.ndarray,
    aecs: np.ndarray,
    group_col: str,
    out_path: str,
) -> None:
    curve_df = pd.DataFrame({"x": x_axis})
    for group_name in sorted(df[group_col].unique()):
        idx = df[group_col].values == group_name
        curve_df[f"{group_name}_mean"] = aecs[idx].mean(axis=0)
        curve_df[f"{group_name}_std"] = aecs[idx].std(axis=0)
    curve_df.to_csv(out_path, index=False, encoding="utf-8-sig")


def run_group_analysis(
    df: pd.DataFrame,
    x_axis: np.ndarray,
    aecs: np.ndarray,
    group_col: str,
    label: str,
    out_root: str,
) -> None:
    group_dir = os.path.join(out_root, label)
    ensure_dir(group_dir)

    save_mean_aec_plot(
        df=df,
        x_axis=x_axis,
        aecs=aecs,
        group_col=group_col,
        title=f"{label}: AEC comparison",
        ylabel="X-ray Tube Current (mA)",
        out_path=os.path.join(group_dir, "mean_aec.png"),
        y_min=0.0,
        y_max=800.0,
    )
    save_overlay_plot(
        df=df,
        x_axis=x_axis,
        aecs=aecs,
        group_col=group_col,
        title=f"{label}: AEC overlay",
        ylabel="X-ray Tube Current (mA)",
        out_path=os.path.join(group_dir, "overlay.png"),
        y_min=0.0,
        y_max=800.0,
    )
    save_difference_plot(
        df=df,
        x_axis=x_axis,
        aecs=aecs,
        group_col=group_col,
        title=f"{label}: mean difference",
        ylabel="Tube Current Difference (mA)",
        out_path=os.path.join(group_dir, "mean_difference.png"),
    )
    save_feature_boxplots(df, group_col, group_dir)
    save_group_stats(df, group_col, group_dir)
    save_group_curve_csv(
        df=df,
        x_axis=x_axis,
        aecs=aecs,
        group_col=group_col,
        out_path=os.path.join(group_dir, "mean_curves.csv"),
    )


def save_overall_outputs(
    df: pd.DataFrame,
    failures: List[dict],
    metadata: Dict[str, float],
    out_dir: str,
) -> None:
    ensure_dir(out_dir)
    df.to_csv(
        os.path.join(out_dir, "matched_analysis_data.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    if failures:
        pd.DataFrame(failures).to_csv(
            os.path.join(out_dir, "failed_or_unmatched_cases.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("=== AEC Group Comparison Summary ===\n\n")
        f.write(f"Analyzed cases: {len(df)}\n")
        f.write(f"Failed or unmatched cases: {len(failures)}\n")
        f.write(f"SRC_Report median: {metadata['src_report_mean']:.4f}\n")
        f.write(f"BMI threshold: {metadata['bmi_threshold']:.1f}\n\n")
        f.write("[Sex counts]\n")
        f.write(df["sex_group"].value_counts().to_string())
        f.write("\n\n[SRC_Report median split counts]\n")
        f.write(df["src_group"].value_counts().to_string())
        f.write("\n\n[BMI threshold counts]\n")
        f.write(df["bmi_group"].value_counts().to_string())
        f.write("\n")


# Entry point
def run_single_sex_filter() -> None:
    cfg = parse_args()
    ensure_dir(cfg.out_dir)

    df, x_axis, aecs, failures, metadata = load_filtered_dataset(cfg)
    save_overall_outputs(df, failures, metadata, cfg.out_dir)

    run_group_analysis(
        df=df,
        x_axis=x_axis,
        aecs=aecs,
        group_col="sex_group",
        label="sex",
        out_root=cfg.out_dir,
    )
    run_group_analysis(
        df=df,
        x_axis=x_axis,
        aecs=aecs,
        group_col="src_group",
        label="TAMA",
        out_root=cfg.out_dir,
    )
    run_group_analysis(
        df=df,
        x_axis=x_axis,
        aecs=aecs,
        group_col="bmi_group",
        label="BMI",
        out_root=cfg.out_dir,
    )

    print("Done")
    print(f"Sex filter: {SEX_FILTER_DIR}")
    print(f"Analyzed cases: {len(df)}")
    print(f"Failed or unmatched cases: {len(failures)}")
    print(f"SRC_Report median: {metadata['src_report_mean']}")
    print(f"Output dir: {cfg.out_dir}")


def main() -> None:
    for sex_filter in resolve_requested_sex_filters():
        configure_sex_filter(sex_filter)
        run_single_sex_filter()


if __name__ == "__main__":
    main()
