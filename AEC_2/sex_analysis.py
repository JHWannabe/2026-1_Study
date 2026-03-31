import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================================================
# Config
# =========================================================
@dataclass
class CFG:
    # 중요: 실제 엑셀 환자 ID 컬럼명으로 수정 필요
    id_col: str = "PatientID"

    # None이면 첫 번째 시트 사용
    sheet_name: Optional[str] = None

    n_points: int = 256
    smooth_kernel: int = 9
    line_mask_min_pixels: int = 150
    normalize_mode: str = "zscore"   # ["none", "zscore", "minmax"]
    
    excel_path: str = r"C:\Users\user\Desktop\Study\data\AEC\신촌\신촌_DLO_Results.xlsx"
    image_dir: str = r"C:\Users\user\Desktop\Study\data\AEC\신촌\AEC"
    out_dir: str = rf"C:\Users\user\Desktop\Study\result\Case3\신촌\{normalize_mode}"

    # 디버깅용
    max_images: int = 0


# =========================================================
# Argparse
# =========================================================
def parse_args() -> CFG:
    parser = argparse.ArgumentParser(
        description="Compare AEC graph shape between male and female groups."
    )
    parser.add_argument("--excel-path", default=CFG.excel_path)
    parser.add_argument("--image-dir", default=CFG.image_dir)
    parser.add_argument("--out-dir", default=CFG.out_dir)
    parser.add_argument("--id-col", default=CFG.id_col)
    parser.add_argument("--sheet-name", default=CFG.sheet_name)
    parser.add_argument("--n-points", type=int, default=CFG.n_points)
    parser.add_argument("--smooth-kernel", type=int, default=CFG.smooth_kernel)
    parser.add_argument("--line-mask-min-pixels", type=int, default=CFG.line_mask_min_pixels)
    parser.add_argument(
        "--normalize-mode",
        choices=["none", "zscore", "minmax"],
        default=CFG.normalize_mode,
    )
    parser.add_argument("--max-images", type=int, default=CFG.max_images)

    args = parser.parse_args()

    cfg = CFG(
        excel_path=args.excel_path,
        image_dir=args.image_dir,
        out_dir=args.out_dir,
        id_col=args.id_col,
        sheet_name=args.sheet_name,
        n_points=args.n_points,
        smooth_kernel=args.smooth_kernel,
        line_mask_min_pixels=args.line_mask_min_pixels,
        normalize_mode=args.normalize_mode,
        max_images=args.max_images,
    )
    return cfg


# =========================================================
# Utils
# =========================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_excel_safe(excel_path: str, sheet_name=None) -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    # sheet_name=None 이면 dict 반환 가능
    if isinstance(df, dict):
        sheet_names = list(df.keys())
        print("현재 workbook의 시트 목록:", sheet_names)
        first_sheet = sheet_names[0]
        print(f"첫 번째 시트 '{first_sheet}'를 사용합니다.")
        df = df[first_sheet].copy()

    return df


def normalize_profile(profile: np.ndarray, mode: str) -> np.ndarray:
    profile = profile.astype(np.float32)

    if mode == "none":
        return profile

    if mode == "minmax":
        lo = float(np.min(profile))
        hi = float(np.max(profile))
        if hi - lo < 1e-6:
            return np.zeros_like(profile, dtype=np.float32)
        return (profile - lo) / (hi - lo)

    if mode == "zscore":
        mean = float(np.mean(profile))
        std = float(np.std(profile))
        if std < 1e-6:
            return np.zeros_like(profile, dtype=np.float32)
        return (profile - mean) / std

    raise ValueError(f"Unsupported normalize mode: {mode}")


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

    # ✅ 핵심 변경 (직관적 좌표계)
    height = mask.shape[0]
    curve_y = height - curve_y

    # ✅ 선택: 정규화 (강력 추천)
    curve_y = curve_y - np.min(curve_y)
    if np.max(curve_y) > 0:
        curve_y = curve_y / np.max(curve_y)

    resample_x = np.linspace(0, width - 1, n_points, dtype=np.float32)
    profile = np.interp(resample_x, full_index, curve_y).astype(np.float32)
    return profile


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


def find_matching_image(patient_id: str, image_dir: str) -> Optional[str]:
    """
    파일명과 patient_id 매칭
    우선순위:
    1) stem == patient_id
    2) stem startswith(patient_id + "_")
    3) stem.split("_")[0] == patient_id
    """
    patient_id = str(patient_id).strip()

    for name in os.listdir(image_dir):
        if not name.lower().endswith(".png"):
            continue
        stem = os.path.splitext(name)[0].strip()

        if stem == patient_id:
            return os.path.join(image_dir, name)

    for name in os.listdir(image_dir):
        if not name.lower().endswith(".png"):
            continue
        stem = os.path.splitext(name)[0].strip()

        if stem.startswith(patient_id + "_"):
            return os.path.join(image_dir, name)

    for name in os.listdir(image_dir):
        if not name.lower().endswith(".png"):
            continue
        stem = os.path.splitext(name)[0].strip()

        if stem.split("_")[0] == patient_id:
            return os.path.join(image_dir, name)

    return None


def safe_shapiro(x: np.ndarray) -> float:
    from scipy.stats import shapiro
    x = np.asarray(x, dtype=np.float32)
    x = x[~np.isnan(x)]
    if len(x) < 3 or len(x) > 5000:
        return np.nan
    try:
        return float(shapiro(x).pvalue)
    except Exception:
        return np.nan


def compare_numeric_groups(f_vals: np.ndarray, m_vals: np.ndarray) -> Dict[str, float]:
    from scipy.stats import ttest_ind, mannwhitneyu

    f_vals = np.asarray(f_vals, dtype=np.float32)
    m_vals = np.asarray(m_vals, dtype=np.float32)

    f_vals = f_vals[~np.isnan(f_vals)]
    m_vals = m_vals[~np.isnan(m_vals)]

    out = {
        "F_n": int(len(f_vals)),
        "M_n": int(len(m_vals)),
        "F_mean": float(np.mean(f_vals)) if len(f_vals) > 0 else np.nan,
        "M_mean": float(np.mean(m_vals)) if len(m_vals) > 0 else np.nan,
        "F_std": float(np.std(f_vals, ddof=1)) if len(f_vals) > 1 else np.nan,
        "M_std": float(np.std(m_vals, ddof=1)) if len(m_vals) > 1 else np.nan,
        "F_median": float(np.median(f_vals)) if len(f_vals) > 0 else np.nan,
        "M_median": float(np.median(m_vals)) if len(m_vals) > 0 else np.nan,
        "F_shapiro_p": safe_shapiro(f_vals),
        "M_shapiro_p": safe_shapiro(m_vals),
        "welch_t_p": np.nan,
        "mannwhitney_p": np.nan,
    }

    if len(f_vals) >= 2 and len(m_vals) >= 2:
        try:
            out["welch_t_p"] = float(ttest_ind(f_vals, m_vals, equal_var=False, nan_policy="omit").pvalue)
        except Exception:
            pass

        try:
            out["mannwhitney_p"] = float(mannwhitneyu(f_vals, m_vals, alternative="two-sided").pvalue)
        except Exception:
            pass

    return out


# =========================================================
# Data loading / matching
# =========================================================
def load_filtered_clinical_data(cfg: CFG) -> pd.DataFrame:
    df = read_excel_safe(cfg.excel_path, cfg.sheet_name)

    print(f"원본 데이터 shape: {df.shape}")
    print("컬럼 목록:")
    print(df.columns.tolist())

    required_cols = [cfg.id_col, "SRC_Report", "BMI", "PatientSex"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"필수 컬럼이 없습니다: {missing_cols}")

    df = df.copy()
    df = df[df["SRC_Report"].notna() & df["BMI"].notna()]
    df = df[df["PatientSex"].isin(["M", "F"])].copy()

    print(f"SRC_Report/BMI 존재 + 성별(M/F) 필터 후 shape: {df.shape}")
    print(df["PatientSex"].value_counts(dropna=False))

    dup_n = int(df.duplicated(subset=[cfg.id_col]).sum())
    print(f"중복 ID 개수: {dup_n}")

    df = df.drop_duplicates(subset=[cfg.id_col]).copy()
    print(f"중복 제거 후 shape: {df.shape}")

    return df


def load_features_with_clinical(cfg: CFG) -> Tuple[pd.DataFrame, np.ndarray, List[dict], pd.DataFrame]:
    clinical_df = load_filtered_clinical_data(cfg)

    records = []
    features = []
    failures = []
    unmatched_rows = []

    if cfg.max_images > 0:
        clinical_df = clinical_df.head(cfg.max_images).copy()

    for _, row in clinical_df.iterrows():
        patient_id = str(row[cfg.id_col]).strip()
        sex = row["PatientSex"]
        src_report = row["SRC_Report"]
        bmi = row["BMI"]

        image_path = find_matching_image(patient_id, cfg.image_dir)
        if image_path is None:
            unmatched_rows.append({
                cfg.id_col: patient_id,
                "PatientSex": sex,
                "SRC_Report": src_report,
                "BMI": bmi,
                "reason": "No matching PNG image",
            })
            continue

        image_name = os.path.basename(image_path)

        try:
            feature = extract_shape_feature(image_path, cfg)
            features.append(feature)
            records.append({
                cfg.id_col: patient_id,
                "patient_id": patient_id,
                "image_name": image_name,
                "image_path": image_path,
                "PatientSex": sex,
                "SRC_Report": src_report,
                "BMI": bmi,
            })
        except Exception as exc:
            failures.append({
                cfg.id_col: patient_id,
                "image_name": image_name,
                "image_path": image_path,
                "error": str(exc),
            })

    if not features:
        raise RuntimeError("No valid graph profiles were extracted.")

    df = pd.DataFrame(records)
    X = np.vstack(features).astype(np.float32)
    unmatched_df = pd.DataFrame(unmatched_rows)

    return df, X, failures, unmatched_df


# =========================================================
# Shape statistics
# =========================================================
def extract_profile_summary_features(X: np.ndarray) -> pd.DataFrame:
    """
    profile 자체를 바로 비교하는 것도 중요하지만,
    해석 가능한 요약 feature도 같이 뽑아두면 좋음
    """
    rows = []
    x_axis = np.linspace(0.0, 1.0, X.shape[1], dtype=np.float32)

    for profile in X:
        peak_idx = int(np.argmax(profile))
        trough_idx = int(np.argmin(profile))

        row = {
            "profile_mean": float(np.mean(profile)),
            "profile_std": float(np.std(profile)),
            "profile_min": float(np.min(profile)),
            "profile_max": float(np.max(profile)),
            "profile_range": float(np.max(profile) - np.min(profile)),
            "peak_position": float(x_axis[peak_idx]),
            "trough_position": float(x_axis[trough_idx]),
            "slope_mean": float(np.mean(np.diff(profile))),
            "abs_slope_mean": float(np.mean(np.abs(np.diff(profile)))),
            "curvature_mean": float(np.mean(np.abs(np.diff(profile, n=2)))) if len(profile) >= 3 else np.nan,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def compare_profile_by_sex(X: np.ndarray, sex_labels: np.ndarray) -> pd.DataFrame:
    rows = []
    n_points = X.shape[1]

    for i in range(n_points):
        f_vals = X[sex_labels == "F", i]
        m_vals = X[sex_labels == "M", i]
        stat = compare_numeric_groups(f_vals, m_vals)
        stat["point_idx"] = i
        rows.append(stat)

    return pd.DataFrame(rows)


# =========================================================
# Plotting
# =========================================================
def save_mean_profile_plot(df: pd.DataFrame, X: np.ndarray, cfg: CFG) -> None:
    sex_labels = df["PatientSex"].values
    x_axis = np.linspace(0.0, 1.0, X.shape[1])

    X_f = X[sex_labels == "F"]
    X_m = X[sex_labels == "M"]

    plt.figure(figsize=(11, 5))

    if len(X_f) > 0:
        mean_f = X_f.mean(axis=0)
        std_f = X_f.std(axis=0)
        plt.plot(x_axis, mean_f, linewidth=2.5, label=f"F mean (n={len(X_f)})")
        plt.fill_between(x_axis, mean_f - std_f, mean_f + std_f, alpha=0.2)

    if len(X_m) > 0:
        mean_m = X_m.mean(axis=0)
        std_m = X_m.std(axis=0)
        plt.plot(x_axis, mean_m, linewidth=2.5, label=f"M mean (n={len(X_m)})")
        plt.fill_between(x_axis, mean_m - std_m, mean_m + std_m, alpha=0.2)

    plt.title("AEC graph mean profile by sex")
    plt.xlabel("Normalized instance number")
    plt.ylabel(f"Curve profile ({cfg.normalize_mode})")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "sex_mean_profile.png"), dpi=200)
    plt.close()


def save_all_profiles_plot(df: pd.DataFrame, X: np.ndarray, cfg: CFG) -> None:
    x_axis = np.linspace(0.0, 1.0, X.shape[1])
    sex_labels = df["PatientSex"].values

    plt.figure(figsize=(12, 5))

    for profile in X[sex_labels == "F"][:200]:
        plt.plot(x_axis, profile, alpha=0.08, linewidth=1)

    for profile in X[sex_labels == "M"][:200]:
        plt.plot(x_axis, profile, alpha=0.08, linewidth=1)

    plt.title("AEC graph profiles by sex")
    plt.xlabel("Normalized instance number")
    plt.ylabel(f"Curve profile ({cfg.normalize_mode})")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "sex_all_profiles_overlay.png"), dpi=200)
    plt.close()


def save_profile_difference_plot(df: pd.DataFrame, X: np.ndarray, cfg: CFG) -> None:
    x_axis = np.linspace(0.0, 1.0, X.shape[1])
    sex_labels = df["PatientSex"].values

    X_f = X[sex_labels == "F"]
    X_m = X[sex_labels == "M"]

    if len(X_f) == 0 or len(X_m) == 0:
        return

    mean_f = X_f.mean(axis=0)
    mean_m = X_m.mean(axis=0)
    diff = mean_m - mean_f

    plt.figure(figsize=(11, 4.5))
    plt.plot(x_axis, diff, linewidth=2.5, label="M - F")
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.title("Mean profile difference (M - F)")
    plt.xlabel("Normalized instance number")
    plt.ylabel("Difference")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "sex_mean_profile_difference.png"), dpi=200)
    plt.close()


def save_distribution_plot(df: pd.DataFrame, col: str, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    f_vals = df.loc[df["PatientSex"] == "F", col].dropna().values
    m_vals = df.loc[df["PatientSex"] == "M", col].dropna().values

    all_vals = np.concatenate([f_vals, m_vals]) if len(f_vals) + len(m_vals) > 0 else np.array([])

    if len(all_vals) == 0:
        plt.close()
        return

    # 공통 구간으로 동일한 폭의 bin 생성
    n_bins = 20
    vmin = float(np.min(all_vals))
    vmax = float(np.max(all_vals))

    # 값이 모두 동일한 경우 예외 처리
    if abs(vmax - vmin) < 1e-8:
        vmin -= 0.5
        vmax += 0.5

    bin_edges = np.linspace(vmin, vmax, n_bins + 1)

    axes[0].hist(f_vals, bins=bin_edges, alpha=0.5, label="F")
    axes[0].hist(m_vals, bins=bin_edges, alpha=0.5, label="M")
    axes[0].set_xlim(vmin, vmax)
    axes[0].set_title(f"{col} histogram by sex")
    axes[0].set_xlabel(col)
    axes[0].set_ylabel("Count")
    axes[0].legend()

    data = [f_vals, m_vals]
    axes[1].boxplot(data, tick_labels=["F", "M"])
    axes[1].set_title(f"{col} boxplot by sex")
    axes[1].set_ylabel(col)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_summary_feature_boxplots(df: pd.DataFrame, feature_cols: List[str], cfg: CFG) -> None:
    out_dir = os.path.join(cfg.out_dir, "summary_feature_boxplots")
    ensure_dir(out_dir)

    for col in feature_cols:
        plt.figure(figsize=(6, 4))
        data = [
            df.loc[df["PatientSex"] == "F", col].dropna().values,
            df.loc[df["PatientSex"] == "M", col].dropna().values,
        ]
        plt.boxplot(data, tick_labels=["F", "M"])
        plt.title(f"{col} by sex")
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{col}.png"), dpi=200)
        plt.close()


# =========================================================
# Save outputs
# =========================================================
def save_outputs(
    base_df: pd.DataFrame,
    X: np.ndarray,
    failures: List[dict],
    unmatched_df: pd.DataFrame,
    cfg: CFG,
) -> None:
    ensure_dir(cfg.out_dir)

    # 원본 + profile summary 결합
    summary_feat_df = extract_profile_summary_features(X)
    result_df = pd.concat([base_df.reset_index(drop=True), summary_feat_df], axis=1)

    result_df.to_csv(
        os.path.join(cfg.out_dir, "sex_shape_analysis_data.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    if failures:
        pd.DataFrame(failures).to_csv(
            os.path.join(cfg.out_dir, "failed_images.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    if len(unmatched_df) > 0:
        unmatched_df.to_csv(
            os.path.join(cfg.out_dir, "unmatched_cases.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    # profile point-wise 비교
    pointwise_stats_df = compare_profile_by_sex(X, result_df["PatientSex"].values)
    pointwise_stats_df.to_csv(
        os.path.join(cfg.out_dir, "profile_pointwise_sex_comparison.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # summary feature 비교
    summary_cols = list(summary_feat_df.columns)
    stat_rows = []
    for col in summary_cols:
        f_vals = result_df.loc[result_df["PatientSex"] == "F", col].values
        m_vals = result_df.loc[result_df["PatientSex"] == "M", col].values
        stat = compare_numeric_groups(f_vals, m_vals)
        stat["feature"] = col
        stat_rows.append(stat)

    summary_stats_df = pd.DataFrame(stat_rows)
    summary_stats_df.to_csv(
        os.path.join(cfg.out_dir, "summary_feature_sex_comparison.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # SRC_Report, BMI 성별 비교
    clinical_stat_rows = []
    for col in ["SRC_Report", "BMI"]:
        f_vals = result_df.loc[result_df["PatientSex"] == "F", col].values
        m_vals = result_df.loc[result_df["PatientSex"] == "M", col].values
        stat = compare_numeric_groups(f_vals, m_vals)
        stat["feature"] = col
        clinical_stat_rows.append(stat)

        save_distribution_plot(
            result_df,
            col=col,
            out_path=os.path.join(cfg.out_dir, f"{col}_distribution_by_sex.png"),
        )

    clinical_stats_df = pd.DataFrame(clinical_stat_rows)
    clinical_stats_df.to_csv(
        os.path.join(cfg.out_dir, "clinical_sex_comparison.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # 기본 plot
    save_mean_profile_plot(result_df, X, cfg)
    save_all_profiles_plot(result_df, X, cfg)
    save_profile_difference_plot(result_df, X, cfg)
    save_summary_feature_boxplots(result_df, summary_cols, cfg)

    # 성별 요약
    sex_counts = result_df["PatientSex"].value_counts().reset_index()
    sex_counts.columns = ["PatientSex", "count"]
    sex_counts.to_csv(
        os.path.join(cfg.out_dir, "sex_counts.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # 텍스트 요약
    with open(os.path.join(cfg.out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("=== Sex-based AEC Graph Shape Analysis Summary ===\n\n")
        f.write(f"Normalize mode: {cfg.normalize_mode}\n")
        f.write(f"n_points: {cfg.n_points}\n")
        f.write(f"smooth_kernel: {cfg.smooth_kernel}\n\n")
        f.write(f"Total analyzed: {len(result_df)}\n")
        f.write(f"F count: {(result_df['PatientSex'] == 'F').sum()}\n")
        f.write(f"M count: {(result_df['PatientSex'] == 'M').sum()}\n")
        f.write(f"Failed images: {len(failures)}\n")
        f.write(f"Unmatched cases: {len(unmatched_df)}\n\n")

        f.write("[Clinical comparison]\n")
        if len(clinical_stats_df) > 0:
            f.write(clinical_stats_df.to_string(index=False))
            f.write("\n\n")

        f.write("[Summary shape feature comparison]\n")
        if len(summary_stats_df) > 0:
            f.write(summary_stats_df.to_string(index=False))
            f.write("\n")


# =========================================================
# Main
# =========================================================
def main():
    cfg = parse_args()
    ensure_dir(cfg.out_dir)

    print("[1] Loading filtered clinical data + matching images...")
    df, X, failures, unmatched_df = load_features_with_clinical(cfg)

    print(f"[2] 분석 대상 수: {len(df)}")
    print(df["PatientSex"].value_counts(dropna=False))

    print("[3] Saving outputs...")
    save_outputs(df, X, failures, unmatched_df, cfg)

    print("완료되었습니다.")
    print("결과 폴더:", cfg.out_dir)


if __name__ == "__main__":
    main()