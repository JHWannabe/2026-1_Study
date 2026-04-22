import os
import pydicom
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from scipy import stats, signal
import pywt

AEC_POINTS = 128

def find_plot_y_bounds(image_bgr) -> tuple[int, int]:
    """
    Hough 변환으로 긴 수평선을 검출해 plot 영역의 top(y=800) / bottom(y=0) 픽셀 행을 반환.
    검출 실패 시 이미지 전체 높이를 사용.
    """
    height, width = image_bgr.shape[:2]
    gray  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=width // 5,
        minLineLength=width // 4,
        maxLineGap=50,
    )

    if lines is None:
        return 0, height

    h_ys = [
        (y1 + y2) / 2
        for _, y1, _, y2 in lines[:, 0]
        if abs(y2 - y1) < 3
    ]

    if len(h_ys) < 2:
        return 0, height

    return int(min(h_ys)), int(max(h_ys))


def extract_aec(image_path: Path, n_points: int = AEC_POINTS, y_max: float = 800.0) -> np.ndarray:
    raw = np.fromfile(str(image_path), dtype=np.uint8)
    if raw.size == 0:
        raise ValueError(f"Empty file (0 bytes): {image_path}")
    image_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Image decode failed: {image_path}")

    # HSV 변환 후 파란색 마스크 (Hue 100~130°)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 80, 80], dtype=np.uint8)
    upper_blue = np.array([130, 255, 255], dtype=np.uint8)
    blue_mask  = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel    = np.ones((3, 3), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

    y_idx, x_idx = np.where(blue_mask > 0)
    if len(x_idx) == 0:
        raise ValueError(f"Blue line not found: {image_path}")

    unique_x = np.unique(x_idx)
    mean_y   = np.array([y_idx[x_idx == x].mean() for x in unique_x], dtype=float)
    target_x = np.linspace(unique_x.min(), unique_x.max(), n_points)
    interp_y = np.interp(target_x, unique_x, mean_y)  # 픽셀 행 (위=0)

    # 그래프 영역 경계 자동 검출
    plot_top, plot_bottom = find_plot_y_bounds(image_bgr)  # top=y_max, bottom=y=0

    # 픽셀 → 실제 mA 값 변환
    # plot_bottom 픽셀행 → 0 mA, plot_top 픽셀행 → y_max mA
    span = plot_bottom - plot_top
    if span <= 0:
        raise ValueError(f"Plot bounds detection failed: top={plot_top}, bottom={plot_bottom}")

    return (plot_bottom - interp_y) / span * y_max

def extract_aec_features(aec: np.ndarray) -> dict:
    """
    AEC 곡선(1-D array, 길이 n_points)에서 통계·형태·주파수 feature를 추출.

    [중복 고려]
    - AUC (shape): 시간 도메인 적분 → 총 선량 누적량 의미
    - spectral_energy (frequency): 주파수 도메인 파워 총합 → 곡선 진동 에너지
      두 값은 파슨발 정리로 관련되지만, 스케일·해석이 달라 모두 포함.
    - skewness/kurtosis (stat)와 spectral_centroid (frequency)는 도메인이 달라 비중복.
    """
    feats = {}
    x = np.asarray(aec, dtype=float)
    n = len(x)

    # ── 1. 기본 통계 feature ──────────────────────────────────────────
    feats["stat_mean"]     = float(np.mean(x))
    feats["stat_std"]      = float(np.std(x, ddof=1))
    feats["stat_min"]      = float(np.min(x))
    feats["stat_max"]      = float(np.max(x))
    feats["stat_range"]    = float(np.max(x) - np.min(x))
    feats["stat_median"]   = float(np.median(x))
    q75, q25               = np.percentile(x, [75, 25])
    feats["stat_iqr"]      = float(q75 - q25)
    feats["stat_skewness"] = float(stats.skew(x))
    feats["stat_kurtosis"] = float(stats.kurtosis(x))  # Fisher (정규=0)

    # 구간별 평균 (4등분)
    for i, seg in enumerate(np.array_split(x, 4), 1):
        feats[f"stat_seg{i}_mean"] = float(np.mean(seg))

    # ── 2. Shape feature ─────────────────────────────────────────────
    # Peak
    peak_idx, _ = signal.find_peaks(x, prominence=5)
    feats["shape_n_peaks"]          = int(len(peak_idx))
    if len(peak_idx) > 0:
        heights = x[peak_idx]
        top_idx = peak_idx[np.argmax(heights)]
        feats["shape_peak_max_height"]    = float(np.max(heights))
        feats["shape_peak_max_pos"]       = float(top_idx / n)          # 0~1 정규화
        widths, *_ = signal.peak_widths(x, peak_idx, rel_height=0.5)
        feats["shape_peak_mean_width"]    = float(np.mean(widths))
    else:
        feats["shape_peak_max_height"]    = 0.0
        feats["shape_peak_max_pos"]       = float("nan")
        feats["shape_peak_mean_width"]    = 0.0

    # Valley (피크 반전)
    valley_idx, _ = signal.find_peaks(-x, prominence=5)
    feats["shape_n_valleys"] = int(len(valley_idx))

    # Slope 평균 (이웃 차분)
    slopes = np.diff(x)
    feats["shape_slope_mean"]    = float(np.mean(slopes))
    feats["shape_slope_abs_mean"]= float(np.mean(np.abs(slopes)))
    feats["shape_slope_std"]     = float(np.std(slopes, ddof=1))

    # Area under curve (사다리꼴 적분, x축=인덱스)
    feats["shape_auc"] = float(np.trapezoid(x))

    # ── 3. 주파수 feature ────────────────────────────────────────────
    # FFT (DC 제거, 단측 스펙트럼)
    fft_vals  = np.fft.rfft(x - np.mean(x))
    fft_mag   = np.abs(fft_vals)
    freqs     = np.fft.rfftfreq(n)                     # 0 ~ 0.5 (정규화 주파수)
    freqs_pos = freqs[1:]                              # DC(0) 제외
    mag_pos   = fft_mag[1:]

    feats["freq_fft_mean_mag"]      = float(np.mean(mag_pos))
    feats["freq_fft_max_mag"]       = float(np.max(mag_pos)) if len(mag_pos) > 0 else 0.0

    # Dominant frequency (가장 에너지 큰 성분의 정규화 주파수)
    if len(mag_pos) > 0:
        dom_idx = np.argmax(mag_pos)
        feats["freq_dominant_freq"] = float(freqs_pos[dom_idx])
    else:
        feats["freq_dominant_freq"] = float("nan")

    # Spectral centroid (주파수의 가중 평균)
    mag_sum = np.sum(mag_pos)
    feats["freq_spectral_centroid"] = (
        float(np.sum(freqs_pos * mag_pos) / mag_sum) if mag_sum > 0 else float("nan")
    )

    # Spectral energy (주파수 도메인 파워 총합; 시간 도메인 AUC와 보완적)
    feats["freq_spectral_energy"] = float(np.sum(mag_pos ** 2))

    # Band-wise energy (저·중·고 3구간)
    third = len(mag_pos) // 3
    feats["freq_band_low_energy"]  = float(np.sum(mag_pos[:third] ** 2))
    feats["freq_band_mid_energy"]  = float(np.sum(mag_pos[third:2*third] ** 2))
    feats["freq_band_high_energy"] = float(np.sum(mag_pos[2*third:] ** 2))

    # Wavelet (db4, level 3) — 각 레벨의 detail 계수 에너지
    coeffs = pywt.wavedec(x, wavelet="db4", level=3)
    for lvl, c in enumerate(coeffs[1:], 1):          # coeffs[0]=근사, [1~]=세부
        feats[f"freq_wavelet_d{lvl}_energy"] = float(np.sum(c ** 2))
        feats[f"freq_wavelet_d{lvl}_std"]    = float(np.std(c))

    return feats


SITE = "강남"

# ── 1. DLO 데이터 정제 (메모리) ─────────────────────────────────────
dlo_path = rf"C:\Users\user\Desktop\Study\data\AEC\{SITE}\Results\DLO_Results.xlsx"
df_dlo = pd.read_excel(dlo_path)
df_dlo = df_dlo[["PatientID", "PatientAge", "PatientSex", "TAMA"]].drop_duplicates(subset="PatientID", keep="first")
df_dlo = df_dlo[pd.to_numeric(df_dlo["TAMA"], errors="coerce").notna()]

# ── 2. DICOM 태그 추출 (메모리) ──────────────────────────────────────
root_dir = rf"D:\데이터서비스팀 영상제공\{SITE}\{SITE}_axial"
folders  = sorted([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))])
records  = []

for folder_name in tqdm(folders, desc=f"[{SITE}] DICOM", unit="폴더"):
    folder_path = os.path.join(root_dir, folder_name)
    dcm_files   = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".dcm")])

    if len(dcm_files) < 20:
        tqdm.write(f"[건너뜀] [{SITE}] {folder_name}: DCM 파일이 {len(dcm_files)}개 (20개 미만)")
        continue

    try:
        ds = pydicom.dcmread(os.path.join(folder_path, dcm_files[19]), stop_before_pixels=True)
        records.append({
            "PatientID": str(ds.get("PatientID", "")).strip(),
            "ManufacturerModelName": str(ds.get("ManufacturerModelName", "")).strip(),
        })
    except Exception as e:
        tqdm.write(f"[오류] [{SITE}] {folder_name}: {e}")

df_dicom = pd.DataFrame(records)

# ── 3. 병합 ──────────────────────────────────────────────────────────
df_dicom["PatientID"] = df_dicom["PatientID"].astype(str).str.strip()
df_dlo["PatientID"]   = df_dlo["PatientID"].astype(float).astype(int).astype(str)
df_merged = pd.merge(df_dicom, df_dlo, on="PatientID", how="inner")

# ── 4. AEC 128 포인트 추출 ───────────────────────────────────────────
aec_dir = Path(rf"C:\Users\user\Desktop\Study\data\AEC\{SITE}\AEC")
png_map = {p.stem: p for p in aec_dir.glob("*.png")}

aec_vectors = []
for pid in tqdm(df_merged["PatientID"], desc=f"[{SITE}] AEC 추출", unit="명"):
    if str(pid) in png_map:
        try:
            points = extract_aec(png_map[str(pid)])
            aec_vectors.append([round(v, 2) for v in points.tolist()])
        except Exception as e:
            tqdm.write(f"[오류] {pid}: {e}")
            aec_vectors.append(None)
    else:
        aec_vectors.append(None)

df_merged["AEC"] = aec_vectors
df_merged = df_merged[df_merged["AEC"].notna()].reset_index(drop=True)

# ── 5. AEC feature 추출 ──────────────────────────────────────────────
feat_rows = []
for aec in tqdm(df_merged["AEC"], desc=f"[{SITE}] AEC feature", unit="명"):
    feat_rows.append(extract_aec_features(np.array(aec, dtype=float)))

df_feats  = pd.DataFrame(feat_rows)
df_merged = pd.concat([df_merged.reset_index(drop=True), df_feats], axis=1)

# ── 6. 최종 저장 ─────────────────────────────────────────────────────
final_path = rf"C:\Users\user\Desktop\Study\data\AEC\{SITE}\Results\{SITE}_final.xlsx"
df_merged.to_excel(final_path, index=False)
print(f"[{SITE}] 저장 완료: {final_path} ({len(df_merged)}행, feature {len(df_feats.columns)}개)")

