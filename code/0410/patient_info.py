import os
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import cv2
from pathlib import Path

# ─── 경로 설정 ────────────────────────────────────────────────────────────────
SITE            = "신촌"
IMAGE_DIR       = Path(rf"C:\Users\user\Desktop\Study\data\AEC\{SITE}\raw\Image")
DLO_EXCEL_PATH  = Path(rf"C:\Users\user\Desktop\Study\data\AEC\{SITE}\raw\{SITE}_DLO_Results.xlsx")
INFO_EXCEL_PATH = Path(rf"C:\Users\user\Desktop\Study\data\AEC\{SITE}\raw\{SITE}_patient_info.xlsx")

AEC_POINTS = 128


# ─── AEC 추출 함수 ────────────────────────────────────────────────────────────
def extract_aec(image_path: Path, n_points: int = AEC_POINTS) -> np.ndarray:
    raw = np.fromfile(str(image_path), dtype=np.uint8)
    if raw.size == 0:
        raise ValueError(f"Empty file: {image_path}")
    image_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Decode failed: {image_path}")

    hsv        = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 80, 80],   dtype=np.uint8)
    upper_blue = np.array([130, 255, 255], dtype=np.uint8)
    blue_mask  = cv2.inRange(hsv, lower_blue, upper_blue)
    kernel     = np.ones((3, 3), np.uint8)
    blue_mask  = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

    y_idx, x_idx = np.where(blue_mask > 0)
    if len(x_idx) == 0:
        raise ValueError(f"Blue line not found: {image_path}")

    unique_x = np.unique(x_idx)
    mean_y   = np.array([y_idx[x_idx == x].mean() for x in unique_x], dtype=float)
    target_x = np.linspace(unique_x.min(), unique_x.max(), n_points)
    interp_y = np.interp(target_x, unique_x, mean_y)

    return image_bgr.shape[0] - interp_y



# ─── 1. 기존 patient_info 로드 ────────────────────────────────────────────────
df_info = pd.read_excel(INFO_EXCEL_PATH, dtype={"PatientID": str})
print(f"patient_info 행 수: {len(df_info)}")


# ─── 2. TAMA 집계 ─────────────────────────────────────────────────────────────
df_dlo = pd.read_excel(DLO_EXCEL_PATH, dtype={"PatientID": str})
df_dlo["TAMA"] = (
    df_dlo["TAMA"].astype(str).str.strip()
    .replace({"NO_LINK": np.nan, "N/A": np.nan, "na": np.nan,
              "n/a": np.nan, "": np.nan, "nan": np.nan})
)
df_dlo["TAMA"] = pd.to_numeric(df_dlo["TAMA"], errors="coerce")

tama_dict = (
    df_dlo.groupby("PatientID")["TAMA"]
    .first()
    .dropna()
    .to_dict()
)
print(f"TAMA 유효 PatientID 수: {len(tama_dict)}")


# ─── 3. AEC 추출 → 128 포인트 → 하나의 컬럼에 저장 ──────────────────────────
failures    = []
feature_rows = []

for image_path in sorted(IMAGE_DIR.glob("*.png")):
    pid = image_path.stem
    try:
        aec = extract_aec(image_path)

        feature_rows.append({
            "PatientID":   pid,
            "aec_feature": aec.tolist(),        # 128 포인트 리스트로 저장
        })
        print(f"  [완료] {pid}  →  vector shape: {aec.shape}")

    except Exception as e:
        failures.append(f"{pid}: {e}")
        print(f"  [실패] {pid} → {e}")

print(f"\n추출 성공: {len(feature_rows)}건 / 실패: {len(failures)}건")

df_features = pd.DataFrame(feature_rows)


# ─── 4. TAMA 컬럼 추가 ───────────────────────────────────────────────────────
df_features["TAMA"] = df_features["PatientID"].map(tama_dict)


# ─── 5. patient_info에 merge 후 저장 ─────────────────────────────────────────
df_merged = df_info.merge(df_features, on="PatientID", how="left")

print(f"\n최종 컬럼: {list(df_merged.columns)}")
print(f"최종 행 수: {len(df_merged)}")

# merge 후 TAMA 컬럼명 처리 (중복 시 _x/_y 접미사 발생 대응)
if "TAMA" not in df_merged.columns:
    if "TAMA_y" in df_merged.columns:
        # df_info에 TAMA가 이미 있어서 분리된 경우 → df_features의 TAMA 사용
        df_merged["TAMA"] = df_merged["TAMA_y"].fillna(df_merged.get("TAMA_x"))
        df_merged.drop(columns=["TAMA_x", "TAMA_y"], errors="ignore", inplace=True)
    else:
        # df_features에 TAMA가 없는 경우 (추출 성공 0건 등) → tama_dict로 직접 매핑
        df_merged["TAMA"] = df_merged["PatientID"].map(tama_dict)

print(f"TAMA 채워진 행: {df_merged['TAMA'].notna().sum()}")

df_merged.to_excel(INFO_EXCEL_PATH, index=False)
print(f"\n✅ 저장 완료: {INFO_EXCEL_PATH}")