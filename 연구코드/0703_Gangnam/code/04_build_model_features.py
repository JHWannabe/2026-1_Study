"""
강남_liver_merged_features.xlsx의 metadata(임상변수) + aec_128(mA 곡선)에서
모델 입력용 피처 테이블을 만든다.

임상변수: PatientAge, PatientSex, Height, Weight (고정)
AEC 파생변수 (aec_pointwise_derivative_test.png 근거, 128포인트 = pubis(1)->liver upper(128) 정규화 곡선):
  - mean_mA: 곡선 전체 평균 높이. auc_mA와 상관계수 0.99999로 사실상 동일 정보라 auc는 별도로 뽑지 않음
  - slope_48_55: aec_62~aec_71 구간 평균 1차 차분 (남성 유의구간, FDR p<0.05, d=0.41)
  - slope_80_90: aec_103~aec_115 구간 평균 1차 차분 (남 103~113 / 여 104~115 유의구간을 통합)
  - slope_94_97: aec_120~aec_124 구간 평균 1차 차분 (남성 유의구간, d=-0.30)

타깃: Output (0=정상, 1=저SMI/근감소증) - 강남_aec_128.xlsx의 M_normal/M_low_SMI/F_normal/F_low_SMI
시트에 이미 그룹 라벨로 들어있던 값을 PatientID로 조인해서 그대로 가져옴 (임의 threshold 재정의 안 함).
"""

import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXCEL_DIR = os.path.join(BASE_DIR, "..", "excel")

INPUT_FILE = os.path.join(EXCEL_DIR, "강남_liver_merged_features.xlsx")
LABEL_FILE = os.path.join(EXCEL_DIR, "강남_aec_128.xlsx")
OUTPUT_FILE = os.path.join(EXCEL_DIR, "강남_model_features.xlsx")

SEGMENTS = {
    "slope_48_55": (62, 71),
    "slope_80_90": (103, 115),
    "slope_94_97": (120, 124),
}


def segment_slope(X: np.ndarray, start: int, end: int) -> np.ndarray:
    """aec index(1-based) start~end 구간의 평균 1차 차분(기울기)"""
    window = X[:, start - 1:end]
    return np.diff(window, axis=1).mean(axis=1)


def load_output_labels() -> pd.DataFrame:
    xls = pd.ExcelFile(LABEL_FILE)
    sheets = ["M_normal", "M_low_SMI", "F_normal", "F_low_SMI"]
    parts = [pd.read_excel(xls, s, usecols=["PatientID", "Output"]) for s in sheets]
    return pd.concat(parts, ignore_index=True)


def main():
    xls = pd.ExcelFile(INPUT_FILE)
    meta = pd.read_excel(xls, "metadata")
    aec = pd.read_excel(xls, "aec_128")
    aec_cols = [c for c in aec.columns if c.startswith("aec_")]

    df = meta.merge(aec[["PatientID"] + aec_cols], on="PatientID", how="inner")
    df = df.merge(load_output_labels(), on="PatientID", how="left")
    assert df["Output"].isna().sum() == 0, "Output 라벨 조인 실패한 환자 존재"

    X = df[aec_cols].to_numpy(dtype=float)

    out = df[["PatientID", "PatientAge", "PatientSex", "Height", "Weight", "SMI", "Output"]].copy()
    out["mean_mA"] = X.mean(axis=1)
    for name, (start, end) in SEGMENTS.items():
        out[name] = segment_slope(X, start, end)

    out.to_excel(OUTPUT_FILE, index=False)
    print(f"저장 완료: {OUTPUT_FILE} ({out.shape[0]}행 x {out.shape[1]}열)")
    print(out.head())


if __name__ == "__main__":
    main()
