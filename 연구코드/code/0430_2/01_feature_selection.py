# -*- coding: utf-8 -*-
"""
AEC Feature Candidate Selection

단일 기준: 각 AEC 피처와 SMI 간 Pearson |r| >= CORR_WITH_Y 인 피처만 후보로 선정.
저장: results/feature_selection/feature_selection_summary.xlsx
      sheet "corr_with_smi"    — 전체 피처의 |r| 순위표
      sheet "candidate_features" — 선택된 후보 피처 목록
"""

import warnings
warnings.filterwarnings("ignore")

import os, subprocess, tempfile, shutil
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr

# ── 경로 & 상수 ──────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent.parent
DATA_DIR    = SCRIPT_DIR.parent / "data"
RESULT_DIR  = SCRIPT_DIR.parent / "results" / "feature_selection"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# SMI와의 Pearson |r| 이 값 이상인 피처만 후보로 선정
CORR_WITH_Y = 0.10


# ── 데이터 로드 ───────────────────────────────────────────────────────────────

def _load_xlsx(xlsx_path: Path):
    """OneDrive 잠금 우회: PowerShell로 임시 폴더에 복사 후 읽기."""
    tmp = Path(tempfile.gettempdir()) / f"aec_temp_{xlsx_path.stem}.xlsx"
    subprocess.run(
        ["powershell", "-Command",
         f'Copy-Item -Path "{xlsx_path}" -Destination "{tmp}" -Force'],
        capture_output=True,
    )
    feat_df = pd.read_excel(str(tmp), sheet_name="features")
    meta_df = pd.read_excel(str(tmp), sheet_name="metadata-bmi_add")
    return feat_df, meta_df


def _prepare(feat_df: pd.DataFrame, meta_df: pd.DataFrame):
    meta_df = meta_df.copy()

    # SMI 컬럼명 탐색 (대소문자·공백 변형 허용)
    smi_col = next((c for c in meta_df.columns if c.strip().upper() == "SMI"), None)
    if smi_col is None:
        print(f"  [경고] metadata 컬럼 목록: {meta_df.columns.tolist()}")
        raise KeyError("'SMI' 컬럼을 metadata 시트에서 찾을 수 없습니다.")
    if smi_col != "SMI":
        print(f"  [info] SMI 컬럼명 '{smi_col}' → 'SMI'로 통일")
        meta_df = meta_df.rename(columns={smi_col: "SMI"})

    meta_df["SMI"] = pd.to_numeric(meta_df["SMI"], errors="coerce")
    df = feat_df.merge(
        meta_df.dropna(subset=["SMI"])[["PatientID", "SMI"]],
        on="PatientID", how="inner"
    )
    df = pd.concat([df.drop(columns=["PatientID", "SMI"]),
                    df["SMI"]], axis=1).dropna().reset_index(drop=True)
    return df.drop(columns=["SMI"]), df["SMI"].astype(float)


# ── 핵심: Pearson |r| 기반 단일 단계 선택 ────────────────────────────────────

def select_by_corr(X: pd.DataFrame, y: pd.Series, threshold=CORR_WITH_Y):
    """
    각 피처와 y(SMI) 간 Pearson |r| 계산 후 threshold 이상만 선택.
    Pearson r = Cov(X,y) / (σ_X · σ_y) — 선형 상관의 강도와 방향
    """
    corr_df = pd.DataFrame({
        "feature": X.columns,
        "abs_r":   [abs(pearsonr(X[c], y)[0]) for c in X.columns],
    }).sort_values("abs_r", ascending=False).reset_index(drop=True)

    candidates = corr_df.loc[corr_df["abs_r"] >= threshold, "feature"].tolist()
    print(f"  |r| ≥ {threshold}: {len(candidates)} / {len(X.columns)} 피처 선택")
    print(corr_df.head(15).to_string(index=False))
    return candidates, corr_df


# ── 파이프라인 실행 ───────────────────────────────────────────────────────────

def run_pipeline(feat_df, meta_df, label, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*55}\n  {label}\n{'='*55}")

    X, y = _prepare(feat_df, meta_df)
    print(f"  샘플 {len(X)}명 | 피처 {X.shape[1]}개 | SMI mean={y.mean():.2f}")

    candidates, corr_df = select_by_corr(X, y)

    with pd.ExcelWriter(out_dir / "feature_selection_summary.xlsx") as writer:
        corr_df.to_excel(writer, sheet_name="corr_with_smi", index=False)
        pd.DataFrame({"candidate_features": candidates}).to_excel(
            writer, sheet_name="candidate_features", index=False)

    print(f"  저장: {out_dir / 'feature_selection_summary.xlsx'}")
    print(f"  후보 피처 ({len(candidates)}): {sorted(candidates)}")
    return candidates


# ── 데이터 탐색 & 실행 ────────────────────────────────────────────────────────

all_xlsx = {f: DATA_DIR / f for f in os.listdir(DATA_DIR)
            if f.endswith("raw.xlsx") and "merged_features" in f}
gangnam_path = next((v for k, v in all_xlsx.items() if "강남" in k), None)

if not gangnam_path:
    raise FileNotFoundError(f"강남 merged_features xlsx 파일을 {DATA_DIR}에서 찾을 수 없습니다.")

feat_gn, meta_gn = _load_xlsx(gangnam_path)
results = {"gangnam": run_pipeline(feat_gn, meta_gn, "강남", RESULT_DIR / "gangnam")}

shutil.copy2(RESULT_DIR / "gangnam" / "feature_selection_summary.xlsx",
             RESULT_DIR / "feature_selection_summary.xlsx")
print(f"\n  primary source: 'gangnam' → {RESULT_DIR / 'feature_selection_summary.xlsx'}")

print(f"\n{'='*55}\nALL DONE")
for key, cands in results.items():
    print(f"  [{key}] {len(cands)}개: {sorted(cands)}")
print("="*55)
