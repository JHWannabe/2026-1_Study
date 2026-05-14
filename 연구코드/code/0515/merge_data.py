import pandas as pd
import numpy as np

SITE = "강남"
BASE = rf"C:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data\{SITE}"
meta_path   = rf"{BASE}\{SITE}_metadata.xlsx"
aecraw_path = rf"{BASE}\{SITE}_aec_raw.xlsx"
out_path    = rf"{BASE}\{SITE}_merged_features.xlsx"

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
meta   = pd.read_excel(meta_path)
aecraw = pd.read_excel(aecraw_path)

slice_cols = [c for c in aecraw.columns if c.startswith("slice_")]

# ── inner join + metadata 결측치 제거 ────────────────────────────────────────
merged     = pd.merge(meta, aecraw[["PatientID"]], on="PatientID", how="inner")
meta_clean = merged.dropna()

# AEC 값 변화 없는 환자 제거 (std == 0)
aec_std = aecraw.set_index("PatientID")[slice_cols].std(axis=1)
no_var_ids = set(aec_std[aec_std == 0].index)
meta_clean = meta_clean[~meta_clean["PatientID"].isin(no_var_ids)]

# ManufacturerModelName 비율 5% 미만 제거
model_ratio = meta_clean["ManufacturerModelName"].value_counts(normalize=True)
rare_models = set(model_ratio[model_ratio < 0.05].index)
meta_clean = meta_clean[~meta_clean["ManufacturerModelName"].isin(rare_models)]

# kVp == 100 만 유지
meta_clean = meta_clean[meta_clean["kVp"] == 100]

valid_ids = set(meta_clean["PatientID"])

after_dropna = merged.dropna()
print(f"meta rows:            {len(meta)}")
print(f"aecraw rows:          {len(aecraw)}")
print(f"after meta dropna:    {len(after_dropna)}")
print(f"dropped (no var):     {len(no_var_ids & set(after_dropna['PatientID']))}")
print(f"dropped (rare model): {len(rare_models)} models → {len(after_dropna[~after_dropna['PatientID'].isin(no_var_ids) & after_dropna['ManufacturerModelName'].isin(rare_models)])}")
print(f"dropped (kVp!=100):   {(meta_clean['kVp'] != 100).sum() if 'kVp' in meta_clean else 'N/A'}")
print(f"final:                {len(meta_clean)}")
print(f"rare models removed:  {rare_models}")

# ── Sheet1: metadata ──────────────────────────────────────────────────────────
sheet_meta = meta_clean.reset_index(drop=True)

# ── Sheet2: aec_raw ───────────────────────────────────────────────────────────
sheet_aecraw = aecraw[aecraw["PatientID"].isin(valid_ids)].copy()
# valid_ids 순서를 meta_clean과 맞춤
sheet_aecraw = (
    meta_clean[["PatientID"]]
    .merge(sheet_aecraw, on="PatientID", how="left")
    .reset_index(drop=True)
)

# ── Sheet3~5: aec 보간 (64 / 128 / 256 포인트) ───────────────────────────────
def make_interp_sheet(aecraw_df, pid_list, slice_cols, n_points):
    x_new = np.linspace(0, 1, n_points)
    cols = ["PatientID"] + [f"z_{i+1}" for i in range(n_points)]
    rows = []
    for pid in pid_list:
        vals = aecraw_df[aecraw_df["PatientID"] == pid].iloc[0][slice_cols].dropna().values.astype(float)
        x_orig = np.linspace(0, 1, len(vals))
        rows.append([pid] + np.interp(x_new, x_orig, vals).tolist())
    return pd.DataFrame(rows, columns=cols)

pid_list = meta_clean["PatientID"].tolist()
sheet_aec64  = make_interp_sheet(aecraw, pid_list, slice_cols, 64)
sheet_aec128 = make_interp_sheet(aecraw, pid_list, slice_cols, 128)
sheet_aec256 = make_interp_sheet(aecraw, pid_list, slice_cols, 256)

def round_floats(df, decimals=2):
    float_cols = df.select_dtypes(include="float").columns
    df[float_cols] = df[float_cols].round(decimals)
    return df

# ── 저장 ─────────────────────────────────────────────────────────────────────
with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    round_floats(sheet_meta).to_excel(writer,   sheet_name="metadata", index=False)
    round_floats(sheet_aecraw).to_excel(writer, sheet_name="aec_raw",  index=False)
    round_floats(sheet_aec64).to_excel(writer,  sheet_name="aec_64",   index=False)
    round_floats(sheet_aec128).to_excel(writer, sheet_name="aec_128",  index=False)
    round_floats(sheet_aec256).to_excel(writer, sheet_name="aec_256",  index=False)

print(f"\nSaved: {out_path}")
print(f"  metadata : {sheet_meta.shape}")
print(f"  aec_raw  : {sheet_aecraw.shape}")
print(f"  aec_64   : {sheet_aec64.shape}")
print(f"  aec_128  : {sheet_aec128.shape}")
print(f"  aec_256  : {sheet_aec256.shape}")
