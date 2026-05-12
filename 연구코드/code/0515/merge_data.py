import pandas as pd
SITE = "신촌"
aec_path   = f"연구코드/data/{SITE}/{SITE}_aec_256_features.xlsx"
meta_path  = f"연구코드/data/{SITE}/{SITE}_metadata.xlsx"
aec256_path = f"연구코드/data/{SITE}/{SITE}_aec_256.xlsx"
out_path   = f"연구코드/data/{SITE}/{SITE}_merged_features.xlsx"

# ── 데이터 로드 ──────────────────────────────────────────────────────────────
aec   = pd.read_excel(aec_path)
meta  = pd.read_excel(meta_path)
aec256 = pd.read_excel(aec256_path)

# ── Sheet1: metadata + aec_256_features inner join, 결측치 제거 ───────────────
merged       = pd.merge(meta, aec, on="PatientID", how="inner")
merged_clean = merged.dropna()

print(f"AEC features rows: {len(aec)}")
print(f"Metadata rows:     {len(meta)}")
print(f"After join:        {len(merged)}")
print(f"After dropna:      {len(merged_clean)}")
print(f"Columns (Sheet1):  {merged_clean.shape[1]}")

# ── Sheet2: (결측치 없는 metadata) + aec_256 raw signal inner join ────────────
meta_cols  = list(meta.columns)          # PatientID ~ SMI (15 cols)
meta_clean = merged_clean[meta_cols]     # dropna 필터 이미 적용된 metadata

merged_aec256 = pd.merge(meta_clean, aec256, on="PatientID", how="inner")

print(f"\naec_256 rows:      {len(aec256)}")
print(f"After join:        {len(merged_aec256)}")
print(f"Columns (Sheet2):  {merged_aec256.shape[1]}")

# ── 저장: 두 시트를 한 번에 ────────────────────────────────────────────────────
with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    merged_clean.to_excel(writer,  sheet_name="meta_aec_features", index=False)
    merged_aec256.to_excel(writer, sheet_name="meta_aec256",       index=False)

print(f"\nSaved '{out_path}'")
print("  Sheet1: meta_aec_features")
print("  Sheet2: meta_aec256")
