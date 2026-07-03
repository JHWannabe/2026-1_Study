import pandas as pd

META_PATH = r"C:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data\신촌\metadata\신촌_DLO_Results_SMI_kVp100.xlsx"
AEC_PATH  = r"C:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data\신촌\aec\신촌_aec_cropped.xlsx"

_xl = pd.ExcelFile(META_PATH, engine="openpyxl")
_sheet = str(_xl.sheet_names[0])
meta = pd.read_excel(_xl, sheet_name=_sheet, engine="openpyxl")
aec  = pd.read_excel(AEC_PATH,  engine="openpyxl", usecols=["PatientID", "manufacturer_model"])

# SOMATOM Definition으로 시작하는 행 대상
somatom_mask = meta["Manufacturer"].astype(str).str.startswith("SOMATOM Definition")
target_ids   = meta.loc[somatom_mask, "PatientID"]

# AEC에서 manufacturer_model 조회 (PatientID → manufacturer_model 매핑)
id_to_model = aec.set_index("PatientID")["manufacturer_model"]

# 매핑 후 덮어쓰기 (AEC에 없는 PatientID는 원본 유지)
mapped = target_ids.map(id_to_model)
not_found = mapped.isna().sum()
if not_found:
    print(f"[경고] AEC에서 찾지 못한 PatientID {not_found}개 - 해당 행은 원본 유지")
    print(meta.loc[somatom_mask][mapped.isna()][["PatientID", "Manufacturer"]])

meta.loc[somatom_mask & mapped.notna(), "Manufacturer"] = mapped[mapped.notna()].values

updated = (somatom_mask & mapped.notna()).sum()
print(f"업데이트 완료: {updated}개 행의 Manufacturer 값을 manufacturer_model로 교체")

meta.to_excel(META_PATH, index=False, sheet_name=_sheet, engine="openpyxl")
print("저장 완료:", META_PATH)
