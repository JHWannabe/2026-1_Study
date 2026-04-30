import numpy as np
import pandas as pd
from pathlib import Path
from scipy import interpolate

# 데이터 파일 경로
data_path = r'연구코드\data\신촌_merged_features.xlsx'

# 시트 로드
metadata_bmi = pd.read_excel(data_path, sheet_name='metadata-bmi_add')
# PatientID, SMI 칼럼만 선택, 결측치 제거
metadata_bmi = metadata_bmi[['PatientID', 'SMI', 'n_slices', 'z_range_mm']].dropna()

aec_raw = pd.read_excel(data_path, sheet_name='aec-raw')

N_TARGET = 256

def interpolate_aec(aec_df, n_target=N_TARGET):
    """각 PatientID의 AEC 값을 n_target 포인트로 선형 보간"""
    aec_cols = [c for c in aec_df.columns if c != 'PatientID']
    rows = []
    for _, row in aec_df.iterrows():
        values = row[aec_cols].dropna().values.astype(float)
        if len(values) < 2:
            continue
        x_orig = np.linspace(0, 1, len(values))
        x_new  = np.linspace(0, 1, n_target)
        resampled = np.round(interpolate.interp1d(x_orig, values, kind='linear')(x_new), 2)
        rows.append([row['PatientID']] + resampled.tolist())
    cols = ['PatientID'] + [f'aec_{i}' for i in range(n_target)]
    return pd.DataFrame(rows, columns=cols)

aec_interp = interpolate_aec(aec_raw)

# patientID 기준으로 merge
merged_df = pd.merge(metadata_bmi, aec_interp, on='PatientID', how='inner').drop(columns=['n_slices', 'z_range_mm'])

print("메타데이터 (metadata-bmi_add) 로드 완료:")
print(f"  - 행: {len(metadata_bmi)}, 열: {len(metadata_bmi.columns)}")
print(f"  - 칼럼: {metadata_bmi.columns.tolist()}\n")

print("AEC 데이터 (aec-raw) 로드 완료:")
print(f"  - 행: {len(aec_raw)}, 열: {len(aec_raw.columns)}")

print(f"\nAEC 보간 결과 ({N_TARGET}포인트):")
print(f"  - 행: {len(aec_interp)}, 열: {len(aec_interp.columns)}\n")

print("병합 결과:")
print(f"  - 행: {len(merged_df)}, 열: {len(merged_df.columns)}")
print(f"\n첫 5개 행:")
print(merged_df.head())

# 보간 결과를 새 시트로 저장
with pd.ExcelWriter(data_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    aec_interp.to_excel(writer, sheet_name='aec-interp', index=False)
    merged_df.to_excel(writer, sheet_name='merged', index=False)

print(f"\n저장 완료: '{data_path}'")
print(f"  - 시트 'aec-interp': AEC 보간 결과 ({len(aec_interp)}행 x {len(aec_interp.columns)}열)")
print(f"  - 시트 'merged': 병합 결과 ({len(merged_df)}행 x {len(merged_df.columns)}열)")
