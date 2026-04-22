import pandas as pd
import numpy as np  # 에러 방지를 위해 numpy 추가

# 1. 파일 불러오기
df = pd.read_excel(r'C:\Users\jhjun\Downloads\bigdata.xlsx', sheet_name='신장체중(예진)_전체')
metadata_df = pd.read_excel(r'C:\Users\jhjun\Downloads\matching.xlsx', sheet_name='metadata-value')

# 2. 매핑 테이블 생성 (마지막 값 기준)
height_lookup = df.groupby('연구등록번호')['신장'].last()
weight_lookup = df.groupby('연구등록번호')['체중'].last()

# 3. 데이터 매칭
metadata_df['신장'] = metadata_df['PatientID'].map(height_lookup)
metadata_df['체중'] = metadata_df['PatientID'].map(weight_lookup)

# 4. BMI 재계산 (에러 방지 로직 추가)
# 신장이 0인 경우를 NaN으로 바꾸어 '0으로 나누기 에러'를 방지합니다.
# 또한 데이터가 없는(None) 행은 자동으로 계산 결과가 NaN이 되어 에러 없이 넘어갑니다.
temp_height = pd.to_numeric(metadata_df['신장'], errors='coerce').replace(0, np.nan)
temp_weight = pd.to_numeric(metadata_df['체중'], errors='coerce')

metadata_df['BMI'] = (temp_weight / ((temp_height / 100) ** 2)).round(2)

# 5. 결과 확인 및 저장
matched_count = metadata_df['신장'].notna().sum()
matched_weight_count = metadata_df['체중'].notna().sum()
bmi_count = metadata_df['BMI'].notna().sum()

print(f"매칭된 신장 데이터: {matched_count}건")
print(f"매칭된 체중 데이터: {matched_weight_count}건")
print(f"계산된 BMI 데이터: {bmi_count}건 (0 또는 누락 데이터 제외)")

# 결과 저장
metadata_df.to_excel(r'C:\Users\jhjun\Downloads\matching_updated.xlsx', index=False)
print("저장이 완료되었습니다.")