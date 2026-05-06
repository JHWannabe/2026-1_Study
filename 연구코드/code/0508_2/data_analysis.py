import pandas as pd
import numpy as np

# Load the specified sheet from the Excel file
file_path = r'연구코드\data\강남_최종_정리본.xlsx'
sheet_name = 'aec_interpolation_final'

try:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    # 1. 성비 분석
    sex_counts = df['PatientSex'].value_counts()
    sex_ratio = df['PatientSex'].value_counts(normalize=True) * 100
    
    # 2. 성별에 따른 SMI, TAMA 분포 통계
    stats_by_sex = df.groupby('PatientSex')[['SMI', 'TAMA']].agg(
        ['mean', 'std', lambda x: x.quantile(0.5), lambda x: x.quantile(0.25)]
    ).rename(columns={'<lambda_0>': 'median', '<lambda_1>': 'p25'})
    
    # 3. 하위 25% 여부 판단 및 O/X 표시 추가
    # 성별마다 기준치가 다르므로 성별 group별로 계산하여 병합
    def mark_bottom_25(group):
        smi_q25 = group['SMI'].quantile(0.25)
        tama_q25 = group['TAMA'].quantile(0.25)
        
        group['SMI_Bottom_25'] = group['SMI'].apply(lambda x: 'O' if x <= smi_q25 else 'X')
        group['TAMA_Bottom_25'] = group['TAMA'].apply(lambda x: 'O' if x <= tama_q25 else 'X')
        group['Both_Bottom_25'] = ((group['SMI'] <= smi_q25) & (group['TAMA'] <= tama_q25)).map({True: 'O', False: 'X'})
        return group

    df_analyzed = df.groupby('PatientSex', group_keys=False).apply(mark_bottom_25)

    # 4. 요약 데이터 생성 (성별 하위 25% 중복 비율 등)
    summary_list = []
    for sex in df_analyzed['PatientSex'].unique():
        sex_df = df_analyzed[df_analyzed['PatientSex'] == sex]
        total = len(sex_df)
        both_o = (sex_df['Both_Bottom_25'] == 'O').sum()
        
        summary_list.append({
            'Sex': sex,
            'SMI_p25_Threshold': df[df['PatientSex']==sex]['SMI'].quantile(0.25),
            'TAMA_p25_Threshold': df[df['PatientSex']==sex]['TAMA'].quantile(0.25),
            'Both_Bottom_Count': both_o,
            'Total_Count': total,
            'Percentage': (both_o / total) * 100
        })
    comparison_df = pd.DataFrame(summary_list)

    # 결과 저장
    output_file = '강남_최종_정리본_분석결과.xlsx'
    with pd.ExcelWriter(output_file) as writer:
        # 하위 25% O/X 표시가 포함된 전체 데이터
        df_analyzed.to_excel(writer, sheet_name='Patient_Analysis', index=False)
        # 통계 요약 시트들
        sex_ratio.to_frame(name='Percentage').to_excel(writer, sheet_name='Sex_Ratio')
        stats_by_sex.to_excel(writer, sheet_name='SMI_TAMA_Stats')
        comparison_df.to_excel(writer, sheet_name='Bottom_25_Summary', index=False)

    print("분석 및 파일 생성이 완료되었습니다.")
    print(df_analyzed[['PatientID', 'PatientSex', 'SMI', 'SMI_Bottom_25', 'TAMA', 'TAMA_Bottom_25', 'Both_Bottom_25']].head())

except Exception as e:
    print(f"Error: {e}")