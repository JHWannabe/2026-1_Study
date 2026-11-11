"""5-x f_u 시트(질환별 진단 기록) 환자만 대상으로, 5년(1826일) 위험집단 기준 filtered 형식 xlsx를 질환별로 만든다.
출력: data/예후데이터/5y_filtered/{질환}_5y_filtered.xlsx (patient_metadata 시트 + Read_Me)
위험집단: 해당 질환 기저 과거력(history_*)이 없는 환자. onset_within_5y=1이면 5년 이내, 0이면 5년 이후 발병."""
import os
import pandas as pd

D = r"C:\Users\jhjun\Desktop\2026-1_Study\연구코드\data\예후데이터"
X = pd.ExcelFile(os.path.join(D, "longitudinal_cohort_cleaned_before_roster.xlsx"))
M = X.parse("Patient_Master")
E = X.parse("Events_Long")
DAYS = 5 * 365 + 1  # 1826일, 기존 disease_onset_5y와 동일
out = os.path.join(D, "5y_filtered")
os.makedirs(out, exist_ok=True)

for o in ["dm", "htn", "dl", "osteoporosis", "mi", "stroke"]:
    ev = E[E.outcome == o].groupby("patient_id").event_days_from_ct.min()  # 환자별 최초 진단
    m = M[M.patient_id.isin(ev.index)].copy()
    n_all = len(m)
    m = m[m[f"history_{o}"].fillna(0) == 0]  # 과거력 있는 환자는 위험집단 아님
    df = pd.DataFrame({
        "PatientID": m.patient_id, "PatientSex": m.sex, "PatientAge": m.age_at_ct,
        "Height": m.height_cm_analysis, "Weight": m.weight_kg_analysis, "BMI": m.bmi_analysis,
        "hw_implausible": 0,  # ponytail: 분석용 키/체중은 이미 타당범위만 보존됨 -> 결측은 NaN, 비현실값 없음
        "event_days_from_ct": m.patient_id.map(ev),
    })
    df["onset_within_5y"] = (df.event_days_from_ct <= DAYS).astype(int)
    note = pd.DataFrame({"항목": ["source", "대상", "위험집단", "onset_within_5y", "N(시트 환자)", "N(과거력 제외 후)"],
                         "내용": ["longitudinal_cohort_cleaned_before_roster.xlsx", f"{o} f_u 시트 진단 환자만",
                                  f"history_{o}=0", f"event_days_from_ct<={DAYS}이면 1, 아니면 5년 이후 발병(0)",
                                  n_all, len(df)]})
    with pd.ExcelWriter(os.path.join(out, f"{o}_5y_filtered.xlsx")) as w:
        df.to_excel(w, "patient_metadata", index=False)
        note.to_excel(w, "Read_Me", index=False)
    print(o, n_all, "->", len(df), "within5y", df.onset_within_5y.sum(), "after5y", (1 - df.onset_within_5y).sum())
