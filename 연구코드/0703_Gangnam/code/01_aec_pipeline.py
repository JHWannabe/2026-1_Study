"""
AEC(aec_1~aec_128) 연결성 데이터 - SMI 저하군 vs 정상군 비교 분석 파이프라인

입력: 강남_aec_128.xlsx (aec_128 시트는 성별을 합친 것이라 confounding 위험 -> 제외)
사용 시트: M_normal, M_low_SMI, F_normal, F_low_SMI (성별로 층화된 데이터)

단계:
  1. 성별(M/F)로 층화하여 normal vs low_SMI 비교
  2. 비정규분포 데이터이므로 Mann-Whitney U (+ 참고용 Welch's t-test) 사용
  3. 다중비교 보정: FDR(Benjamini-Hochberg) 적용 (aec 변수 간 상관이 매우 높아 Bonferroni는 과도하게 보수적)
  4. Effect size(Cohen's d) 병기
  5. 연속형 SMI와의 상관(Spearman) 병기 - 이분화로 인한 정보 손실 보완
  6. 남/여 결과를 Fisher's method로 메타분석하여 통합 p-value 산출 후 재보정

출력: aec_analysis_results.xlsx (변수별 결과표), 콘솔에 요약 통계 출력
"""

import os

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXCEL_DIR = os.path.join(BASE_DIR, "..", "excel")

INPUT_FILE = os.path.join(EXCEL_DIR, "강남_aec_128.xlsx")
OUTPUT_FILE = os.path.join(EXCEL_DIR, "aec_analysis_results.xlsx")


def cohend(a: pd.Series, b: pd.Series) -> float:
    na, nb = len(a), len(b)
    pooled_sd = np.sqrt(((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2))
    return (a.mean() - b.mean()) / pooled_sd


def compare_group(normal: pd.DataFrame, low_smi: pd.DataFrame, aec_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in aec_cols:
        a = normal[col].dropna()
        b = low_smi[col].dropna()

        _, p_mw = stats.mannwhitneyu(a, b, alternative="two-sided")
        _, p_t = stats.ttest_ind(a, b, equal_var=False)  # Welch, 참고용
        d = cohend(a, b)

        smi_all = pd.concat([normal["SMI"], low_smi["SMI"]])
        aec_all = pd.concat([a, b])
        rho, p_corr = stats.spearmanr(aec_all, smi_all)

        rows.append(
            {
                "aec_var": col,
                "p_mannwhitney": p_mw,
                "p_welch_t": p_t,
                "cohend": d,
                "spearman_rho_vs_SMI": rho,
                "spearman_p_vs_SMI": p_corr,
            }
        )
    return pd.DataFrame(rows)


def add_fdr(df: pd.DataFrame, pcol: str, out_col: str) -> pd.DataFrame:
    df[out_col] = multipletests(df[pcol], method="fdr_bh")[1]
    return df


def fisher_combine(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Fisher's method로 두 독립 검정(남/여)의 p-value를 통합"""
    chi2 = -2 * (np.log(p1) + np.log(p2))
    return 1 - stats.chi2.cdf(chi2, df=4)


def main():
    xls = pd.ExcelFile(INPUT_FILE)
    m_normal = pd.read_excel(xls, "M_normal")
    m_low = pd.read_excel(xls, "M_low_SMI")
    f_normal = pd.read_excel(xls, "F_normal")
    f_low = pd.read_excel(xls, "F_low_SMI")

    aec_cols = [c for c in m_normal.columns if c.startswith("aec_")]
    print(f"aec 변수 개수: {len(aec_cols)}")
    print(f"M_normal n={len(m_normal)}, M_low_SMI n={len(m_low)}")
    print(f"F_normal n={len(f_normal)}, F_low_SMI n={len(f_low)}")

    male = compare_group(m_normal, m_low, aec_cols)
    female = compare_group(f_normal, f_low, aec_cols)

    male = add_fdr(male, "p_mannwhitney", "p_fdr")
    female = add_fdr(female, "p_mannwhitney", "p_fdr")

    combined = pd.DataFrame({"aec_var": aec_cols})
    combined["p_male_raw"] = male["p_mannwhitney"].to_numpy()
    combined["p_male_fdr"] = male["p_fdr"].to_numpy()
    combined["cohend_male"] = male["cohend"].to_numpy()
    combined["p_female_raw"] = female["p_mannwhitney"].to_numpy()
    combined["p_female_fdr"] = female["p_fdr"].to_numpy()
    combined["cohend_female"] = female["cohend"].to_numpy()
    combined["p_fisher_combined"] = fisher_combine(
        combined["p_male_raw"].to_numpy(), combined["p_female_raw"].to_numpy()
    )
    combined = add_fdr(combined, "p_fisher_combined", "p_fisher_combined_fdr")

    with pd.ExcelWriter(OUTPUT_FILE) as writer:
        male.to_excel(writer, sheet_name="Male_normal_vs_lowSMI", index=False)
        female.to_excel(writer, sheet_name="Female_normal_vs_lowSMI", index=False)
        combined.to_excel(writer, sheet_name="Combined_Fisher", index=False)

    print()
    print("=== 요약 ===")
    print(f"남성: raw p<0.05 = {(male['p_mannwhitney'] < 0.05).sum()}/{len(aec_cols)}, "
          f"FDR<0.05 = {(male['p_fdr'] < 0.05).sum()}/{len(aec_cols)}")
    print(f"여성: raw p<0.05 = {(female['p_mannwhitney'] < 0.05).sum()}/{len(aec_cols)}, "
          f"FDR<0.05 = {(female['p_fdr'] < 0.05).sum()}/{len(aec_cols)}")
    print(f"통합(Fisher): FDR<0.05 = {(combined['p_fisher_combined_fdr'] < 0.05).sum()}/{len(aec_cols)}")
    print()
    print(f"결과 저장 완료: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
