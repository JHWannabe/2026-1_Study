"""
지금까지(01~09)는 전부 "AEC 곡선을 clinic 변수와 함께 예측 모델에 넣으면 도움이 되는가"라는
같은 축의 질문을, 정규화/구조를 바꿔가며 반복했지만 전부 clinic-only를 못 이겼다(0702 결론).
이 스크립트는 완전히 다른 방법론으로 접근한다: 회귀/신경망으로 clinic 영향을 "통계적으로 통제"하는
대신, **Propensity Score Matching**으로 clinic 변수(Age/Height/Weight)가 사실상 동일한
low-SMI - normal 짝을 구성해서, confounding을 애초에 구조적으로 제거한 뒤 AEC 곡선만 비교한다.

방법:
  1. 성별 층화(0701/0702와 동일하게 M/F 따로 - SMI cutoff 자체가 성별 특이적이므로).
  2. 성별 내에서 PatientAge, Height, Weight로 로지스틱 회귀 propensity score 적합,
     logit(ps)로 변환.
  3. Greedy 1:1 nearest-neighbor matching (caliper = 0.2 x SD(logit ps), Austin 2011 권장값,
     무작위 순서로 처리 - 순서에 의한 편향 방지).
  4. 매칭 전/후 covariate balance를 표준화평균차(SMD)로 확인(<0.1이면 balanced로 간주).
  5. 매칭된 pair 안에서만 AEC 곡선을 비교:
     - level(원곡선) pointwise paired Wilcoxon signed-rank + BH-FDR
     - derivative(1차 차분) pointwise paired Wilcoxon signed-rank + BH-FDR
       (02_aec_fda_pipeline.py의 pointwise derivative test와 동일한 통계, unpaired -> paired로만 교체)
     - mean_mA paired Wilcoxon
  6. 0701에서 찾은 유의구간(M: 62-71/103-113/120-124, F: 104-115)이 matched 표본에서도
     살아남는지 직접 대조. 부호는 02_aec_fda_pipeline.py와 동일하게 "normal - low_SMI"로
     통일한다(양수 = normal이 더 큼) - matched pair 쪽은 diff = control(normal) - case(low_SMI).

출력: 콘솔 요약, figures/matched_balance_love_plot.png,
      figures/matched_pointwise_level_test.png, figures/matched_pointwise_derivative_test.png,
      excel/matched_comparison_results.xlsx (Segment_comparison_vs_0701 시트 포함)
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXCEL_DIR = os.path.join(BASE_DIR, "..", "excel")
FIGURES_DIR = os.path.join(BASE_DIR, "..", "figures", "10")
os.makedirs(FIGURES_DIR, exist_ok=True)

MERGED_FILE = os.path.join(EXCEL_DIR, "강남_liver_merged_features.xlsx")
LABEL_FILE = os.path.join(EXCEL_DIR, "강남_aec_128.xlsx")
RESULTS_FILE = os.path.join(EXCEL_DIR, "matched_comparison_results.xlsx")

COVARIATES = ["PatientAge", "Height", "Weight"]
CALIPER_MULT = 0.2  # Austin (2011) 권장: 0.2 x SD(logit propensity score)
SEED = 42
RNG = np.random.default_rng(SEED)

# 0701 pointwise derivative FDR 검정에서 찾은 유의구간 (1-based, inclusive)
PRIOR_SEGMENTS = {
    "M": [(62, 71), (103, 113), (120, 124)],
    "F": [(104, 115)],
}

# 0701(02_aec_fda_pipeline.py)이 보고한 구간별 Cohen's d (부호: normal - low_SMI, 0701.md 표 기준).
# matched 표본에서 방향/크기가 얼마나 유지되는지 직접 대조하기 위한 참조값.
PRIOR_COHEND = {
    ("M", 62, 71): 0.41,
    ("M", 103, 113): -0.46,
    ("M", 120, 124): -0.30,
    ("F", 104, 115): -0.22,
}


def load_data():
    xls = pd.ExcelFile(MERGED_FILE)
    meta = pd.read_excel(xls, "metadata")
    aec = pd.read_excel(xls, "aec_128")
    aec_cols = [c for c in aec.columns if c.startswith("aec_")]
    df = meta.merge(aec[["PatientID"] + aec_cols], on="PatientID", how="inner")

    label_xls = pd.ExcelFile(LABEL_FILE)
    sheets = ["M_normal", "M_low_SMI", "F_normal", "F_low_SMI"]
    labels = pd.concat(
        [pd.read_excel(label_xls, s, usecols=["PatientID", "Output"]) for s in sheets],
        ignore_index=True,
    )
    df = df.merge(labels, on="PatientID", how="left")
    assert df["Output"].isna().sum() == 0, "Output 라벨 조인 실패한 환자 존재"
    return df.reset_index(drop=True), aec_cols


def smd(x1, x0):
    """표준화평균차(standardized mean difference). |SMD|<0.1이면 balanced로 간주(Austin 2009)."""
    v1, v0 = np.var(x1, ddof=1), np.var(x0, ddof=1)
    pooled_sd = np.sqrt((v1 + v0) / 2)
    if pooled_sd == 0:
        return 0.0
    return (np.mean(x1) - np.mean(x0)) / pooled_sd


def fit_propensity(df_sex):
    X = df_sex[COVARIATES].to_numpy(float)
    y = df_sex["Output"].to_numpy(int)
    Xs = StandardScaler().fit_transform(X)
    model = LogisticRegression(max_iter=1000).fit(Xs, y)
    p = model.predict_proba(Xs)[:, 1]
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit_ps = np.log(p / (1 - p))
    return logit_ps


def greedy_match(case_pos, control_pos, logit_ps, caliper, rng):
    """무작위 순서로 케이스를 처리하며 caliper 안에서 가장 가까운 미사용 대조군을 1:1로 매칭(비복원)."""
    order = rng.permutation(len(case_pos))
    available = list(control_pos)
    pairs = []
    for oi in order:
        ci = case_pos[oi]
        if not available:
            break
        avail_arr = np.array(available)
        diffs = np.abs(logit_ps[avail_arr] - logit_ps[ci])
        j = int(np.argmin(diffs))
        if diffs[j] <= caliper:
            pairs.append((ci, avail_arr[j]))
            available.pop(j)
    return pairs


def match_sex(df_sex, sex_label):
    logit_ps = fit_propensity(df_sex)
    caliper = CALIPER_MULT * np.std(logit_ps, ddof=1)

    case_pos = np.flatnonzero(df_sex["Output"].to_numpy() == 1)
    control_pos = np.flatnonzero(df_sex["Output"].to_numpy() == 0)

    pairs = greedy_match(case_pos, control_pos, logit_ps, caliper, RNG)
    print(f"[{sex_label}] case={len(case_pos)}, control={len(control_pos)}, "
          f"caliper={caliper:.4f}, matched pairs={len(pairs)} "
          f"({len(pairs) / len(case_pos):.1%} of cases matched)")

    # Balance: 매칭 전(전체 case vs control) vs 매칭 후(matched pairs만)
    balance_rows = []
    case_idx_all = df_sex.index[case_pos].to_numpy()
    control_idx_all = df_sex.index[control_pos].to_numpy()
    matched_case_idx = df_sex.index[[p[0] for p in pairs]].to_numpy()
    matched_control_idx = df_sex.index[[p[1] for p in pairs]].to_numpy()
    for cov in COVARIATES:
        before = smd(df_sex.loc[case_idx_all, cov], df_sex.loc[control_idx_all, cov])
        after = smd(df_sex.loc[matched_case_idx, cov], df_sex.loc[matched_control_idx, cov])
        balance_rows.append({"sex": sex_label, "covariate": cov, "smd_before": before, "smd_after": after})
    balance_df = pd.DataFrame(balance_rows)

    return matched_case_idx, matched_control_idx, balance_df


def pointwise_paired_test(case_curves, control_curves, aec_cols):
    """matched pair의 곡선 차이를 위치별 Wilcoxon signed-rank + BH-FDR로 검정.
    부호는 02_aec_fda_pipeline.py(0701)와 동일하게 normal(control) - low_SMI(case)로 맞춰서,
    cohend_paired를 0701이 보고한 Cohen's d와 부호까지 직접 비교할 수 있게 한다."""
    diff = control_curves - case_curves  # (n_pairs, n_positions), normal - low_SMI
    pvals = np.array([
        stats.wilcoxon(diff[:, k])[1] if np.any(diff[:, k] != 0) else 1.0
        for k in range(diff.shape[1])
    ])
    fdr = multipletests(pvals, method="fdr_bh")[1]
    cohend = diff.mean(axis=0) / diff.std(axis=0, ddof=1)
    return pd.DataFrame({
        "aec_col": aec_cols, "index": np.arange(1, diff.shape[1] + 1),
        "mean_diff_normal_minus_lowsmi": diff.mean(axis=0),
        "p_wilcoxon": pvals, "p_fdr": fdr, "cohend_paired": cohend,
    })


def fdr_segments(pw_df, alpha=0.05):
    sig = pw_df["p_fdr"].to_numpy() < alpha
    segments = []
    start = None
    for i, s in enumerate(sig):
        if s and start is None:
            start = i
        elif not s and start is not None:
            segments.append((start + 1, i))  # 1-based inclusive
            start = None
    if start is not None:
        segments.append((start + 1, len(sig)))
    return segments


def plot_pointwise(results_by_sex, title_prefix, fname):
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.5))
    for ax, sex_label in zip(axes, ["M", "F"]):
        pw_df = results_by_sex[sex_label]
        idx = pw_df["index"].to_numpy()
        ax.plot(idx, pw_df["mean_diff_normal_minus_lowsmi"], color="#D04F5B", label="mean diff (normal-low_SMI)")
        ax.axhline(0, color="gray", linewidth=0.8)
        for start, end in fdr_segments(pw_df):
            ax.axvspan(start - 1, end - 1, color="red", alpha=0.15)
        for start, end in PRIOR_SEGMENTS.get(sex_label, []):
            ax.axvspan(start - 1, end - 1, color="blue", alpha=0.08, hatch="//")
        ax.set_title(f"{title_prefix} ({sex_label}) - 빨강: matched FDR<0.05, 파랑 빗금: 0701 유의구간")
        ax.set_xlabel("aec index (1=pubis, 128=liver upper)")
        ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, fname)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"저장: {path}")


def segment_comparison_vs_0701(pointwise_deriv):
    """0701이 찾은 각 유의구간에서 matched derivative 검정 결과(방향/크기/유의성)를
    0701이 보고한 Cohen's d와 나란히 놓고 대조한다 - FDR 문턱을 넘는지뿐 아니라
    '방향이 같은지', '크기가 얼마나 줄었는지'까지 확인하기 위함."""
    rows = []
    for (sex_label, start, end), original_d in PRIOR_COHEND.items():
        pw = pointwise_deriv[sex_label]
        sub = pw[(pw["index"] >= start) & (pw["index"] <= end)]
        matched_d = sub["cohend_paired"].mean()
        rows.append({
            "sex": sex_label, "segment": f"{start}-{end}",
            "cohend_0701_unmatched": original_d,
            "cohend_matched": matched_d,
            "same_direction": bool(np.sign(matched_d) == np.sign(original_d)),
            "magnitude_ratio_matched_over_0701": matched_d / original_d,
            "min_p_wilcoxon_in_segment": sub["p_wilcoxon"].min(),
            "min_p_fdr_in_segment": sub["p_fdr"].min(),
        })
    return pd.DataFrame(rows)


def plot_balance(balance_df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, sex_label in zip(axes, ["M", "F"]):
        sub = balance_df[balance_df["sex"] == sex_label]
        y = np.arange(len(sub))
        ax.scatter(sub["smd_before"], y, color="#999999", label="매칭 전", zorder=3)
        ax.scatter(sub["smd_after"], y, color="#D04F5B", label="매칭 후", zorder=3)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.axvline(0.1, color="gray", linestyle="--", linewidth=0.8)
        ax.axvline(-0.1, color="gray", linestyle="--", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["covariate"])
        ax.set_xlabel("Standardized Mean Difference")
        ax.set_title(f"Covariate balance ({sex_label})")
        ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "matched_balance_love_plot.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"저장: {path}")


def main():
    df, aec_cols = load_data()
    df["PatientSex"] = df["PatientSex"].astype(str).str.upper()

    pointwise_level = {}
    pointwise_deriv = {}
    balance_all = []
    mean_ma_rows = []
    matched_pairs_all = []

    for sex_label in ["M", "F"]:
        df_sex = df[df["PatientSex"] == sex_label].reset_index(drop=True)
        matched_case_idx, matched_control_idx, balance_df = match_sex(df_sex, sex_label)
        balance_all.append(balance_df)

        case_curves = df_sex.loc[matched_case_idx, aec_cols].to_numpy(float)
        control_curves = df_sex.loc[matched_control_idx, aec_cols].to_numpy(float)

        pointwise_level[sex_label] = pointwise_paired_test(case_curves, control_curves, aec_cols)

        case_deriv = np.gradient(case_curves, axis=1)
        control_deriv = np.gradient(control_curves, axis=1)
        pointwise_deriv[sex_label] = pointwise_paired_test(case_deriv, control_deriv, aec_cols)

        mean_ma_case = case_curves.mean(axis=1)
        mean_ma_control = control_curves.mean(axis=1)
        d = mean_ma_control - mean_ma_case  # normal - low_SMI (0701과 동일 부호)
        stat_p = stats.wilcoxon(d)[1]
        mean_ma_rows.append({
            "sex": sex_label, "n_pairs": len(d),
            "mean_mA_normal": mean_ma_control.mean(), "mean_mA_low_smi": mean_ma_case.mean(),
            "mean_diff_normal_minus_lowsmi": d.mean(), "cohend_paired": d.mean() / d.std(ddof=1),
            "p_wilcoxon": stat_p,
        })

        matched_pairs_all.append(pd.DataFrame({
            "sex": sex_label,
            "case_PatientID": df_sex.loc[matched_case_idx, "PatientID"].to_numpy(),
            "control_PatientID": df_sex.loc[matched_control_idx, "PatientID"].to_numpy(),
        }))

        sig_level = fdr_segments(pointwise_level[sex_label])
        sig_deriv = fdr_segments(pointwise_deriv[sex_label])
        print(f"[{sex_label}] matched level 유의구간(FDR<0.05): {sig_level}")
        print(f"[{sex_label}] matched derivative 유의구간(FDR<0.05): {sig_deriv}")
        print(f"[{sex_label}] 0701 원래 유의구간(unpaired, unmatched): {PRIOR_SEGMENTS[sex_label]}")
        print(f"[{sex_label}] mean_mA matched paired diff={d.mean():+.3f}, "
              f"cohend={d.mean() / d.std(ddof=1):.3f}, p={stat_p:.4g}\n")

    balance_df = pd.concat(balance_all, ignore_index=True)
    mean_ma_df = pd.DataFrame(mean_ma_rows)
    matched_pairs_df = pd.concat(matched_pairs_all, ignore_index=True)

    print("=== Balance (SMD, |SMD|<0.1 = balanced) ===")
    print(balance_df.to_string(index=False))
    print("\n=== mean_mA: matched paired 비교 (normal - low_SMI) ===")
    print(mean_ma_df.to_string(index=False))

    segment_df = segment_comparison_vs_0701(pointwise_deriv)
    print("\n=== 0701 유의구간별 대조 (matched derivative, 부호=normal-low_SMI로 0701과 동일) ===")
    print(segment_df.to_string(index=False))

    plot_balance(balance_df)
    plot_pointwise(pointwise_level, "Matched pointwise level test", "matched_pointwise_level_test.png")
    plot_pointwise(pointwise_deriv, "Matched pointwise derivative test", "matched_pointwise_derivative_test.png")

    with pd.ExcelWriter(RESULTS_FILE, engine="openpyxl") as writer:
        balance_df.to_excel(writer, sheet_name="Balance_SMD", index=False)
        mean_ma_df.to_excel(writer, sheet_name="MeanMA_paired_test", index=False)
        matched_pairs_df.to_excel(writer, sheet_name="Matched_pairs", index=False)
        segment_df.to_excel(writer, sheet_name="Segment_comparison_vs_0701", index=False)
        for sex_label in ["M", "F"]:
            pointwise_level[sex_label].to_excel(writer, sheet_name=f"Pointwise_level_{sex_label}", index=False)
            pointwise_deriv[sex_label].to_excel(writer, sheet_name=f"Pointwise_deriv_{sex_label}", index=False)
    print(f"\n저장: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
