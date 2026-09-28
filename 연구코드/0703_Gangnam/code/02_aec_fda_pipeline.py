"""
aec_1~aec_128 = CT 스캔 위치별 X-ray Tube Current(mA) 프로파일 (AEC = Automatic Exposure Control).
환자별로 aec_1=치골(pubis), aec_128=간 상부(liver upper)가 되도록 랜드마크 정규화되어 있음
(즉 index는 물리적 mm 위치가 아니라 pubis(0%)~liver upper(100%) 상대위치 -> 환자 간 비교 가능).
128개를 독립 변수로 취급하는 대신, 환자별 하나의 "곡선"으로 보고 분석한다.

방법론 (연구적 근거):
  A) Functional Data Analysis (Ramsay & Silverman, 2005)
     1. B-spline 기저로 곡선 스무딩
     2. Level FPCA: 원곡선의 형태 변동(주로 "높이") 주요 모드 추출
     3. Derivative FPCA: 곡선의 1차 도함수(기울기 프로파일)에 대해 별도로 FPCA 수행
        -> level FPCA는 "높이"에 의해 지배되므로, 기울기(변화 패턴) 자체의 그룹 차이는
           도함수 곡선을 따로 분석해야 확인 가능
     4. 두 경우 모두 FPCA 점수에 permutation 기반 다변량 group test 적용 (normal vs low_SMI, 성별 층화)
  B) 임상적 요약지표 (CT dosimetry 문헌에서 실제 사용하는 지표)
     - Level: mean mA, SD, CV(=SD/mean), AUC, 전체 선형회귀 slope
     - Slope: 평균 |기울기|, 최대 상승/하강 기울기, total variation (변화의 누적크기)
     - Mann-Whitney U + FDR(BH), 성별 층화

출력: figures/aec_fpca_components.png, figures/aec_fpca_scores_boxplot.png,
      figures/aec_derivative_components.png, figures/aec_derivative_scores_boxplot.png,
      figures/aec_summary_features_boxplot.png, aec_fda_results.xlsx
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
from skfda import FDataGrid
from skfda.preprocessing.smoothing import KernelSmoother
from skfda.preprocessing.smoothing.kernel_smoothers import NadarayaWatsonSmoother
from skfda.representation.basis import BSplineBasis
from skfda.preprocessing.dim_reduction import FPCA

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXCEL_DIR = os.path.join(BASE_DIR, "..", "excel")

INPUT_FILE = os.path.join(EXCEL_DIR, "강남_aec_128.xlsx")
OUTPUT_FILE = os.path.join(EXCEL_DIR, "aec_fda_results.xlsx")
FIGURES_DIR = os.path.join(BASE_DIR, "..", "figures", "02")
os.makedirs(FIGURES_DIR, exist_ok=True)
N_FPCA_COMPONENTS = 4
N_PERMUTATIONS = 2000
RNG = np.random.default_rng(42)


def fig_path(name: str) -> str:
    return os.path.join(FIGURES_DIR, name)


def idx_to_pct(idx) -> np.ndarray:
    """aec index(1~128) -> pubis(0%)~liver upper(100%) 상대위치(%)"""
    return (np.asarray(idx) - 1) / 127 * 100


def add_pct_top_axis(ax):
    top = ax.secondary_xaxis("top", functions=(idx_to_pct, lambda p: p / 100 * 127 + 1))
    top.set_xlabel("치골(0%) → 간 상부(100%) 상대위치")


def load_data():
    xls = pd.ExcelFile(INPUT_FILE)
    m_normal = pd.read_excel(xls, "M_normal")
    m_low = pd.read_excel(xls, "M_low_SMI")
    f_normal = pd.read_excel(xls, "F_normal")
    f_low = pd.read_excel(xls, "F_low_SMI")
    aec_cols = [c for c in m_normal.columns if c.startswith("aec_")]
    return m_normal, m_low, f_normal, f_low, aec_cols


def to_fdatagrid(df: pd.DataFrame, aec_cols: list[str]) -> FDataGrid:
    grid_points = np.arange(1, len(aec_cols) + 1)
    data_matrix = df[aec_cols].to_numpy()
    fd = FDataGrid(data_matrix=data_matrix, grid_points=grid_points)
    basis = BSplineBasis(n_basis=20, domain_range=fd.domain_range)
    return fd.to_basis(basis)


def permutation_test_fpca(scores_a: np.ndarray, scores_b: np.ndarray, n_perm: int = N_PERMUTATIONS):
    """FPCA 점수 벡터에 대한 permutation 기반 다변량 group test (Hotelling T^2 유사 통계량)"""
    def stat(a, b):
        ma, mb = a.mean(axis=0), b.mean(axis=0)
        pooled_cov = np.cov(np.vstack([a, b]), rowvar=False)
        diff = ma - mb
        return diff @ np.linalg.pinv(pooled_cov) @ diff

    observed = stat(scores_a, scores_b)
    pooled = np.vstack([scores_a, scores_b])
    n_a = len(scores_a)
    perm_stats = np.empty(n_perm)
    for i in range(n_perm):
        idx = RNG.permutation(len(pooled))
        a_p, b_p = pooled[idx[:n_a]], pooled[idx[n_a:]]
        perm_stats[i] = stat(a_p, b_p)
    p = (np.sum(perm_stats >= observed) + 1) / (n_perm + 1)
    return observed, p


def summary_features(df: pd.DataFrame, aec_cols: list[str]) -> pd.DataFrame:
    X = df[aec_cols].to_numpy().astype(float)
    idx = np.arange(1, len(aec_cols) + 1)
    mean_ = X.mean(axis=1)
    sd_ = X.std(axis=1, ddof=1)
    cv_ = sd_ / mean_
    auc_ = np.trapezoid(X, x=idx, axis=1)
    slope_ = np.array([np.polyfit(idx, row, 1)[0] for row in X])

    # 국소 기울기(1차 차분) 기반 지표 -> 전체 선형회귀 slope로는 못 잡는 변화 패턴 포착
    diffs = np.diff(X, axis=1)
    mean_abs_slope = np.abs(diffs).mean(axis=1)
    max_rise = diffs.max(axis=1)
    max_fall = diffs.min(axis=1)
    total_variation = np.abs(diffs).sum(axis=1)

    return pd.DataFrame({
        "mean_mA": mean_, "sd_mA": sd_, "cv_mA": cv_, "auc_mA": auc_, "slope_mA": slope_,
        "mean_abs_slope": mean_abs_slope, "max_rise": max_rise, "max_fall": max_fall,
        "total_variation": total_variation,
    })


def run_fpca(fd_basis, n_normal: int, sex_label: str, mode_label: str):
    """level 또는 derivative FDataBasis 공용 FPCA + permutation test"""
    fpca = FPCA(n_components=N_FPCA_COMPONENTS)
    scores = fpca.fit_transform(fd_basis)

    scores_normal = scores[:n_normal]
    scores_low = scores[n_normal:]

    stat, p_global = permutation_test_fpca(scores_normal, scores_low)
    print(f"[{sex_label}/{mode_label}] FPCA 설명분산비율: {np.round(fpca.explained_variance_ratio_, 3)}")
    print(f"[{sex_label}/{mode_label}] Permutation global test (Hotelling-T^2 style): "
          f"stat={stat:.2f}, p={p_global:.4f}")

    pc_rows = []
    for k in range(N_FPCA_COMPONENTS):
        _, p_k = stats.mannwhitneyu(scores_normal[:, k], scores_low[:, k], alternative="two-sided")
        pc_rows.append({"sex": sex_label, "mode": mode_label, "component": f"FPC{k+1}",
                         "explained_var_ratio": fpca.explained_variance_ratio_[k],
                         "p_mannwhitney": p_k})
    pc_df = pd.DataFrame(pc_rows)

    return fpca, scores, scores_normal, scores_low, p_global, pc_df


def run_fpca_per_sex(normal, low, aec_cols, sex_label):
    both = pd.concat([normal, low], ignore_index=True)
    fd_basis = to_fdatagrid(both, aec_cols)
    n_normal = len(normal)

    level = run_fpca(fd_basis, n_normal, sex_label, "level")
    fd_deriv = fd_basis.derivative(order=1)
    deriv = run_fpca(fd_deriv, n_normal, sex_label, "derivative")

    return level, deriv, fd_basis, fd_deriv, both


def plot_fpca_components(fpca_m, fpca_f, title, ylabel, filename, scale=30):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, fpca, label in [(axes[0], fpca_m, "남성"), (axes[1], fpca_f, "여성")]:
        mean_curve = fpca.mean_.to_grid(np.arange(1, 129)).data_matrix[0, :, 0]
        components = fpca.components_.to_grid(np.arange(1, 129)).data_matrix[:, :, 0]
        for k in range(min(3, components.shape[0])):
            ax.plot(range(1, 129), mean_curve + scale * components[k], label=f"평균 ± FPC{k+1}")
        ax.plot(range(1, 129), mean_curve, "k--", linewidth=2, label="평균곡선")
        ax.set_title(f"{label}: {title}")
        ax.set_xlabel("aec index (1=치골 pubis, 128=간 상부 liver upper)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        add_pct_top_axis(ax)
    plt.tight_layout()
    plt.savefig(fig_path(filename), dpi=150)
    plt.close()
    print(f"저장: {filename}")


def plot_fpca_scores(scores_m, normal_m_n, scores_f, normal_f_n, title, filename):
    rows = []
    for label, scores, n_normal in [("남성", scores_m, normal_m_n), ("여성", scores_f, normal_f_n)]:
        group = ["normal"] * n_normal + ["low_SMI"] * (len(scores) - n_normal)
        for k in range(scores.shape[1]):
            for v, g in zip(scores[:, k], group):
                rows.append({"sex": label, "component": f"FPC{k+1}", "score": v, "group": g})
    df = pd.DataFrame(rows)

    g = sns.catplot(data=df, x="group", y="score", col="component", row="sex", kind="box", height=3, sharey=False)
    g.fig.suptitle(title, y=1.02)
    g.savefig(fig_path(filename), dpi=150)
    plt.close(g.fig)
    print(f"저장: {filename}")


def pointwise_derivative_test(fd_deriv, n_normal: int, sex_label: str):
    """도함수 곡선을 grid point별로 정확히 어디서 그룹차가 나는지 pointwise 검정 (FDR 보정)"""
    grid = np.arange(1, 129)
    values = fd_deriv(grid)[:, :, 0]
    normal_vals = values[:n_normal]
    low_vals = values[n_normal:]

    pvals = np.array([
        stats.mannwhitneyu(normal_vals[:, i], low_vals[:, i], alternative="two-sided")[1]
        for i in range(len(grid))
    ])
    fdr = multipletests(pvals, method="fdr_bh")[1]

    n_a, n_b = normal_vals.shape[0], low_vals.shape[0]
    cohend = np.array([
        (normal_vals[:, i].mean() - low_vals[:, i].mean()) / np.sqrt(
            ((n_a - 1) * normal_vals[:, i].std(ddof=1) ** 2
             + (n_b - 1) * low_vals[:, i].std(ddof=1) ** 2) / (n_a + n_b - 2)
        )
        for i in range(len(grid))
    ])

    df = pd.DataFrame({"sex": sex_label, "aec_index": grid, "pct_from_pubis": idx_to_pct(grid),
                        "p_mannwhitney": pvals, "p_fdr": fdr, "cohend": cohend})
    return df, normal_vals, low_vals


def find_significant_segments(df: pd.DataFrame, alpha: float = 0.05):
    sig = (df["p_fdr"].to_numpy() < alpha)
    idx = df["aec_index"].to_numpy()
    segments, start = [], None
    for i, s in enumerate(sig):
        if s and start is None:
            start = idx[i]
        if (not s or i == len(sig) - 1) and start is not None:
            end = idx[i] if s else idx[i - 1]
            segments.append((int(start), int(end)))
            start = None
    return segments


def plot_pointwise_derivative(df_m, nv_m, lv_m, seg_m, df_f, nv_f, lv_f, seg_f):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    for ax, df, normal_vals, low_vals, segments, label in [
        (axes[0], df_m, nv_m, lv_m, seg_m, "남성"),
        (axes[1], df_f, nv_f, lv_f, seg_f, "여성"),
    ]:
        grid = df["aec_index"].to_numpy()
        ax.plot(grid, normal_vals.mean(axis=0), label="normal 평균 기울기")
        ax.plot(grid, low_vals.mean(axis=0), label="low_SMI 평균 기울기")
        for start, end in segments:
            ax.axvspan(start, end, color="red", alpha=0.15)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_title(f"{label}: pointwise 기울기 비교 (빨간 음영 = FDR<0.05 구간, {len(segments)}개)")
        ax.set_ylabel("d(mA)/d(scan position)")
        ax.legend(fontsize=8)
        add_pct_top_axis(ax)
    axes[-1].set_xlabel("aec index (1=치골 pubis, 128=간 상부 liver upper)")
    plt.tight_layout()
    plt.savefig(fig_path("aec_pointwise_derivative_test.png"), dpi=150)
    plt.close()
    print("저장: aec_pointwise_derivative_test.png")


def segment_slope_features(fd_deriv, n_normal: int, segments: list[tuple[int, int]], sex_label: str):
    """FDR로 찾은 구간별 평균 기울기를 스칼라로 뽑아 검정 -> 구체적 effect size 보고용"""
    grid = np.arange(1, 129)
    values = fd_deriv(grid)[:, :, 0]
    normal_vals = values[:n_normal]
    low_vals = values[n_normal:]

    rows = []
    for start, end in segments:
        mask = (grid >= start) & (grid <= end)
        a = normal_vals[:, mask].mean(axis=1)
        b = low_vals[:, mask].mean(axis=1)
        _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        pooled_sd = np.sqrt(((len(a) - 1) * a.std(ddof=1) ** 2 + (len(b) - 1) * b.std(ddof=1) ** 2)
                             / (len(a) + len(b) - 2))
        d = (a.mean() - b.mean()) / pooled_sd
        pct_start, pct_end = idx_to_pct(start), idx_to_pct(end)
        rows.append({"sex": sex_label, "segment": f"aec_{start}~aec_{end}",
                      "pct_range": f"{pct_start:.0f}%~{pct_end:.0f}%",
                      "mean_slope_normal": a.mean(), "mean_slope_low_SMI": b.mean(),
                      "p_mannwhitney": p, "cohend": d})
    return pd.DataFrame(rows)


def main():
    m_normal, m_low, f_normal, f_low, aec_cols = load_data()

    # ---------- A) Functional Data Analysis: level + derivative(기울기) ----------
    level_m, deriv_m, fd_m, fd_deriv_m, _ = run_fpca_per_sex(m_normal, m_low, aec_cols, "M")
    level_f, deriv_f, fd_f, fd_deriv_f, _ = run_fpca_per_sex(f_normal, f_low, aec_cols, "F")

    fpca_m, scores_m, sn_m, sl_m, p_global_m, pc_df_m = level_m
    fpca_f, scores_f, sn_f, sl_f, p_global_f, pc_df_f = level_f
    dfpca_m, dscores_m, dsn_m, dsl_m, dp_global_m, dpc_df_m = deriv_m
    dfpca_f, dscores_f, dsn_f, dsl_f, dp_global_f, dpc_df_f = deriv_f

    plot_fpca_components(fpca_m, fpca_f, "Level FPCA 주요 변동 모드", "X-ray Tube Current (mA)",
                          "aec_fpca_components.png", scale=30)
    plot_fpca_scores(scores_m, len(m_normal), scores_f, len(f_normal),
                      "성별/그룹별 Level FPCA 점수 분포", "aec_fpca_scores_boxplot.png")

    plot_fpca_components(dfpca_m, dfpca_f, "Derivative(기울기) FPCA 주요 변동 모드", "d(mA)/d(scan position)",
                          "aec_derivative_components.png", scale=5)
    plot_fpca_scores(dscores_m, len(m_normal), dscores_f, len(f_normal),
                      "성별/그룹별 Derivative FPCA 점수 분포", "aec_derivative_scores_boxplot.png")

    global_test_df = pd.DataFrame([
        {"sex": "M", "mode": "level", "test": "Permutation Hotelling-T2 (FPCA scores)", "p_value": p_global_m},
        {"sex": "F", "mode": "level", "test": "Permutation Hotelling-T2 (FPCA scores)", "p_value": p_global_f},
        {"sex": "M", "mode": "derivative", "test": "Permutation Hotelling-T2 (FPCA scores)", "p_value": dp_global_m},
        {"sex": "F", "mode": "derivative", "test": "Permutation Hotelling-T2 (FPCA scores)", "p_value": dp_global_f},
    ])
    print("\n[FDA 요약: level vs derivative]")
    print(global_test_df.to_string(index=False))

    pc_df_all = pd.concat([pc_df_m, pc_df_f, dpc_df_m, dpc_df_f], ignore_index=True)

    # ---------- A-1) Pointwise derivative test: 정확히 어느 구간에서 기울기가 다른가 ----------
    pw_df_m, nv_m, lv_m = pointwise_derivative_test(fd_deriv_m, len(m_normal), "M")
    pw_df_f, nv_f, lv_f = pointwise_derivative_test(fd_deriv_f, len(f_normal), "F")
    seg_m = find_significant_segments(pw_df_m)
    seg_f = find_significant_segments(pw_df_f)

    def seg_pct_str(segments):
        return [(s, e, f"{idx_to_pct(s):.0f}%~{idx_to_pct(e):.0f}%") for s, e in segments]

    print(f"\n[M] FDR<0.05 유의 구간 (index, pubis 0%~liver 100% 기준): {seg_pct_str(seg_m)}")
    print(f"[F] FDR<0.05 유의 구간 (index, pubis 0%~liver 100% 기준): {seg_pct_str(seg_f)}")

    plot_pointwise_derivative(pw_df_m, nv_m, lv_m, seg_m, pw_df_f, nv_f, lv_f, seg_f)

    seg_feat_m = segment_slope_features(fd_deriv_m, len(m_normal), seg_m, "M")
    seg_feat_f = segment_slope_features(fd_deriv_f, len(f_normal), seg_f, "F")
    seg_feat_df = pd.concat([seg_feat_m, seg_feat_f], ignore_index=True)
    print("\n[유의 구간별 평균 기울기 검정]")
    print(seg_feat_df.to_string(index=False))

    # ---------- B) 임상적 요약지표 ----------
    feat_rows = []
    for sex, normal, low in [("M", m_normal, m_low), ("F", f_normal, f_low)]:
        feat_normal = summary_features(normal, aec_cols)
        feat_low = summary_features(low, aec_cols)
        for col in feat_normal.columns:
            a, b = feat_normal[col], feat_low[col]
            _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            na, nb = a.std(ddof=1), b.std(ddof=1)
            pooled_sd = np.sqrt(((len(a) - 1) * na**2 + (len(b) - 1) * nb**2) / (len(a) + len(b) - 2))
            d = (a.mean() - b.mean()) / pooled_sd
            feat_rows.append({"sex": sex, "feature": col, "p_mannwhitney": p, "cohend": d})
    feat_df = pd.DataFrame(feat_rows)
    for sex in ["M", "F"]:
        mask = feat_df["sex"] == sex
        feat_df.loc[mask, "p_fdr"] = multipletests(feat_df.loc[mask, "p_mannwhitney"], method="fdr_bh")[1]

    print("\n[임상 요약지표 검정 결과]")
    print(feat_df.to_string(index=False))

    # boxplot of summary features
    all_feat_rows = []
    for sex, normal, low in [("M", m_normal, m_low), ("F", f_normal, f_low)]:
        fn = summary_features(normal, aec_cols); fn["group"] = "normal"; fn["sex"] = sex
        fl = summary_features(low, aec_cols); fl["group"] = "low_SMI"; fl["sex"] = sex
        all_feat_rows.append(fn); all_feat_rows.append(fl)
    feat_long = pd.concat(all_feat_rows, ignore_index=True).melt(
        id_vars=["group", "sex"], var_name="feature", value_name="value"
    )
    g = sns.catplot(data=feat_long, x="group", y="value", col="feature", row="sex", kind="box",
                     height=3, sharey=False)
    g.fig.suptitle("성별/그룹별 임상 요약지표 분포", y=1.02)
    g.savefig(fig_path("aec_summary_features_boxplot.png"), dpi=150)
    plt.close(g.fig)
    print("저장: aec_summary_features_boxplot.png")

    with pd.ExcelWriter(OUTPUT_FILE) as writer:
        global_test_df.to_excel(writer, sheet_name="FPCA_global_test", index=False)
        pc_df_all.to_excel(writer, sheet_name="FPCA_component_tests", index=False)
        pd.concat([pw_df_m, pw_df_f], ignore_index=True).to_excel(
            writer, sheet_name="Pointwise_derivative_test", index=False
        )
        seg_feat_df.to_excel(writer, sheet_name="Segment_slope_features", index=False)
        feat_df.to_excel(writer, sheet_name="Summary_features", index=False)

    print(f"\n결과 저장 완료: {OUTPUT_FILE}")
    print(f"이미지 저장 완료 ({FIGURES_DIR}/): aec_fpca_components.png, aec_fpca_scores_boxplot.png, "
          f"aec_derivative_components.png, aec_derivative_scores_boxplot.png, "
          f"aec_pointwise_derivative_test.png, aec_summary_features_boxplot.png")


if __name__ == "__main__":
    main()
