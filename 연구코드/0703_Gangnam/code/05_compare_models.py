"""
강남_model_features.xlsx로 clinic-only 대비 aec 추가/변형 방식들을 비교한다.
타깃: Output (0=정상, 1=저SMI/근감소증)

베이스라인 결과(이전 실행): clinic+aec(raw)를 단순 추가하면 ROC-AUC는 변화 없고(p=0.53)
PR-AUC는 오히려 유의하게 나빠짐(p<0.0001). mean_mA-Weight corr=0.54, slope_48_55-Weight
corr=0.40 -> aec 신호 상당수가 이미 clinic(체중)에 들어있는 정보와 중복되는 것으로 추정.
이를 검증하기 위해 아래 변형들을 "동일한 CV fold" 위에서 비교한다:

  1) clinic-only              : PatientAge, Sex_M, Height, Weight
  2) clinic+aec(raw)          : 1) + mean_mA, slope_48_55, slope_80_90, slope_94_97
  3) clinic+aec(residualized) : aec 4개를 Height+Weight에 대해 회귀한 잔차로 대체
                                (체중/키로 설명되는 부분을 제거하고 남는 정보만 사용,
                                 회귀는 매 fold의 train 데이터로만 적합 -> leakage 없음)
  4) clinic+aec+sex_interaction : 1)+2) + Sex_M*aec_i 교차항 4개
                                   (남/여 유의구간이 서로 달랐다는 사실을 모델에 명시적으로 반영)
  5) clinic+aec(elasticnet)   : 2)와 동일 피처, L2 로지스틱 대신 elasticnet(L1+L2) 사용
                                -> 계수가 0으로 죽는 피처가 있는지 확인 (변수 선택 관점)

모든 변형은 동일한 RepeatedStratifiedKFold(5-fold) 분할을 공유해서
Wilcoxon signed-rank test로 clinic-only 대비 fold-쌍 비교를 한다.
추가로 elasticnet은 전체 데이터로 한 번 더 적합해서 표준화 계수(변수 선택 결과)를 출력한다.

출력: 콘솔 비교표 + Wilcoxon 검정, figures/model_comparison_roc.png,
      excel/model_comparison_results.xlsx (CV_scores, Summary_vs_baseline, Elasticnet_coefficients)
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXCEL_DIR = os.path.join(BASE_DIR, "..", "excel")
FIGURES_DIR = os.path.join(BASE_DIR, "..", "figures", "05")
os.makedirs(FIGURES_DIR, exist_ok=True)

INPUT_FILE = os.path.join(EXCEL_DIR, "강남_model_features.xlsx")
RESULTS_FILE = os.path.join(EXCEL_DIR, "model_comparison_results.xlsx")

CLINIC_FEATURES = ["PatientAge", "Sex_M", "Height", "Weight"]
AEC_FEATURES = ["mean_mA", "slope_48_55", "slope_80_90", "slope_94_97"]
COVARIATES_FOR_RESID = ["Height", "Weight"]

N_SPLITS = 5
N_REPEATS = 1
RANDOM_STATE = 42
BASELINE_NAME = "1) clinic-only"


# ---------- 피처 생성 함수: 매 fold의 (train_idx, test_idx)를 받아 (X_train, X_test) 반환 ----------

def features_static(cols):
    def fn(df, tr, te):
        return df.iloc[tr][cols].to_numpy(float), df.iloc[te][cols].to_numpy(float)
    return fn


def features_residualized():
    cols = CLINIC_FEATURES + AEC_FEATURES

    def fn(df, tr, te):
        train_df, test_df = df.iloc[tr].copy(), df.iloc[te].copy()
        for col in AEC_FEATURES:
            lr = LinearRegression().fit(train_df[COVARIATES_FOR_RESID], train_df[col])
            train_df[col] = train_df[col] - lr.predict(train_df[COVARIATES_FOR_RESID])
            test_df[col] = test_df[col] - lr.predict(test_df[COVARIATES_FOR_RESID])
        return train_df[cols].to_numpy(float), test_df[cols].to_numpy(float)
    return fn


def features_sex_interaction():
    base_cols = CLINIC_FEATURES + AEC_FEATURES

    def fn(df, tr, te):
        train_df, test_df = df.iloc[tr], df.iloc[te]
        inter_tr = train_df[AEC_FEATURES].to_numpy(float) * train_df[["Sex_M"]].to_numpy(float)
        inter_te = test_df[AEC_FEATURES].to_numpy(float) * test_df[["Sex_M"]].to_numpy(float)
        Xtr = np.hstack([train_df[base_cols].to_numpy(float), inter_tr])
        Xte = np.hstack([test_df[base_cols].to_numpy(float), inter_te])
        return Xtr, Xte
    return fn


# ---------- 분류기 생성 함수 ----------

def make_logreg():
    return LogisticRegression(class_weight="balanced", max_iter=1000)


def make_elasticnet():
    return LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, C=1.0,
                               class_weight="balanced", max_iter=5000)


VARIANTS = [
    ("1) clinic-only", features_static(CLINIC_FEATURES), make_logreg),
    ("2) clinic+aec(raw)", features_static(CLINIC_FEATURES + AEC_FEATURES), make_logreg),
    ("3) clinic+aec(residualized)", features_residualized(), make_logreg),
    ("4) clinic+aec+sex_interaction", features_sex_interaction(), make_logreg),
    ("5) clinic+aec(elasticnet)", features_static(CLINIC_FEATURES + AEC_FEATURES), make_elasticnet),
]


def load_data() -> pd.DataFrame:
    df = pd.read_excel(INPUT_FILE)
    df["Sex_M"] = (df["PatientSex"] == "M").astype(int)
    return df.reset_index(drop=True)


def run_variant(df, y, folds, get_features, make_clf, name) -> pd.DataFrame:
    rows = []
    for fold_id, (tr, te) in enumerate(folds):
        Xtr, Xte = get_features(df, tr, te)
        scaler = StandardScaler().fit(Xtr)
        clf = make_clf()
        clf.fit(scaler.transform(Xtr), y[tr])
        proba = clf.predict_proba(scaler.transform(Xte))[:, 1]
        rows.append({
            "variant": name, "fold": fold_id,
            "roc_auc": roc_auc_score(y[te], proba),
            "pr_auc": average_precision_score(y[te], proba),
        })
    return pd.DataFrame(rows)


def summarize_and_test(all_scores: pd.DataFrame) -> pd.DataFrame:
    baseline = all_scores[all_scores["variant"] == BASELINE_NAME].sort_values("fold")
    rows = []
    for name in all_scores["variant"].unique():
        sub = all_scores[all_scores["variant"] == name].sort_values("fold")
        row = {"variant": name,
               "roc_auc_mean": sub["roc_auc"].mean(), "roc_auc_sd": sub["roc_auc"].std(),
               "pr_auc_mean": sub["pr_auc"].mean(), "pr_auc_sd": sub["pr_auc"].std()}
        if name != BASELINE_NAME:
            for metric in ["roc_auc", "pr_auc"]:
                diff = sub[metric].to_numpy() - baseline[metric].to_numpy()
                _, p = stats.wilcoxon(diff)
                row[f"{metric}_diff_vs_baseline"] = diff.mean()
                row[f"{metric}_wilcoxon_p"] = p
        rows.append(row)
    return pd.DataFrame(rows)


def plot_oof_roc(df, y, folds_5fold) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for name, get_features, make_clf in VARIANTS:
        proba = np.empty(len(y))
        for tr, te in folds_5fold:
            Xtr, Xte = get_features(df, tr, te)
            scaler = StandardScaler().fit(Xtr)
            clf = make_clf()
            clf.fit(scaler.transform(Xtr), y[tr])
            proba[te] = clf.predict_proba(scaler.transform(Xte))[:, 1]
        fpr, tpr, _ = roc_curve(y, proba)
        auc_val = roc_auc_score(y, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="chance")
    ax.set_title("모델별 OOF ROC curve (5-fold)")
    ax.legend(fontsize=7)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "model_comparison_roc.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"저장: {path}")


def elasticnet_coefficients(df, y) -> pd.DataFrame:
    """변수 선택 관점 확인용: 전체 데이터로 한 번 적합(성능평가 아님, 해석용)"""
    rows = []
    cols = CLINIC_FEATURES + AEC_FEATURES
    X = StandardScaler().fit_transform(df[cols].to_numpy(float))
    clf = make_elasticnet().fit(X, y)
    for col, coef in zip(cols, clf.coef_[0]):
        rows.append({"variant": "clinic+aec(raw)", "feature": col, "std_coef": coef})

    inter_cols = [f"SexM_x_{c}" for c in AEC_FEATURES]
    X_int = np.hstack([df[cols].to_numpy(float),
                        df[AEC_FEATURES].to_numpy(float) * df[["Sex_M"]].to_numpy(float)])
    X_int = StandardScaler().fit_transform(X_int)
    clf = make_elasticnet().fit(X_int, y)
    for col, coef in zip(cols + inter_cols, clf.coef_[0]):
        rows.append({"variant": "clinic+aec+sex_interaction", "feature": col, "std_coef": coef})

    return pd.DataFrame(rows)


def main():
    df = load_data()
    y = df["Output"].to_numpy()

    rkf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    folds = list(rkf.split(df, y))

    print(f"=== {N_SPLITS}-fold x {N_REPEATS} repeat 교차검증 (n={len(df)}, "
          f"Output=1 비율={y.mean():.3f}) ===\n")

    all_scores = pd.concat(
        [run_variant(df, y, folds, get_features, make_clf, name)
         for name, get_features, make_clf in VARIANTS],
        ignore_index=True,
    )

    summary = summarize_and_test(all_scores)
    pd.set_option("display.width", 140)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    folds_5fold = list(skf.split(df, y))
    plot_oof_roc(df, y, folds_5fold)

    coef_df = elasticnet_coefficients(df, y)
    print("\n[Elasticnet 표준화 계수 (|coef| 내림차순, 전체데이터 1회 적합 - 해석용)]")
    print(coef_df.reindex(coef_df["std_coef"].abs().sort_values(ascending=False).index)
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    with pd.ExcelWriter(RESULTS_FILE) as writer:
        all_scores.to_excel(writer, sheet_name="CV_scores", index=False)
        summary.to_excel(writer, sheet_name="Summary_vs_baseline", index=False)
        coef_df.to_excel(writer, sheet_name="Elasticnet_coefficients", index=False)
    print(f"\n저장: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
