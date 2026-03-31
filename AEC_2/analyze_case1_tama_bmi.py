import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


BASE_DIR = Path(r"C:\Users\user\Desktop\Study")
CASE1_DIR = BASE_DIR / "result" / "Case1" / "강남"
SOURCE_XLSX = BASE_DIR / "data" / "AEC" / "강남" / "강남_DLO_Results.xlsx"
OUTPUT_DIR = CASE1_DIR
COMMON_OUTPUT_DIR = CASE1_DIR / "tama_bmi_analysis"

NORMALIZATION_ORDER = ["none", "zscore"]
VALUE_COLUMNS = {
    "BMI": "BMI",
    "TAMA": "SRC_Report",
}


def clean_patient_id(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith(".0"):
        text = text[:-2]
    return text


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    COMMON_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for normalization in NORMALIZATION_ORDER:
        normalization_output_dir(normalization).mkdir(parents=True, exist_ok=True)


def normalization_output_dir(normalization: str) -> Path:
    return CASE1_DIR / normalization / "tama_bmi_analysis"


def load_source_dataframe() -> pd.DataFrame:
    df = pd.read_excel(SOURCE_XLSX)
    needed = ["PatientID", "BMI", "SRC_Report"]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in source excel: {missing}")

    df = df[needed].copy()
    df["patient_id"] = df["PatientID"].map(clean_patient_id)
    df["BMI"] = pd.to_numeric(df["BMI"], errors="coerce")
    df["TAMA"] = pd.to_numeric(df["SRC_Report"], errors="coerce")
    df = df.drop(columns=["PatientID", "SRC_Report"])
    df = df.dropna(subset=["patient_id", "BMI", "TAMA"]).copy()

    duplicate_mask = df["patient_id"].duplicated(keep=False)
    if duplicate_mask.any():
        duplicate_df = (
            df.loc[duplicate_mask]
            .sort_values(["patient_id", "BMI", "TAMA"])
            .copy()
        )
        duplicate_df.to_csv(COMMON_OUTPUT_DIR / "source_duplicates.csv", index=False, encoding="utf-8-sig")
        duplicate_summary = (
            duplicate_df.groupby("patient_id")
            .agg(
                record_count=("patient_id", "size"),
                BMI_median=("BMI", "median"),
                BMI_unique_count=("BMI", "nunique"),
                TAMA_median=("TAMA", "median"),
                TAMA_min=("TAMA", "min"),
                TAMA_max=("TAMA", "max"),
                TAMA_unique_count=("TAMA", "nunique"),
            )
            .reset_index()
        )
        duplicate_summary.to_csv(
            COMMON_OUTPUT_DIR / "source_duplicate_patient_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    patient_level_df = (
        df.groupby("patient_id", as_index=False)
        .agg(
            BMI=("BMI", "median"),
            TAMA=("TAMA", "median"),
        )
        .reset_index(drop=True)
    )

    patient_level_df.to_csv(
        COMMON_OUTPUT_DIR / "source_patient_level_bmi_tama.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return patient_level_df


def load_cluster_members(normalization: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cluster_root = CASE1_DIR / normalization / "cluster_examples"
    member_frames: list[pd.DataFrame] = []
    coverage_rows: list[dict] = []

    for cluster_dir in sorted(cluster_root.glob("cluster_*")):
        all_members_path = cluster_dir / "all_members.csv"
        if not all_members_path.exists():
            continue

        cluster_id = int(cluster_dir.name.split("_")[-1])
        cluster_df = pd.read_csv(all_members_path, dtype={"patient_id": "string"})
        cluster_df["patient_id"] = cluster_df["patient_id"].map(clean_patient_id)
        cluster_df["normalization"] = normalization
        cluster_df["cluster_id"] = cluster_id
        cluster_df["cluster_label"] = f"cluster_{cluster_id}"
        cluster_df["source_csv"] = str(all_members_path)
        cluster_df = cluster_df.dropna(subset=["patient_id"]).copy()
        member_frames.append(cluster_df)

        coverage_rows.append(
            {
                "normalization": normalization,
                "cluster_id": cluster_id,
                "cluster_label": f"cluster_{cluster_id}",
                "member_count": len(cluster_df),
            }
        )

    if not member_frames:
        raise ValueError(f"No cluster members found under {cluster_root}")

    return pd.concat(member_frames, ignore_index=True), pd.DataFrame(coverage_rows)


def summarize_series(series: pd.Series) -> dict:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {
            "n": 0,
            "mean": math.nan,
            "std": math.nan,
            "median": math.nan,
            "q1": math.nan,
            "q3": math.nan,
            "min": math.nan,
            "max": math.nan,
            "iqr": math.nan,
        }

    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    return {
        "n": int(clean.shape[0]),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=1)) if clean.shape[0] > 1 else 0.0,
        "median": float(clean.median()),
        "q1": float(q1),
        "q3": float(q3),
        "min": float(clean.min()),
        "max": float(clean.max()),
        "iqr": float(q3 - q1),
    }


def build_summary(merged_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for normalization, norm_df in merged_df.groupby("normalization"):
        for cluster_id, cluster_df in norm_df.groupby("cluster_id"):
            row = {
                "normalization": normalization,
                "cluster_id": int(cluster_id),
                "cluster_label": f"cluster_{int(cluster_id)}",
                "member_count": int(len(cluster_df)),
            }
            for value_col in VALUE_COLUMNS:
                stats_dict = summarize_series(cluster_df[value_col])
                for stat_name, stat_value in stats_dict.items():
                    row[f"{value_col}_{stat_name}"] = stat_value
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["normalization", "cluster_id"]).reset_index(drop=True)


def kruskal_analysis(merged_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for normalization, norm_df in merged_df.groupby("normalization"):
        for value_col in VALUE_COLUMNS:
            groups = []
            group_sizes = []
            labels = []
            for cluster_id, cluster_df in sorted(norm_df.groupby("cluster_id"), key=lambda item: item[0]):
                values = cluster_df[value_col].dropna()
                if values.empty:
                    continue
                groups.append(values)
                group_sizes.append(int(values.shape[0]))
                labels.append(f"cluster_{int(cluster_id)}")

            if len(groups) < 2:
                rows.append(
                    {
                        "normalization": normalization,
                        "variable": value_col,
                        "test": "kruskal",
                        "n_groups": len(groups),
                        "group_sizes": ",".join(map(str, group_sizes)),
                        "statistic": math.nan,
                        "p_value": math.nan,
                        "significant_0_05": False,
                        "note": "Not enough groups",
                    }
                )
                continue

            statistic, p_value = stats.kruskal(*groups)
            rows.append(
                {
                    "normalization": normalization,
                    "variable": value_col,
                    "test": "kruskal",
                    "n_groups": len(groups),
                    "group_sizes": ",".join(map(str, group_sizes)),
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "significant_0_05": bool(p_value < 0.05),
                    "note": "|".join(labels),
                }
            )
    return pd.DataFrame(rows)


def pairwise_analysis(merged_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for normalization, norm_df in merged_df.groupby("normalization"):
        cluster_ids = sorted(norm_df["cluster_id"].dropna().unique().tolist())
        for value_col in VALUE_COLUMNS:
            raw_results: list[dict] = []
            for index, cluster_a in enumerate(cluster_ids):
                for cluster_b in cluster_ids[index + 1 :]:
                    values_a = norm_df.loc[norm_df["cluster_id"] == cluster_a, value_col].dropna()
                    values_b = norm_df.loc[norm_df["cluster_id"] == cluster_b, value_col].dropna()
                    if values_a.empty or values_b.empty:
                        continue

                    statistic, p_value = stats.mannwhitneyu(values_a, values_b, alternative="two-sided")
                    median_a = float(values_a.median())
                    median_b = float(values_b.median())
                    raw_results.append(
                        {
                            "normalization": normalization,
                            "variable": value_col,
                            "cluster_a": int(cluster_a),
                            "cluster_b": int(cluster_b),
                            "n_a": int(values_a.shape[0]),
                            "n_b": int(values_b.shape[0]),
                            "median_a": median_a,
                            "median_b": median_b,
                            "median_diff_a_minus_b": median_a - median_b,
                            "u_statistic": float(statistic),
                            "p_value": float(p_value),
                        }
                    )

            if not raw_results:
                continue

            correction = len(raw_results)
            for result in raw_results:
                corrected = min(result["p_value"] * correction, 1.0)
                result["p_value_bonferroni"] = corrected
                result["significant_0_05"] = bool(corrected < 0.05)
                rows.append(result)

    return pd.DataFrame(rows).sort_values(["normalization", "variable", "cluster_a", "cluster_b"]).reset_index(drop=True)


def correlation_analysis(merged_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for normalization, norm_df in merged_df.groupby("normalization"):
        subsets = [("all", norm_df)]
        subsets.extend((f"cluster_{int(cluster_id)}", cluster_df) for cluster_id, cluster_df in norm_df.groupby("cluster_id"))

        for subset_name, subset_df in subsets:
            valid_df = subset_df[["BMI", "TAMA"]].dropna()
            if len(valid_df) < 3:
                rows.append(
                    {
                        "normalization": normalization,
                        "subset": subset_name,
                        "n": int(len(valid_df)),
                        "spearman_rho": math.nan,
                        "p_value": math.nan,
                        "significant_0_05": False,
                    }
                )
                continue

            rho, p_value = stats.spearmanr(valid_df["BMI"], valid_df["TAMA"])
            rows.append(
                {
                    "normalization": normalization,
                    "subset": subset_name,
                    "n": int(len(valid_df)),
                    "spearman_rho": float(rho),
                    "p_value": float(p_value),
                    "significant_0_05": bool(p_value < 0.05),
                }
            )
    return pd.DataFrame(rows).sort_values(["normalization", "subset"]).reset_index(drop=True)


def save_boxplots(merged_df: pd.DataFrame) -> None:
    for normalization, norm_df in merged_df.groupby("normalization"):
        clusters = sorted(norm_df["cluster_id"].unique().tolist())
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for axis_index, value_col in enumerate(["BMI", "TAMA"]):
            data = [
                norm_df.loc[norm_df["cluster_id"] == cluster_id, value_col].dropna().values
                for cluster_id in clusters
            ]
            axes[axis_index].boxplot(data, tick_labels=[f"C{cluster_id}" for cluster_id in clusters], showfliers=True)
            axes[axis_index].set_title(f"{normalization}: {value_col} distribution")
            axes[axis_index].set_xlabel("Cluster")
            axes[axis_index].set_ylabel(value_col)
            axes[axis_index].grid(alpha=0.2, linestyle="--")

        fig.tight_layout()
        fig.savefig(normalization_output_dir(normalization) / "bmi_tama_boxplots.png", dpi=200)
        plt.close(fig)


def save_scatterplots(merged_df: pd.DataFrame) -> None:
    for normalization, norm_df in merged_df.groupby("normalization"):
        fig, ax = plt.subplots(figsize=(7, 6))
        for cluster_id, cluster_df in sorted(norm_df.groupby("cluster_id"), key=lambda item: item[0]):
            ax.scatter(
                cluster_df["BMI"],
                cluster_df["TAMA"],
                label=f"cluster_{int(cluster_id)}",
                alpha=0.65,
                s=20,
            )
        ax.set_title(f"{normalization}: BMI vs TAMA")
        ax.set_xlabel("BMI")
        ax.set_ylabel("TAMA")
        ax.grid(alpha=0.2, linestyle="--")
        ax.legend()
        fig.tight_layout()
        fig.savefig(normalization_output_dir(normalization) / "bmi_tama_scatter.png", dpi=200)
        plt.close(fig)


def build_report(
    source_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    kruskal_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append("# Case1 Cluster BMI/TAMA Analysis")
    lines.append("")
    lines.append(f"- Source excel: `{SOURCE_XLSX}`")
    lines.append(f"- Total source patients with valid BMI and TAMA: {len(source_df)}")
    total_unique_matched = int(merged_df["patient_id"].nunique())
    lines.append(f"- Total matched cluster rows: {len(merged_df)}")
    lines.append(
        "- Unique matched patients: {} / {} ({:.1f}%)".format(
            total_unique_matched,
            len(source_df),
            100.0 * total_unique_matched / len(source_df) if len(source_df) else 0.0,
        )
    )
    lines.append("")

    for normalization in NORMALIZATION_ORDER:
        norm_df = merged_df.loc[merged_df["normalization"] == normalization].copy()
        if norm_df.empty:
            continue

        lines.append(f"## {normalization}")
        lines.append("")
        member_counts = coverage_df.loc[
            coverage_df["normalization"] == normalization,
            ["cluster_label", "member_count", "matched_count", "matched_rate"],
        ]
        counts_text = ", ".join(
            "{}={} total / {} matched ({:.1f}%)".format(
                row.cluster_label,
                int(row.member_count),
                int(row.matched_count),
                float(row.matched_rate),
            )
            for row in member_counts.itertuples()
        )
        lines.append(f"- Cluster coverage: {counts_text}")
        lines.append(
            "- Normalization-wide matched patients: {} / {} ({:.1f}%)".format(
                int(norm_df["patient_id"].nunique()),
                len(source_df),
                100.0 * norm_df["patient_id"].nunique() / len(source_df) if len(source_df) else 0.0,
            )
        )

        overall_bmi = summarize_series(norm_df["BMI"])
        overall_tama = summarize_series(norm_df["TAMA"])
        lines.append(
            "- Overall BMI median {:.2f} (IQR {:.2f}-{:.2f}), TAMA median {:.2f} (IQR {:.2f}-{:.2f})".format(
                overall_bmi["median"],
                overall_bmi["q1"],
                overall_bmi["q3"],
                overall_tama["median"],
                overall_tama["q1"],
                overall_tama["q3"],
            )
        )

        for value_col in ["BMI", "TAMA"]:
            sub_summary = summary_df.loc[summary_df["normalization"] == normalization].copy()
            median_col = f"{value_col}_median"
            top_row = sub_summary.sort_values(median_col, ascending=False).iloc[0]
            bottom_row = sub_summary.sort_values(median_col, ascending=True).iloc[0]
            lines.append(
                "- {} median highest: {} ({:.2f}), lowest: {} ({:.2f})".format(
                    value_col,
                    top_row["cluster_label"],
                    top_row[median_col],
                    bottom_row["cluster_label"],
                    bottom_row[median_col],
                )
            )

        kruskal_rows = kruskal_df.loc[kruskal_df["normalization"] == normalization]
        for row in kruskal_rows.itertuples():
            verdict = "significant" if row.significant_0_05 else "not significant"
            lines.append(
                f"- Kruskal-Wallis {row.variable}: H={row.statistic:.3f}, p={row.p_value:.4g} ({verdict})"
            )

        sig_pairs = pairwise_df.loc[
            (pairwise_df["normalization"] == normalization) & (pairwise_df["significant_0_05"])
        ].copy()
        if sig_pairs.empty:
            lines.append("- Pairwise cluster differences after Bonferroni correction: none")
        else:
            top_pairs = sig_pairs.sort_values("p_value_bonferroni").head(5)
            for row in top_pairs.itertuples():
                lines.append(
                    "- Pairwise {}: cluster_{} vs cluster_{} | median diff={:.2f} | corrected p={:.4g}".format(
                        row.variable,
                        row.cluster_a,
                        row.cluster_b,
                        row.median_diff_a_minus_b,
                        row.p_value_bonferroni,
                    )
                )

        corr_rows = corr_df.loc[corr_df["normalization"] == normalization].copy()
        overall_corr = corr_rows.loc[corr_rows["subset"] == "all"].iloc[0]
        if pd.notna(overall_corr["spearman_rho"]):
            corr_verdict = "significant" if overall_corr["significant_0_05"] else "not significant"
            lines.append(
                "- Overall BMI-TAMA Spearman rho={:.3f}, p={:.4g} ({})".format(
                    overall_corr["spearman_rho"],
                    overall_corr["p_value"],
                    corr_verdict,
                )
            )
        else:
            lines.append("- Overall BMI-TAMA Spearman correlation: not enough data")

        lines.append("")

    return "\n".join(lines)


def main() -> None:
    ensure_output_dir()
    source_df = load_source_dataframe()

    all_members = []
    all_coverage = []
    unmatched_rows = []

    for normalization in NORMALIZATION_ORDER:
        members_df, coverage_df = load_cluster_members(normalization)
        merged_df = members_df.merge(source_df, on="patient_id", how="left", validate="many_to_one")

        unmatched_df = merged_df.loc[merged_df["BMI"].isna() | merged_df["TAMA"].isna()].copy()
        if not unmatched_df.empty:
            unmatched_rows.append(unmatched_df)

        merged_df = merged_df.dropna(subset=["BMI", "TAMA"]).copy()
        matched_counts = (
            merged_df.groupby("cluster_id")["patient_id"]
            .size()
            .rename("matched_count")
            .reset_index()
        )
        coverage_df = coverage_df.merge(matched_counts, on="cluster_id", how="left")
        coverage_df["matched_count"] = coverage_df["matched_count"].fillna(0).astype(int)
        coverage_df["unmatched_count"] = coverage_df["member_count"] - coverage_df["matched_count"]
        coverage_df["matched_rate"] = (
            coverage_df["matched_count"] / coverage_df["member_count"] * 100.0
        ).round(1)

        all_members.append(merged_df)
        all_coverage.append(coverage_df)

    combined_df = pd.concat(all_members, ignore_index=True)
    coverage_df = pd.concat(all_coverage, ignore_index=True)

    summary_df = build_summary(combined_df)
    kruskal_df = kruskal_analysis(combined_df)
    pairwise_df = pairwise_analysis(combined_df)
    corr_df = correlation_analysis(combined_df)

    combined_df.to_csv(COMMON_OUTPUT_DIR / "case1_cluster_members_with_bmi_tama.csv", index=False, encoding="utf-8-sig")
    coverage_df.to_csv(COMMON_OUTPUT_DIR / "cluster_member_counts.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(COMMON_OUTPUT_DIR / "cluster_summary_bmi_tama.csv", index=False, encoding="utf-8-sig")
    kruskal_df.to_csv(COMMON_OUTPUT_DIR / "kruskal_tests.csv", index=False, encoding="utf-8-sig")
    pairwise_df.to_csv(COMMON_OUTPUT_DIR / "pairwise_mannwhitney_bonferroni.csv", index=False, encoding="utf-8-sig")
    corr_df.to_csv(COMMON_OUTPUT_DIR / "bmi_tama_correlations.csv", index=False, encoding="utf-8-sig")

    for normalization in NORMALIZATION_ORDER:
        norm_dir = normalization_output_dir(normalization)
        norm_members = combined_df.loc[combined_df["normalization"] == normalization].copy()
        if not norm_members.empty:
            norm_members.to_csv(norm_dir / "cluster_members_with_bmi_tama.csv", index=False, encoding="utf-8-sig")

        norm_coverage = coverage_df.loc[coverage_df["normalization"] == normalization].copy()
        if not norm_coverage.empty:
            norm_coverage.to_csv(norm_dir / "cluster_member_counts.csv", index=False, encoding="utf-8-sig")

        norm_summary = summary_df.loc[summary_df["normalization"] == normalization].copy()
        if not norm_summary.empty:
            norm_summary.to_csv(norm_dir / "cluster_summary_bmi_tama.csv", index=False, encoding="utf-8-sig")

        norm_kruskal = kruskal_df.loc[kruskal_df["normalization"] == normalization].copy()
        if not norm_kruskal.empty:
            norm_kruskal.to_csv(norm_dir / "kruskal_tests.csv", index=False, encoding="utf-8-sig")

        norm_pairwise = pairwise_df.loc[pairwise_df["normalization"] == normalization].copy()
        if not norm_pairwise.empty:
            norm_pairwise.to_csv(norm_dir / "pairwise_mannwhitney_bonferroni.csv", index=False, encoding="utf-8-sig")

        norm_corr = corr_df.loc[corr_df["normalization"] == normalization].copy()
        if not norm_corr.empty:
            norm_corr.to_csv(norm_dir / "bmi_tama_correlations.csv", index=False, encoding="utf-8-sig")

    if unmatched_rows:
        pd.concat(unmatched_rows, ignore_index=True).to_csv(
            COMMON_OUTPUT_DIR / "unmatched_cluster_members.csv",
            index=False,
            encoding="utf-8-sig",
        )

    save_boxplots(combined_df)
    save_scatterplots(combined_df)

    report_text = build_report(
        source_df=source_df,
        merged_df=combined_df,
        summary_df=summary_df,
        kruskal_df=kruskal_df,
        pairwise_df=pairwise_df,
        corr_df=corr_df,
        coverage_df=coverage_df,
    )
    (COMMON_OUTPUT_DIR / "analysis_report.md").write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"\nSaved normalization outputs under: {CASE1_DIR / '<normalization>' / 'tama_bmi_analysis'}")


if __name__ == "__main__":
    main()
