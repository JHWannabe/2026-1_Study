import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SEX_FILTER_LABELS = {
    "F": "female",
    "M": "male",
}
AGE_GROUP_BINS: dict[str, tuple[int, int]] = {
    "40-": (0, 40),
    "40s": (40, 50),
    "50s": (50, 60),
    "60s": (60, 70),
    "70+": (70, 200),
}
AGE_COL = "PatientAge"
SITE_NAME = os.environ.get("AEC_SITE_NAME", "신촌")
REQUESTED_SEX_FILTER = os.environ.get("AEC_SEX_FILTER", "").strip().upper()
REQUESTED_AGE_GROUP = os.environ.get("AEC_AGE_GROUP", "").strip()
SEX_FILTER = ""
SEX_FILTER_DIR = ""
AGE_GROUP = ""

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "AEC" / SITE_NAME / "Result_Filter.xlsx"
RESULT_DIR = Path()

OUTPUT_PATH = Path()

REQUIRED_COLUMNS = ["PatientID", "PatientSex", "SRC_Report", "BMI"]
SEX_LABELS = {"F": "Female", "M": "Male"}
QUADRANT_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")

BMI_REFERENCE = 25.0
OVERALL_TAMA_REFERENCE = "mean"
BY_SEX_TAMA_REFERENCE = "mean"
FONT_SCALE = 1.3
LEGEND_FONT_SIZE = 8


@dataclass(frozen=True)
class ReferencePoint:
    tama: float
    bmi: float


def resolve_requested_sex_filters() -> list[str]:
    if not REQUESTED_SEX_FILTER:
        return list(SEX_FILTER_LABELS)
    if REQUESTED_SEX_FILTER not in SEX_FILTER_LABELS:
        valid = ", ".join(SEX_FILTER_LABELS)
        raise ValueError(
            f"Unsupported AEC_SEX_FILTER '{REQUESTED_SEX_FILTER}'. Expected one of: {valid}"
        )
    return [REQUESTED_SEX_FILTER]


def configure_sex_filter(sex_filter: str) -> None:
    global SEX_FILTER, SEX_FILTER_DIR, RESULT_DIR, OUTPUT_PATH, AGE_GROUP

    SEX_FILTER = sex_filter
    SEX_FILTER_DIR = SEX_FILTER_LABELS[sex_filter]
    AGE_GROUP = ""
    RESULT_DIR = BASE_DIR / "result" / "연령별" / SITE_NAME / SEX_FILTER_DIR
    OUTPUT_PATH = RESULT_DIR / "bmi_tama_scatter.png"


def configure_age_group(age_group: str) -> None:
    global SEX_FILTER, SEX_FILTER_DIR, RESULT_DIR, OUTPUT_PATH, AGE_GROUP

    if age_group not in AGE_GROUP_BINS:
        valid = ", ".join(AGE_GROUP_BINS)
        raise ValueError(f"Unsupported AEC_AGE_GROUP '{age_group}'. Expected one of: {valid}")
    AGE_GROUP = age_group
    SEX_FILTER = ""
    SEX_FILTER_DIR = age_group
    RESULT_DIR = BASE_DIR / "result" / "연령별" / SITE_NAME / SEX_FILTER_DIR
    OUTPUT_PATH = RESULT_DIR / "bmi_tama_scatter.png"


# Configuration helpers
def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10 * FONT_SCALE,
            "axes.titlesize": 14 * FONT_SCALE,
            "axes.labelsize": 12 * FONT_SCALE,
            "xtick.labelsize": 10 * FONT_SCALE,
            "ytick.labelsize": 10 * FONT_SCALE,
            "legend.fontsize": 11 * FONT_SCALE,
        }
    )


# Data preparation helpers
def normalize_patient_sex(series: pd.Series) -> pd.Series:
    return (
        series.astype(str).str.strip().str.upper().replace({"MALE": "M", "FEMALE": "F"})
    )


def normalize_patient_id(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def apply_sex_filter(df: pd.DataFrame) -> pd.DataFrame:
    if not SEX_FILTER:
        return df
    filtered_df = df[df["PatientSex"] == SEX_FILTER].copy()
    if filtered_df.empty:
        raise ValueError(
            f"No rows found for site '{SITE_NAME}' with sex filter '{SEX_FILTER_DIR}'."
        )
    return filtered_df


def apply_age_group_filter(df: pd.DataFrame) -> pd.DataFrame:
    if not AGE_GROUP:
        return df
    if AGE_COL not in df.columns:
        raise ValueError(f"Column '{AGE_COL}' not found. Cannot apply age group filter.")
    min_age, max_age = AGE_GROUP_BINS[AGE_GROUP]
    filtered_df = df[(df[AGE_COL] >= min_age) & (df[AGE_COL] < max_age)].copy()
    if filtered_df.empty:
        raise ValueError(
            f"No rows found for site '{SITE_NAME}' with age group '{AGE_GROUP}'."
        )
    return filtered_df


def load_bmi_tama_data(file_path: Path) -> pd.DataFrame:
    usecols = REQUIRED_COLUMNS.copy()
    df_all = pd.read_excel(file_path).copy()
    if AGE_COL in df_all.columns:
        usecols = usecols + [AGE_COL]
    df = df_all[usecols].copy()
    df = df.rename(columns={"SRC_Report": "TAMA"})
    df["TAMA"] = pd.to_numeric(df["TAMA"], errors="coerce")
    df["BMI"] = pd.to_numeric(df["BMI"], errors="coerce")
    df["PatientSex"] = normalize_patient_sex(df["PatientSex"])
    df["PatientID"] = df["PatientID"].apply(normalize_patient_id)
    df = df[df["PatientID"] != ""]
    df = df.dropna(subset=["PatientID", "TAMA", "BMI"]).copy()
    if AGE_GROUP and AGE_COL in df.columns:
        df[AGE_COL] = pd.to_numeric(df[AGE_COL], errors="coerce")
    df = apply_sex_filter(df)
    return apply_age_group_filter(df)


def get_tama_reference(df: pd.DataFrame, strategy: str) -> float:
    strategy_map = {
        "mean": df["TAMA"].mean,
        "median": df["TAMA"].median,
    }
    if strategy not in strategy_map:
        raise ValueError(f"Unsupported TAMA reference strategy: {strategy}")
    return float(strategy_map[strategy]())


def build_reference_point(df: pd.DataFrame, tama_strategy: str) -> ReferencePoint:
    return ReferencePoint(
        tama=get_tama_reference(df, tama_strategy),
        bmi=BMI_REFERENCE,
    )


def build_quadrants(
    df: pd.DataFrame, reference: ReferencePoint
) -> list[dict[str, object]]:
    return [
        {
            "label": f"TAMA < {reference.tama:g}, BMI < {reference.bmi:g}",
            "mask": (df["TAMA"] < reference.tama) & (df["BMI"] < reference.bmi),
            "color": QUADRANT_COLORS[0],
        },
        {
            "label": f"TAMA < {reference.tama:g}, BMI >= {reference.bmi:g}",
            "mask": (df["TAMA"] < reference.tama) & (df["BMI"] >= reference.bmi),
            "color": QUADRANT_COLORS[1],
        },
        {
            "label": f"TAMA >= {reference.tama:g}, BMI < {reference.bmi:g}",
            "mask": (df["TAMA"] >= reference.tama) & (df["BMI"] < reference.bmi),
            "color": QUADRANT_COLORS[2],
        },
        {
            "label": f"TAMA >= {reference.tama:g}, BMI >= {reference.bmi:g}",
            "mask": (df["TAMA"] >= reference.tama) & (df["BMI"] >= reference.bmi),
            "color": QUADRANT_COLORS[3],
        },
    ]


# Plotting helpers
def draw_scatter(
    ax: plt.Axes, df: pd.DataFrame, reference: ReferencePoint, title: str
) -> None:
    for quadrant in build_quadrants(df, reference):
        quadrant_df = df.loc[quadrant["mask"]]
        ax.scatter(
            quadrant_df["TAMA"],
            quadrant_df["BMI"],
            alpha=0.75,
            color=quadrant["color"],
            label=f"{quadrant['label']} (n={len(quadrant_df)})",
        )

    ax.axvline(reference.tama, color="gray", linestyle="--", linewidth=1.2)
    ax.axhline(reference.bmi, color="gray", linestyle="--", linewidth=1.2)
    ax.set_xlabel("TAMA")
    ax.set_ylabel("BMI")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)


def save_figure(
    fig: plt.Figure,
    output_path: Path,
    tight_layout_rect: tuple[float, float, float, float] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if tight_layout_rect is None:
        fig.tight_layout()
    else:
        fig.tight_layout(rect=tight_layout_rect)
    fig.savefig(output_path, dpi=300)

    if "agg" not in plt.get_backend().lower():
        plt.show()

    plt.close(fig)


# Output helpers
def save_scatter_plot(
    df: pd.DataFrame,
    output_path: Path,
    title: str,
    tama_reference_strategy: str,
) -> None:
    reference = build_reference_point(df, tama_reference_strategy)
    fig, ax = plt.subplots(figsize=(8, 6))
    draw_scatter(ax, df, reference, title)
    ax.legend(loc="upper left", fontsize=LEGEND_FONT_SIZE)
    save_figure(fig, output_path)


def save_scatter_plot_by_sex(
    df: pd.DataFrame,
    output_path: Path,
    tama_reference_strategy: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    for ax, sex_code in zip(axes, SEX_LABELS):
        sex_name = SEX_LABELS[sex_code]
        sex_df = df[df["PatientSex"] == sex_code].copy()
        ax.tick_params(axis="y", labelleft=True)

        if sex_df.empty:
            ax.text(
                0.5,
                0.5,
                f"No {sex_name} data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"{sex_name} (n=0)")
            ax.set_xlabel("TAMA")
            ax.set_ylabel("BMI")
            ax.grid(True, linestyle="--", alpha=0.4)
            continue

        reference = build_reference_point(sex_df, tama_reference_strategy)
        draw_scatter(ax, sex_df, reference, f"{sex_name} (n={len(sex_df)})")
        ax.legend(loc="upper left", fontsize=LEGEND_FONT_SIZE)

    save_figure(fig, output_path, tight_layout_rect=(0, 0, 1, 0.96))


# Entry point
def run_single_sex_filter() -> None:
    configure_plot_style()
    df = load_bmi_tama_data(DATA_PATH)

    save_scatter_plot(
        df,
        output_path=OUTPUT_PATH,
        title="BMI vs TAMA",
        tama_reference_strategy=OVERALL_TAMA_REFERENCE,
    )

    print(f"Loaded {len(df)} rows from {DATA_PATH}")
    print(f"Sex filter: {SEX_FILTER_DIR}")
    print(f"Scatter plot saved to {OUTPUT_PATH}")


def main() -> None:
    if REQUESTED_AGE_GROUP:
        configure_age_group(REQUESTED_AGE_GROUP)
        run_single_sex_filter()
    else:
        for age_group in AGE_GROUP_BINS:
            configure_age_group(age_group)
            run_single_sex_filter()


if __name__ == "__main__":
    main()
