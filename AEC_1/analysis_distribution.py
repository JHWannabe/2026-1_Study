from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pydicom
from pydicom.errors import InvalidDicomError
from datetime import datetime


ROOT_DIR = Path(r"D:\데이터서비스팀 영상제공\신촌_원본")
OUTPUT_DIR = Path(r"D:\데이터서비스팀 영상제공\신촌_결과")
AGE_OUTPUT_PATH = OUTPUT_DIR / "shinchan_age_distribution.png"
SEX_OUTPUT_PATH = OUTPUT_DIR / "shinchan_sex_distribution.png"


def to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_age(value: object) -> Optional[float]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None

    if raw.endswith(("Y", "M", "W", "D")) and len(raw) >= 2:
        unit = raw[-1]
        number = to_float(raw[:-1])
        if number is None:
            return None
        if unit == "Y":
            return number
        if unit == "M":
            return number / 12.0
        if unit == "W":
            return number / 52.0
        if unit == "D":
            return number / 365.0
    return to_float(raw)


def parse_dicom_date(value: object) -> Optional[datetime]:
    if value is None:
        return None

    raw = str(value).strip()
    if not raw:
        return None

    try:
        return datetime.strptime(raw, "%Y%m%d")
    except ValueError:
        return None


def calculate_age_from_dates(ds: pydicom.Dataset) -> Optional[float]:
    birth_date = parse_dicom_date(getattr(ds, "PatientBirthDate", None))
    if birth_date is None:
        return None

    study_date = parse_dicom_date(getattr(ds, "StudyDate", None))
    series_date = parse_dicom_date(getattr(ds, "SeriesDate", None))
    acquisition_date = parse_dicom_date(getattr(ds, "AcquisitionDate", None))

    ref_date = study_date or series_date or acquisition_date
    if ref_date is None:
        return None

    age_years = (ref_date - birth_date).days / 365.25
    if age_years < 0:
        return None

    return age_years


def find_one_dicom_file(folder: Path) -> Optional[Path]:
    for pattern in ("*.dcm", "*.DCM"):
        found = next(folder.rglob(pattern), None)
        if found:
            return found

    for file_path in folder.rglob("*"):
        if not file_path.is_file():
            continue
        try:
            pydicom.dcmread(file_path, stop_before_pixels=True)
            return file_path
        except (InvalidDicomError, PermissionError, OSError):
            continue
    return None



def extract_row(dicom_path: Path) -> dict[str, object]:
    ds = pydicom.dcmread(dicom_path, stop_before_pixels=True, force=True)

    raw_age = getattr(ds, "PatientAge", None)
    age = parse_age(raw_age)

    # PatientAge가 없으면 BirthDate + 촬영일로 계산
    if age is None:
        age = calculate_age_from_dates(ds)

    sex = str(getattr(ds, "PatientSex", "Unknown")).strip().upper()

    return {"age": age, "sex": sex}


def _apply_log_yaxis(ax: plt.Axes, max_count: int) -> None:
    ax.set_yscale("log")
    if max_count > 0:
        ax.set_ylim(bottom=0.8, top=max_count * 1.2 + 1)


def _add_numeric_summary(ax: plt.Axes, values: list[float], title_prefix: str) -> None:
    if not values:
        text = f"{title_prefix}\nN=0"
    else:
        sorted_values = sorted(values)
        n = len(sorted_values)
        mean_val = sum(sorted_values) / n
        if n % 2 == 1:
            median_val = sorted_values[n // 2]
        else:
            median_val = (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2.0
        text = (
            f"{title_prefix}\n"
            f"N={n}\n"
            f"Mean={mean_val:.2f}\n"
            f"Median={median_val:.2f}"
        )
    ax.text(
        0.98,
        0.97,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="#666666"),
    )


def plot_age_distribution(rows: list[dict[str, object]]) -> Path:
    ages = [v for v in (row["age"] for row in rows) if isinstance(v, (int, float))]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))

    age_counts, _, _ = ax.hist(ages, bins=20, color="#4C78A8", edgecolor="black")

    ax.set_title("Patient Age")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Count (log scale)")
    ax.grid(alpha=0.2)

    _apply_log_yaxis(ax, int(max(age_counts)) if len(age_counts) else 0)
    _add_numeric_summary(ax, ages, "Age Stats")

    plt.tight_layout()
    fig.savefig(AGE_OUTPUT_PATH, dpi=200)
    plt.close(fig)

    return AGE_OUTPUT_PATH


def plot_sex_distribution(rows: list[dict[str, object]]) -> Path:
    sexes = [str(row["sex"]) for row in rows if str(row["sex"]) in {"M", "F"}]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))

    sex_count = Counter(sexes)

    labels = ["M", "F"]
    values = [sex_count.get(label, 0) for label in labels]

    bars = ax.bar(
        labels,
        values,
        color=["#54A24B", "#E45756"],
        edgecolor="black",
    )

    ax.set_title("Patient Sex")
    ax.set_xlabel("Sex")
    ax.set_ylabel("Count (log scale)")
    ax.grid(axis="y", alpha=0.2)

    _apply_log_yaxis(ax, max(values) + 500 if values else 0)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            max(value, 1) * 1.05,
            str(value),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(SEX_OUTPUT_PATH, dpi=200)
    plt.close(fig)

    return SEX_OUTPUT_PATH


def main() -> None:
    if not ROOT_DIR.exists() or not ROOT_DIR.is_dir():
        print(f"Invalid directory: {ROOT_DIR}")
        return

    rows: list[dict[str, object]] = []
    subdirs = sorted([p for p in ROOT_DIR.iterdir() if p.is_dir()])
    if not subdirs:
        print(f"No subfolders found in: {ROOT_DIR}")
        return

    for folder in subdirs:
        dicom_path = find_one_dicom_file(folder)
        if dicom_path is None:
            print(f"[Skip] No DICOM file in: {folder}")
            continue
        try:
            rows.append(extract_row(dicom_path))
        except Exception as exc:
            print(f"[Skip] Failed to read {dicom_path}: {exc}")

    if not rows:
        print("No metadata collected. Nothing to plot.")
        return

    age_path = plot_age_distribution(rows)
    sex_path = plot_sex_distribution(rows)

    print(f"Saved age plot: {age_path}")
    print(f"Saved sex plot: {sex_path}")
    print(f"Processed folders: {len(subdirs)}")
    print(f"Samples used: {len(rows)}")


if __name__ == "__main__":
    main()
