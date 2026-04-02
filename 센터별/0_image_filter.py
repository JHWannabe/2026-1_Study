import os
import shutil
from pathlib import Path

import pandas as pd

SITE_NAME = os.environ.get("AEC_SITE_NAME", "신촌")
BASE_DIR = Path(__file__).resolve().parents[2]
SITE_DIR = BASE_DIR / "data" / "AEC" / SITE_NAME
EXCEL_PATH = SITE_DIR / "Result_Filter.xlsx"
IMAGE_DIR = SITE_DIR / "raw" / "Image"
OUTPUT_DIR = SITE_DIR / "Image_Filter"
FILTERED_EXCEL_PATH = EXCEL_PATH.with_name(
    f"{EXCEL_PATH.stem}_filtered{EXCEL_PATH.suffix}"
)


# Data preparation helpers
def normalize_patient_id(value) -> str:
    if pd.isna(value):
        return ""

    if isinstance(value, str):
        text = value.strip()
        if text.endswith(".0") and text[:-2].isdigit():
            return text[:-2]
        return text

    if isinstance(value, int):
        return str(value)

    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value).strip()

    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def load_patient_ids(excel_path: Path) -> list[str]:
    df = pd.read_excel(excel_path)
    if "PatientID" not in df.columns:
        raise ValueError(f"'PatientID' column not found in: {excel_path}")

    patient_ids = []
    seen = set()

    for value in df["PatientID"]:
        patient_id = normalize_patient_id(value)
        if not patient_id:
            continue
        if patient_id in seen:
            continue
        seen.add(patient_id)
        patient_ids.append(patient_id)

    return patient_ids


def copy_matching_pngs(
    patient_ids: list[str], image_dir: Path, output_dir: Path
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    missing_ids = []
    copied_count = 0

    for patient_id in patient_ids:
        src_path = image_dir / f"{patient_id}.png"
        dst_path = output_dir / src_path.name

        if not src_path.exists():
            missing_ids.append(patient_id)
            continue

        shutil.copy2(src_path, dst_path)
        copied_count += 1

    print(f"Total PatientID count: {len(patient_ids)}")
    print(f"Copied PNG count: {copied_count}")
    print(f"Missing PNG count: {len(missing_ids)}")
    return missing_ids


# Output helpers
def save_missing_report(missing_ids: list[str], output_dir: Path) -> None:
    report_path = output_dir / "missing_patient_ids.csv"
    pd.DataFrame({"PatientID": missing_ids}).to_csv(
        report_path,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Missing ID report saved to: {report_path}")


def remove_missing_ids_from_excel(
    excel_path: Path,
    missing_ids: list[str],
    save_path: Path,
) -> None:
    df = pd.read_excel(excel_path)

    if "PatientID" not in df.columns:
        raise ValueError(f"'PatientID' column not found in: {excel_path}")

    df = df.copy()
    df["_normalized_patient_id"] = df["PatientID"].apply(normalize_patient_id)

    missing_id_set = set(missing_ids)

    before_count = len(df)
    filtered_df = df[~df["_normalized_patient_id"].isin(missing_id_set)].copy()
    removed_count = before_count - len(filtered_df)

    filtered_df.drop(columns=["_normalized_patient_id"], inplace=True)

    filtered_df.to_excel(save_path, index=False)
    print(f"Removed rows count from Excel: {removed_count}")
    print(f"Filtered Excel saved to: {save_path}")


# Entry point
def main() -> None:
    patient_ids = load_patient_ids(EXCEL_PATH)
    missing_ids = copy_matching_pngs(patient_ids, IMAGE_DIR, OUTPUT_DIR)
    save_missing_report(missing_ids, OUTPUT_DIR)

    if missing_ids:
        remove_missing_ids_from_excel(
            excel_path=EXCEL_PATH,
            missing_ids=missing_ids,
            save_path=FILTERED_EXCEL_PATH,
        )
    else:
        print(
            "No missing PatientID found. Filtered Excel was not created because no rows need removal."
        )

    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
