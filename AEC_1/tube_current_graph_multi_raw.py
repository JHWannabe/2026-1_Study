from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict

import re
import pydicom
import matplotlib.pyplot as plt
from tqdm import tqdm


def sanitize_filename(text: str) -> str:
    # 파일명/폴더명에 사용할 수 없는 문자 제거
    if text is None:
        return "UNKNOWN"
    text = str(text).strip()
    text = re.sub(r'[\\/:*?"<>|]+', "_", text)
    text = re.sub(r"\s+", "_", text)
    return text if text else "UNKNOWN"


def get_position_value(ds) -> Optional[float]:
    """
    x축에 사용할 위치값 추출
    우선순위:
    1) SliceLocation
    2) TablePosition
    3) ImagePositionPatient의 z값
    4) 없으면 None
    """
    val = getattr(ds, "SliceLocation", None)
    if val is not None:
        try:
            return float(val)
        except Exception:
            pass

    val = getattr(ds, "TablePosition", None)
    if val is not None:
        try:
            return float(val)
        except Exception:
            pass

    val = getattr(ds, "ImagePositionPatient", None)
    if val is not None:
        try:
            if len(val) >= 3:
                return float(val[2])
        except Exception:
            pass

    return None


def extract_non_cor_ct_series(
    dicom_dir: str,
    recursive: bool = True,
) -> Dict[str, List[Dict]]:
    """
    폴더 내 DICOM 파일 중에서
    - Modality == CT
    - SeriesDescription에 'COR' 미포함
    인 series들을 모두 추출

    반환:
        {
            series_uid1: [record1, record2, ...],
            series_uid2: [record1, record2, ...],
            ...
        }
    """
    dicom_dir = Path(dicom_dir)
    if not dicom_dir.exists():
        raise FileNotFoundError(f"폴더가 존재하지 않습니다: {dicom_dir}")

    files = list(dicom_dir.rglob("*")) if recursive else list(dicom_dir.iterdir())
    files = [p for p in files if p.is_file()]

    if len(files) == 0:
        raise FileNotFoundError(f"DICOM 파일이 없습니다: {dicom_dir}")

    grouped_records = defaultdict(list)

    for fp in files:
        try:
            ds = pydicom.dcmread(str(fp), stop_before_pixels=True, force=True)
        except Exception:
            continue

        modality = getattr(ds, "Modality", None)
        if modality != "CT":
            continue

        series_desc = str(getattr(ds, "SeriesDescription", "")).strip()
        series_desc_norm = series_desc.lower()

        # COR 포함 series 제외
        if "cor" in series_desc_norm:
            continue

        instance_number = getattr(ds, "InstanceNumber", None)
        if instance_number is None:
            continue

        try:
            instance_number = int(instance_number)
        except Exception:
            continue

        tube_current = getattr(ds, "XRayTubeCurrent", None)
        try:
            tube_current = float(tube_current) if tube_current is not None else None
        except Exception:
            tube_current = None

        position_value = get_position_value(ds)

        series_uid = str(getattr(ds, "SeriesInstanceUID", "")).strip()
        if not series_uid:
            series_uid = f"NO_UID::{fp.parent}"

        patient_id = getattr(ds, "PatientID", None)

        grouped_records[series_uid].append({
            "file": str(fp),
            "instance_number": instance_number,
            "position_value": position_value,
            "tube_current_mA": tube_current,
            "series_uid": series_uid,
            "series_description": series_desc,
            "patient_id": str(patient_id) if patient_id is not None else "",
        })

    if len(grouped_records) == 0:
        raise ValueError("SeriesDescription에 'COR'가 포함되지 않은 CT series를 찾지 못했습니다.")

    return grouped_records


def sort_records(records: List[Dict]) -> List[Dict]:
    """
    가능하면 position_value 기준 정렬,
    없으면 instance_number 기준 정렬
    """
    n_pos = sum(r.get("position_value") is not None for r in records)

    if n_pos > 0:
        return sorted(
            records,
            key=lambda x: (
                x.get("position_value") is None,
                x.get("position_value") if x.get("position_value") is not None else float("inf"),
                x["instance_number"],
            )
        )

    return sorted(records, key=lambda x: x["instance_number"])


def is_flat_tube_current(records: List[Dict]) -> bool:
    """
    Tube current가 변동 없는 직선인지 확인
    """
    valid = [r for r in records if r["tube_current_mA"] is not None]

    if len(valid) == 0:
        return True

    y_vals = [r["tube_current_mA"] for r in valid]
    return max(y_vals) - min(y_vals) == 0


def plot_tube_current_vs_position(
    records: List[Dict],
    title: str = "Tube Current vs Position",
    show_markers: bool = True,
    save_path: Optional[str] = None,
):
    """
    우선 position_value를 x축으로 사용하고,
    position_value가 부족하면 instance_number를 x축으로 사용
    """
    valid = [r for r in records if r["tube_current_mA"] is not None]

    if len(valid) == 0:
        raise ValueError("XRayTubeCurrent 값이 있는 슬라이스가 없습니다.")

    valid_pos = [r for r in valid if r.get("position_value") is not None]

    if len(valid_pos) >= 2:
        x_vals = [r["position_value"] for r in valid_pos]
        y_vals = [r["tube_current_mA"] for r in valid_pos]
        x_label = "Position (SliceLocation / TablePosition / ImagePositionPatient[z])"
    else:
        x_vals = [r["instance_number"] for r in valid]
        y_vals = [r["tube_current_mA"] for r in valid]
        x_label = "Instance Number"

    plt.figure(figsize=(12, 5))

    if show_markers:
        plt.plot(x_vals, y_vals, marker="o", markersize=3, linewidth=1)
    else:
        plt.plot(x_vals, y_vals, linewidth=1)

    plt.xlabel(x_label)
    plt.ylabel("X-ray Tube Current (mA)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.close()


def find_patient_dirs(root_dir: str) -> List[Path]:
    """
    root_dir 바로 아래의 하위 폴더들을 환자 폴더로 간주
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"루트 폴더가 존재하지 않습니다: {root}")

    patient_dirs = [p for p in root.iterdir() if p.is_dir()]
    patient_dirs = sorted(patient_dirs)

    if len(patient_dirs) == 0:
        raise ValueError(f"하위 환자 폴더가 없습니다: {root}")

    return patient_dirs


def process_one_patient(
    patient_dir: Path,
    out_dir: Path,
    recursive: bool = True,
) -> int:
    """
    환자 폴더 하나 처리
    반환: 저장된 그래프 개수
    """
    series_dict = extract_non_cor_ct_series(
        dicom_dir=str(patient_dir),
        recursive=recursive,
    )

    saved_count = 0

    items = sorted(
        series_dict.items(),
        key=lambda kv: (
            kv[1][0].get("series_description", ""),
            kv[0]
        )
    )

    for series_idx, (series_uid, records) in enumerate(items, start=1):
        records = sort_records(records)

        # Tube current 변동 없으면 skip
        if is_flat_tube_current(records):
            continue

        patient_id = records[0].get("patient_id", "").strip()
        if not patient_id:
            patient_id = patient_dir.name

        series_desc = records[0].get("series_description", "").strip()
        if not series_desc:
            series_desc = "UNKNOWN_SERIES"

        patient_id_safe = sanitize_filename(patient_id)
        series_desc_safe = sanitize_filename(series_desc)

        # series description별 폴더 생성
        series_out_dir = out_dir / series_desc_safe
        series_out_dir.mkdir(parents=True, exist_ok=True)

        file_name = f"{patient_id_safe}_{series_desc_safe}_{series_idx}_Tube_plot.png"
        save_path = series_out_dir / file_name

        # 이미 파일이 있으면 skip
        if save_path.exists():
            continue

        title = f"{patient_id} | {series_desc} | Tube Current vs Position"

        try:
            plot_tube_current_vs_position(
                records,
                title=title,
                show_markers=False,
                save_path=str(save_path),
            )
            saved_count += 1
        except Exception as e:
            print(f"[PLOT SKIP] {patient_dir.name} | {series_desc} | {series_uid} | {e}")
            continue

    return saved_count


def batch_process_patients(
    root_dir: str,
    out_dir: str,
    recursive: bool = True,
):
    root_dir = Path(root_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    num = 0
    patient_dirs = find_patient_dirs(str(root_dir))[num:]

    success_patient_count = 0
    fail_patient_count = 0
    total_plot_count = 0

    pbar = tqdm(patient_dirs, desc="Processing Patients", unit="patient")

    for idx, patient_dir in enumerate(pbar, start=num):
        pbar.set_postfix(patient=patient_dir.name)

        try:
            saved_count = process_one_patient(
                patient_dir=patient_dir,
                out_dir=out_dir,
                recursive=recursive,
            )

            if saved_count > 0:
                success_patient_count += 1
                total_plot_count += saved_count
            else:
                fail_patient_count += 1

        except Exception as e:
            print(f"[PATIENT FAIL] {patient_dir.name}: {e}")
            fail_patient_count += 1

    return {
        "success_patient_count": success_patient_count,
        "fail_patient_count": fail_patient_count,
        "total_plot_count": total_plot_count,
        "total_patient_count": len(patient_dirs),
    }


if __name__ == "__main__":
    root_dir = r"D:\데이터서비스팀 영상제공\신촌_원본"
    out_dir = r"D:\데이터서비스팀 영상제공\신촌_plots_raws"

    result = batch_process_patients(
        root_dir=root_dir,
        out_dir=out_dir,
        recursive=True,
    )

    print("\n=== RESULT ===")
    for k, v in result.items():
        print(f"{k}: {v}")