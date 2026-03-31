from pathlib import Path
from collections import defaultdict
import pydicom


ROOT_DIR = Path(r"D:\데이터서비스팀 영상제공\강남_원본\11_2385350_20190201_CT")
TARGET_SERIES_DESC = "With Contrast  3.0  B30f"


def normalize_value(value):
    """
    DICOM Value를 비교 가능한 문자열로 정규화
    """
    if value is None:
        return "None"

    # MultiValue, list, tuple 처리
    if isinstance(value, (list, tuple)):
        return tuple(normalize_value(v) for v in value)

    # bytes 처리
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="ignore")
        except Exception:
            return str(value)

    # pydicom 특수 타입 포함해서 문자열로 통일
    return str(value).strip()


def safe_get_series_description(ds):
    try:
        return str(getattr(ds, "SeriesDescription", "")).strip()
    except Exception:
        return ""


def collect_target_dicoms(root_dir: Path, target_series_desc: str):
    """
    root_dir 아래 DICOM 중 SeriesDescription이 target_series_desc인 것만 수집
    """
    matched_files = []

    for fp in root_dir.rglob("*"):
        if not fp.is_file():
            continue

        try:
            ds = pydicom.dcmread(str(fp), stop_before_pixels=True, force=True)
            series_desc = safe_get_series_description(ds)

            if series_desc == target_series_desc:
                matched_files.append(fp)

        except Exception:
            # DICOM이 아니거나 손상된 파일은 무시
            continue

    return matched_files


def find_variable_tags(dicom_files):
    """
    여러 DICOM 파일을 비교해서 값이 변하는 태그만 반환
    """
    tag_to_values = defaultdict(set)
    tag_to_name = {}
    file_count = 0

    for fp in dicom_files:
        try:
            ds = pydicom.dcmread(str(fp), stop_before_pixels=True, force=True)
            file_count += 1

            for elem in ds.iterall():
                # Pixel Data는 비교에서 제외하는 것이 보통 유리
                if elem.tag == (0x7FE0, 0x0010):
                    continue

                tag_key = f"({elem.tag.group:04X},{elem.tag.element:04X})"
                tag_name = elem.name
                value = normalize_value(elem.value)

                tag_to_name[tag_key] = tag_name
                tag_to_values[tag_key].add(value)

        except Exception as e:
            print(f"[WARN] 읽기 실패: {fp} / {e}")

    variable_tags = []

    for tag_key, values in tag_to_values.items():
        if len(values) > 1:
            variable_tags.append({
                "tag": tag_key,
                "name": tag_to_name.get(tag_key, "Unknown"),
                "n_unique_values": len(values),
                "sample_values": list(sorted(values, key=lambda x: str(x)))[:10],
            })

    variable_tags.sort(key=lambda x: x["tag"])
    return variable_tags, file_count


def main():
    dicom_files = collect_target_dicoms(ROOT_DIR, TARGET_SERIES_DESC)

    print(f"[INFO] 대상 SeriesDescription: {TARGET_SERIES_DESC}")
    print(f"[INFO] 매칭된 DICOM 파일 수: {len(dicom_files)}")

    if not dicom_files:
        print("[ERROR] 조건에 맞는 DICOM 파일이 없습니다.")
        return

    variable_tags, file_count = find_variable_tags(dicom_files)

    print(f"[INFO] 실제 비교에 사용된 파일 수: {file_count}")
    print(f"[INFO] 값이 변하는 태그 수: {len(variable_tags)}")
    print("-" * 100)

    for item in variable_tags:
        print(f"{item['tag']} | {item['name']} | unique={item['n_unique_values']}")
        for v in item["sample_values"]:
            print(f"    - {v}")
        print("-" * 100)


if __name__ == "__main__":
    main()