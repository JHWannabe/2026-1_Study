from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from tqdm import tqdm


# =========================
# 설정
# =========================
DICOM_ROOT_DIRS = {
    "신촌": Path(r"D:\데이터서비스팀 영상제공\신촌_원본"),
    "강남": Path(r"D:\데이터서비스팀 영상제공\강남_원본"),
}

GROUP_ROOT_DIR = Path(r"D:\데이터서비스팀 영상제공\강남_결과\Results\DLO")

SAVE_DIR = Path(r"D:\데이터서비스팀 영상제공\analysis_result")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 1) 폴더별 DICOM(.dcm) 개수 세기
# =========================
def count_dcm_per_folder(root_dir: Path) -> tuple[dict[str, int], list[int]]:
    folder_counts: dict[str, int] = {}
    all_counts: list[int] = []

    if not root_dir.exists():
        print(f"[경고] 경로가 존재하지 않습니다: {root_dir}")
        return folder_counts, all_counts

    folders = [f for f in root_dir.iterdir() if f.is_dir()]

    for folder in tqdm(folders, desc=f"Scanning {root_dir}", unit="folder"):
        dcm_count = 0

        try:
            for file in folder.iterdir():
                if file.is_file() and file.suffix.lower() == ".dcm":
                    dcm_count += 1
        except PermissionError:
            print(f"[Skip] 접근 권한 없음: {folder}")
            continue

        # site명을 붙여서 key 충돌 방지
        folder_counts[folder.name] = dcm_count
        all_counts.append(dcm_count)

    return folder_counts, all_counts


def summarize_dcm_counts(site_name: str, folder_counts: dict[str, int], all_counts: list[int]) -> dict[int, int] | None:
    if len(all_counts) == 0:
        print(f"\n[{site_name}] No DICOM files found")
        return None

    print(f"\n====== {site_name} SUMMARY ======")
    print("Total folders :", len(all_counts))
    print("Min DCM count :", min(all_counts))
    print("Max DCM count :", max(all_counts))
    print("Mean DCM count:", sum(all_counts) / len(all_counts))

    dist: dict[int, int] = defaultdict(int)
    for c in all_counts:
        dist[c] += 1

    print(f"\n=== {site_name} Folder distribution ===")
    for k in sorted(dist):
        print(f"{k} files : {dist[k]} folders")

    return dict(dist)


def plot_dcm_distribution(all_counts: list[int], dist: dict[int, int], prefix: str) -> None:
    if not all_counts:
        return

    min_v = min(all_counts)
    max_v = max(all_counts)

    # Histogram 저장
    plt.figure(figsize=(10, 5))
    plt.hist(all_counts, bins=30, edgecolor="black")
    plt.xlim(min_v, max_v)
    plt.xlabel("Number of DICOM files per folder")
    plt.ylabel("Number of folders")
    plt.title(f"{prefix} DICOM Count Distribution")

    save_path = SAVE_DIR / f"{prefix.lower()}_dcm_histogram.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Bar chart 저장
    x = sorted(dist.keys())
    y = [dist[k] for k in x]

    plt.figure(figsize=(12, 5))
    plt.bar(x, y, edgecolor="black")
    plt.xlim(min_v, max_v)
    plt.xlabel("DICOM files per folder")
    plt.ylabel("Number of folders")
    plt.title(f"{prefix} Folder Count by DICOM File Number")

    save_path = SAVE_DIR / f"{prefix.lower()}_dcm_bar_distribution.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[{prefix}] Graphs saved to: {SAVE_DIR}")


# =========================
# 2) 폴더명 prefix 기준 그룹핑
# 예: AAA.BBB.1, AAA.BBB.2 -> AAA.BBB
# =========================
def count_group_by_prefix(root_dir: Path) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)

    if not root_dir.exists():
        print(f"[경고] 그룹 경로가 존재하지 않습니다: {root_dir}")
        return groups

    for entry in root_dir.iterdir():
        if not entry.is_dir():
            continue

        folder_name = entry.name
        parts = folder_name.split(".")

        if len(parts) > 1:
            group_key = ".".join(parts[:-1])
        else:
            group_key = folder_name

        groups[group_key].append(folder_name)

    return groups


def summarize_groups(groups: dict[str, list[str]]) -> None:
    if not groups:
        print("\nNo groups found.")
        return

    print("\n====== PREFIX GROUP SUMMARY ======")
    for k in sorted(groups):
        print(f"{k} -> {len(groups[k])}개")

    print(f"총 그룹 개수 : {len(groups)}")


# =========================
# 실행
# =========================
def main() -> None:
    # 1) site별 DICOM count
    total_folder_counts: dict[str, int] = {}
    total_all_counts: list[int] = []

    for site_name, root_dir in DICOM_ROOT_DIRS.items():
        folder_counts, all_counts = count_dcm_per_folder(root_dir)

        # site명 붙여서 병합
        for folder_name, cnt in folder_counts.items():
            total_folder_counts[f"{site_name}_{folder_name}"] = cnt
        total_all_counts.extend(all_counts)

        dist = summarize_dcm_counts(site_name, folder_counts, all_counts)
        if dist is not None:
            plot_dcm_distribution(all_counts, dist, prefix=site_name)

    # 전체 통합 요약
    total_dist = summarize_dcm_counts("전체", total_folder_counts, total_all_counts)
    if total_dist is not None:
        plot_dcm_distribution(total_all_counts, total_dist, prefix="total")

    # 2) prefix 그룹핑
    groups = count_group_by_prefix(GROUP_ROOT_DIR)
    summarize_groups(groups)


if __name__ == "__main__":
    main()