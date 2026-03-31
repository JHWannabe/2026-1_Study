from __future__ import annotations
from pathlib import Path
import pandas as pd


# =========================
# 빈 폴더 삭제
# =========================
def delete_empty_folders(root: Path) -> int:
    deleted_count = 0

    # 하위 폴더부터 처리
    for directory in sorted(
        (p for p in root.rglob("*") if p.is_dir()),
        key=lambda p: len(p.parts),
        reverse=True,
    ):
        try:
            if not any(directory.iterdir()):
                directory.rmdir()
                deleted_count += 1
                print(f"Deleted empty folder: {directory}")
        except Exception as e:
            print(f"삭제 실패: {directory} | {e}")

    return deleted_count


# =========================
# 폴더별 파일 개수 집계
# =========================
def analyze_folder_counts(base_dir: Path) -> pd.DataFrame:
    rows = []

    for folder in base_dir.iterdir():
        if folder.is_dir():
            try:
                file_count = sum(1 for x in folder.iterdir() if x.is_file())
            except Exception as e:
                print(f"카운트 실패: {folder} | {e}")
                file_count = 0

            rows.append({
                "folder_name": folder.name,
                "file_count": file_count,
            })

    if not rows:
        return pd.DataFrame(columns=["folder_name", "file_count"])

    df = pd.DataFrame(rows)
    df = df.sort_values(by="file_count", ascending=False).reset_index(drop=True)
    return df


# =========================
# 전체 처리
# =========================
def process_one_site(site_name: str, base_dir: Path) -> None:
    if not base_dir.exists():
        print(f"[{site_name}] 경로 없음: {base_dir}")
        return

    print(f"\n[{site_name}] 1) 빈 폴더 삭제 시작")
    deleted = delete_empty_folders(base_dir)
    print(f"[{site_name}] 삭제된 빈 폴더 수: {deleted}")

    print(f"[{site_name}] 2) 폴더별 파일 개수 분석 시작")
    df = analyze_folder_counts(base_dir)

    out_csv = base_dir / "folder_file_counts.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"[{site_name}] 완료: {out_csv}")
    print(df.head())


if __name__ == "__main__":
    base_dirs = {
        "신촌1": Path(r"D:\데이터서비스팀 영상제공\신촌_plots_raws"),
        "신촌2": Path(r"D:\데이터서비스팀 영상제공\신촌_plots"),
        "강남1": Path(r"D:\데이터서비스팀 영상제공\강남_plots_raws"),
        "강남2": Path(r"D:\데이터서비스팀 영상제공\강남_plots"),
    }

    for site_name, base_dir in base_dirs.items():
        process_one_site(site_name, base_dir)