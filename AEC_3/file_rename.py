import os, time, shutil
from pathlib import Path

type = "신촌"
root_dir = rf"D:\데이터서비스팀 영상제공\{type}\{type}_AEC_minmax"
dst_dir = rf"C:\Users\user\Desktop\Study\data\AEC\{type}\AEC_2"


def get_unique_path(dst_path: Path) -> Path:
    """
    동일 파일명 존재 시 (2), (3) 형태로 변경
    """
    if not dst_path.exists():
        return dst_path

    base = dst_path.stem
    ext = dst_path.suffix

    i = 2
    while True:
        new_path = dst_path.with_name(f"{base}({i}){ext}")
        if not new_path.exists():
            return new_path
        i += 1


def rename_png_files(root_dir):
    root = Path(root_dir)
    png_files = [p for p in root.glob("*.png")]

    name_counter = {}

    renamed_files = []

    for file_path in png_files:
        original_name = file_path.stem
        ext = file_path.suffix

        base_name = original_name.split("_")[0]

        if base_name not in name_counter:
            name_counter[base_name] = 1
            new_name = base_name
        else:
            name_counter[base_name] += 1
            new_name = f"{base_name}({name_counter[base_name]})"

        new_path = file_path.with_name(new_name + ext)

        try:
            file_path.rename(new_path)
            print(f"[변경] {file_path.name} → {new_path.name}")
            renamed_files.append(new_path)
        except Exception as e:
            print(f"[에러] {file_path.name}: {e}")

    return renamed_files


def copy_files(files, dst_dir):
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    for file_path in files:
        try:
            dst_path = dst / file_path.name

            # ✅ 기존 파일이 있어도 그대로 덮어쓰기
            shutil.copy2(file_path, dst_path)

            print(f"[덮어쓰기 복사] {file_path.name} → {dst_path.name}")

        except Exception as e:
            print(f"[복사 에러] {file_path.name}: {e}")


if __name__ == "__main__":
    while True:
        try:
            print("\n[실행 시작]")

            renamed_files = rename_png_files(root_dir)

            # ✅ rename된 파일만 복사
            if renamed_files:
                copy_files(renamed_files, dst_dir)

            print("[실행 완료] 30초 대기\n")

        except Exception as e:
            print(f"[전체 에러] {e}")

        time.sleep(30)