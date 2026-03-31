import os
import shutil
import time

type = "신촌"
root_dir = rf"D:\데이터서비스팀 영상제공\{type}\{type}_AEC_minmax"


def move_with_rename(src_file, root_dir):
    filename = os.path.basename(src_file)
    dst_file = os.path.join(root_dir, filename)

    # 파일명 충돌 방지
    if os.path.exists(dst_file):
        base, ext = os.path.splitext(filename)
        i = 1
        while True:
            new_name = f"{base}_{i}{ext}"
            new_dst = os.path.join(root_dir, new_name)
            if not os.path.exists(new_dst):
                dst_file = new_dst
                break
            i += 1

    shutil.move(src_file, dst_file)
    print(f"[이동 완료] {src_file} → {dst_file}")


def process_single_png_folders(root_dir):
    for current_path, dirs, files in os.walk(root_dir, topdown=False):
        if current_path == root_dir:
            continue

        png_files = [f for f in files if f.lower().endswith(".png")]

        if len(png_files) == 0:
            continue

        try:
            # png가 1개뿐이면 이동
            if len(png_files) == 1:
                src_file = os.path.join(current_path, png_files[0])

                move_with_rename(src_file, root_dir)

                if not os.listdir(current_path):
                    os.rmdir(current_path)

                continue

            # portal / post / with_contrast 포함 파일 우선 선택
            priority_files = [
                f for f in png_files
                if any(keyword in f.lower() for keyword in ["portal", "post", "with_contrast", "hvp", "delay_1.0"])
            ]

            if len(priority_files) > 0:
                selected = priority_files[0]
                src_file = os.path.join(current_path, selected)

                move_with_rename(src_file, root_dir)

                # 나머지 png 삭제
                for f in png_files:
                    if f != selected:
                        os.remove(os.path.join(current_path, f))
                        print(f"[삭제 완료] {os.path.join(current_path, f)}")

                # 폴더가 비었으면 삭제
                if not os.listdir(current_path):
                    os.rmdir(current_path)

        except Exception as e:
            print(f"[에러] {current_path}: {e}")


if __name__ == "__main__":
    while True:
        try:
            print("\n[실행 시작]")
            process_single_png_folders(root_dir)
            print("[실행 완료] 30초 대기\n")
        except Exception as e:
            print(f"[전체 에러] {e}")

        time.sleep(30)