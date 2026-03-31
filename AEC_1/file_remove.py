import os
import shutil
import time

root_dir = r"D:\데이터서비스팀 영상제공\신촌\신촌_AEC_minmax"

def process_single_png_folders(root_dir):
    for current_path, dirs, files in os.walk(root_dir, topdown=False):
        if current_path == root_dir:
            continue

        png_files = [f for f in files if f.lower().endswith(".png")]

        if len(png_files) == 1:
            src_file = os.path.join(current_path, png_files[0])
            dst_file = os.path.join(root_dir, png_files[0])

            try:
                # 파일명 충돌 방지
                if os.path.exists(dst_file):
                    base, ext = os.path.splitext(png_files[0])
                    i = 1
                    while True:
                        new_name = f"{base}_{i}{ext}"
                        new_dst = os.path.join(root_dir, new_name)
                        if not os.path.exists(new_dst):
                            dst_file = new_dst
                            break
                        i += 1

                shutil.move(src_file, dst_file)

                # 폴더 삭제
                if not os.listdir(current_path):
                    os.rmdir(current_path)

                print(f"[이동 완료] {src_file} → {dst_file}")

            except Exception as e:
                print(f"[에러] {src_file}: {e}")


# =========================================================
# 1분마다 반복 실행
# =========================================================
if __name__ == "__main__":
    while True:
        try:
            print("\n[실행 시작]")
            process_single_png_folders(root_dir)
            print("[실행 완료] 60초 대기\n")

        except Exception as e:
            print(f"[전체 에러] {e}")

        time.sleep(60)  # 60초 대기