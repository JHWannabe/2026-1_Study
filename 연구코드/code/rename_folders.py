"""
폴더명 변환 스크립트
대상 경로: D:\영상제공\강남\강남_axial_except

변환 예시:
  "0001_5253702_20180201_CT"  →  "5253702"
  (0번째_1번째_2번째_3번째 중 1번째 추출)

중복 처리:
  동일한 환자 ID를 가진 폴더가 여러 개이면, 하위 파일을 모두
  하나의 대상 폴더로 합칩니다. 파일명이 충돌할 경우 뒤에 오는
  파일은 _2, _3 ... 접미사를 붙여 보존합니다.
"""

import os
import shutil
from collections import defaultdict


BASE_DIR = r"D:\영상제공\강남\강남_axial_except"


def build_plan(base_dir: str) -> dict[str, list[str]]:
    """new_name → [원본 폴더 경로, ...] 매핑 반환."""
    mapping: dict[str, list[str]] = defaultdict(list)

    for name in sorted(os.listdir(base_dir)):
        full_path = os.path.join(base_dir, name)
        if not os.path.isdir(full_path):
            continue

        parts = name.split("_")
        if len(parts) < 2:
            print(f"  [SKIP] 언더스코어 부족, 건너뜀: {name}")
            continue

        new_name = parts[1]   # 두 번째 토큰 (환자 ID)
        mapping[new_name].append(full_path)

    return dict(mapping)


def safe_move(src_file: str, dst_dir: str) -> str:
    """파일을 dst_dir로 이동. 이름 충돌 시 _2, _3 ... 접미사 부여."""
    fname = os.path.basename(src_file)
    dst_file = os.path.join(dst_dir, fname)

    if not os.path.exists(dst_file):
        shutil.move(src_file, dst_file)
        return dst_file

    # 충돌 → 접미사 붙이기
    base, ext = os.path.splitext(fname)
    counter = 2
    while True:
        new_fname = f"{base}_{counter}{ext}"
        dst_file = os.path.join(dst_dir, new_fname)
        if not os.path.exists(dst_file):
            shutil.move(src_file, dst_file)
            return dst_file
        counter += 1


def merge_into(src_dir: str, dst_dir: str) -> int:
    """src_dir 안의 모든 파일을 dst_dir로 이동. 빈 src_dir 삭제. 이동 파일 수 반환."""
    os.makedirs(dst_dir, exist_ok=True)
    moved = 0
    for item in os.listdir(src_dir):
        src_item = os.path.join(src_dir, item)
        if os.path.isfile(src_item):
            result = safe_move(src_item, dst_dir)
            print(f"      이동: {item}  →  {os.path.basename(result)}")
            moved += 1
        elif os.path.isdir(src_item):
            # 하위 폴더도 재귀적으로 합침
            sub_dst = os.path.join(dst_dir, item)
            moved += merge_into(src_item, sub_dst)
    # 빈 폴더 삭제
    if not os.listdir(src_dir):
        os.rmdir(src_dir)
    return moved


def main():
    if not os.path.isdir(BASE_DIR):
        print(f"[ERROR] 경로를 찾을 수 없습니다: {BASE_DIR}")
        return

    plan = build_plan(BASE_DIR)
    if not plan:
        print("변경할 폴더가 없습니다.")
        return

    # ── 미리보기 ──────────────────────────────────────────────
    single = {k: v for k, v in plan.items() if len(v) == 1}
    merged = {k: v for k, v in plan.items() if len(v) > 1}

    print(f"\n[단순 이름 변경]  {len(single)}개")
    for new_name, (src,) in sorted(single.items()):
        print(f"  {os.path.basename(src):<40}  →  {new_name}")

    if merged:
        print(f"\n[합치기]  {len(merged)}개 대상 폴더 (중복 환자 ID)")
        for new_name, srcs in sorted(merged.items()):
            print(f"  → {new_name}/")
            for s in srcs:
                print(f"      {os.path.basename(s)}")

    # ── 사용자 확인 ───────────────────────────────────────────
    total = sum(len(v) for v in plan.values())
    print(f"\n총 {total}개 폴더를 처리합니다.")
    answer = input("계속하시겠습니까? (y/n): ").strip().lower()
    if answer != "y":
        print("작업을 취소했습니다.")
        return

    # ── 실제 처리 ─────────────────────────────────────────────
    success, fail = 0, 0

    # 1) 단순 이름 변경 (1:1)
    for new_name, (src_path,) in sorted(single.items()):
        dst_path = os.path.join(BASE_DIR, new_name)
        if os.path.exists(dst_path):
            print(f"  [SKIP] 이미 존재함: {new_name}")
            fail += 1
            continue
        try:
            os.rename(src_path, dst_path)
            print(f"  [OK]   {os.path.basename(src_path)}  →  {new_name}")
            success += 1
        except Exception as e:
            print(f"  [FAIL] {os.path.basename(src_path)}: {e}")
            fail += 1

    # 2) 합치기 (N:1)
    for new_name, src_list in sorted(merged.items()):
        dst_path = os.path.join(BASE_DIR, new_name)
        print(f"\n  [MERGE] → {new_name}/")
        try:
            for src_path in src_list:
                print(f"    {os.path.basename(src_path)} 에서 합치는 중...")
                merge_into(src_path, dst_path)
            print(f"    [OK] 합치기 완료: {new_name}")
            success += len(src_list)
        except Exception as e:
            print(f"    [FAIL] {new_name}: {e}")
            fail += 1

    print(f"\n완료: 성공 {success}개 / 실패·건너뜀 {fail}개")


if __name__ == "__main__":
    main()
