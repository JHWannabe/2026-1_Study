# -*- coding: utf-8 -*-
"""
0430 연구 통합 실행 스크립트.

실행 순서:
  01_feature_selection.py   — AEC 피처 자동 선택 (AEC_NEW 결정)
  02_run_analysis.py        — 회귀 분석 전체 (병원별 / 성별 / 교차비교 / 외부검증 / BMI)
  03_generate_report.py     — 마크다운 리포트 생성
  04_generate_plots.py      — 고수준 요약·비교 그래프 (figures_0430/)
  05_generate_ppt.py        — 0430 연구 PPT 생성
  06_generate_comparison_ppt.py — 0424 vs 0430 비교 PPT (선택)

사용법:
  python 00_run_all.py              # 전체 실행
  python 00_run_all.py --skip 01   # 01 피처선택 건너뛰기 (이미 결과 있을 때)
  python 00_run_all.py --only 05   # 05만 단독 실행
  python 00_run_all.py --stop 03   # 03까지만 실행
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

STEPS = [
    ("01", "01_feature_selection.py",        "AEC 피처 자동 선택"),
    ("02", "02_run_analysis.py",             "회귀 분석 (병원별 / 성별 / 교차 / BMI)"),
    ("03", "03_generate_report.py",          "마크다운 리포트 생성"),
    ("04", "04_generate_plots.py",           "요약·비교 그래프 생성"),
    ("05", "05_generate_ppt.py",             "0430 연구 PPT 생성"),
    ("06", "06_generate_comparison_ppt.py",  "0424 vs 0430 비교 PPT 생성"),
]


def run_step(num: str, filename: str, description: str) -> bool:
    """단일 스텝 실행. 성공이면 True, 실패이면 False 반환."""
    script = SCRIPT_DIR / filename
    if not script.exists():
        print(f"\n  [SKIP] {filename} 파일 없음")
        return True

    print(f"\n{'='*60}")
    print(f"  STEP {num}: {description}")
    print(f"  파일: {filename}")
    print(f"{'='*60}")

    t0     = time.time()
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(SCRIPT_DIR),
    )
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"\n  [OK] STEP {num} 완료 ({elapsed:.1f}s)")
        return True
    else:
        print(f"\n  [FAIL] STEP {num} 실패 (returncode={result.returncode}, {elapsed:.1f}s)")
        return False


def parse_args():
    parser = argparse.ArgumentParser(description="0430 연구 통합 실행")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--only", metavar="NUM",
                       help="해당 번호 스텝만 실행 (예: --only 05)")
    group.add_argument("--skip", metavar="NUMS", nargs="+",
                       help="건너뛸 스텝 번호들 (예: --skip 01 06)")
    group.add_argument("--stop", metavar="NUM",
                       help="해당 번호 스텝까지만 실행 후 종료 (예: --stop 03)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "="*60)
    print("  0430 연구 통합 실행")
    print("="*60)
    print("실행 순서:")
    for num, fname, desc in STEPS:
        print(f"  STEP {num}: {desc}")
    print()

    total_start = time.time()
    results     = {}

    for num, fname, desc in STEPS:
        # --only: 해당 스텝만
        if args.only and num != args.only:
            continue
        # --skip: 지정 스텝 건너뜀
        if args.skip and num in args.skip:
            print(f"\n  [SKIP] STEP {num}: {desc}")
            continue

        ok = run_step(num, fname, desc)
        results[num] = ok

        # 실패 시 이후 스텝 중단 (--only 모드 제외)
        if not ok and not args.only:
            print(f"\n  STEP {num} 실패 → 이후 스텝 중단")
            break

        # --stop: 지정 스텝까지만
        if args.stop and num == args.stop:
            print(f"\n  --stop {args.stop} 도달 → 종료")
            break

    # ── 요약 ──
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  실행 결과 요약  (총 {total_elapsed:.1f}s)")
    print(f"{'='*60}")
    for num, fname, desc in STEPS:
        if num in results:
            status = "[OK] 성공" if results[num] else "[FAIL] 실패"
            print(f"  STEP {num}  {status}  - {desc}")
    print()


if __name__ == "__main__":
    main()
