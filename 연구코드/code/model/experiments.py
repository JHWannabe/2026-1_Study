"""
단계별 성능 비교 실험 실행기 (레거시 — 단일 config 전환 후 비활성).

단일 config 구조로 전환되었으므로 이 파일은 더 이상 사용하지 않는다.
실험을 실행하려면 main.py를 직접 실행하라:
  python 연구코드/code/model/main.py
"""

import os
import sys

MAIN_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")


def main() -> None:
    print("experiments.py는 단일 config 전환 이후 더 이상 사용하지 않습니다.")
    print(f"대신 main.py를 직접 실행하세요: python {MAIN_PY}")
    sys.exit(0)


if __name__ == "__main__":
    main()
