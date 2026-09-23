"""
새 데이터 10000명 코호트 파이프라인을 처음부터 끝까지 순서대로 실행한다. 이 파일 하나만
실행하면 된다.

  [1/4] 7_select_best_series.py              DICOM 직접 스캔으로 PatientID당 best series
                                               선정(liver~pubis 커버리지 필터 포함, 재시작
                                               가능) → best_series.xlsx
  [2/4] 6_new10000_features.py                예후데이터.xlsx 기반 metadata 저장
                                               → new10000_final_dataset.xlsx
  [3/4] aec/extract_liver_pubis_aec_new10000.py  TotalSegmentator 체성분 추출 + AEC 곡선을
                                               같은 크롭 범위로 함께 계산
                                               (환자당 약 66초 × 10578명 ≈ 8일. 체크포인트로
                                               재시작 가능 — 중간에 꺼도 다시 이 파일을 실행하면
                                               이어서 진행됨)
  [4/4] 9_merge_body_composition.py           체성분+AEC 결과를 new10000_final_dataset.xlsx에
                                               병합 → gangnam_final_dataset.xlsx와 동일한
                                               16개 시트 완성

각 단계는 멱등적이라(이미 만든 산출물은 그대로 재사용) 중단 후 재실행해도 안전하다.
"""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    steps = [
        ("1/4 best series 선정", "select_best_series", ROOT / "7_select_best_series.py"),
        ("2/4 metadata 저장", "new10000_features", ROOT / "6_new10000_features.py"),
        ("3/4 TotalSegmentator 체성분 추출 + AEC 곡선 동시 계산 (수일 소요, 재시작 가능)", "bodycomp",
         ROOT / "aec" / "extract_liver_pubis_aec_new10000.py"),
        ("4/4 체성분+AEC 결과 병합", "merge_bodycomp", ROOT / "9_merge_body_composition.py"),
    ]
    for label, name, path in steps:
        print(f"\n{'='*70}\n[{label}]\n{'='*70}")
        _load(name, path).main()

    print("\n전체 파이프라인 완료")


if __name__ == "__main__":
    main()
