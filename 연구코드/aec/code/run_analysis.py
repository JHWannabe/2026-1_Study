# -*- coding: utf-8 -*-
import sys, io

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if isinstance(sys.stderr, io.TextIOWrapper):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

"""
run_analysis.py - 전체 분석 파이프라인 실행

실행 순서:
  1. feature_selection.py  → AEC feature 상관분석 (결과 확인 후 config.py 수정 권장)
  2. linear_regression.py  → Part 1: 선형 회귀 (단변량 + 다변량)
  3. logistic_regression.py → Part 2: 로지스틱 회귀 (단변량 + 다변량)
  4. multivariable_analysis.py → Part 3: Case 1·2·3 모델 비교

모든 결과는 results/ 폴더에 저장됩니다.

사용법:
  python run_analysis.py              # 전체 실행
  python run_analysis.py --skip-fs    # feature selection 건너뜀
"""

import sys
import os
import time
import traceback

import code.config as config

SKIP_FEATURE_SELECTION = '--skip-fs' in sys.argv


def run_step(name: str, module_main):
    """하나의 분석 단계를 실행하고 성공/실패를 출력."""
    print(f"\n{'━' * 60}")
    print(f"  실행: {name}")
    print(f"{'━' * 60}")
    t0 = time.time()
    try:
        module_main()
        elapsed = time.time() - t0
        print(f"  ✔ 완료 ({elapsed:.1f}초)")
        return True
    except Exception:
        print(f"  ✘ 오류 발생:")
        traceback.print_exc()
        return False


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("  TAMA 예측 회귀분석 연구 - 전체 파이프라인")
    print(f"  데이터: {config.EXCEL_PATH}")
    print(f"  결과 저장: {config.RESULTS_DIR}")
    print(f"  선택된 AEC features: {config.SELECTED_AEC_FEATURES}")
    print(f"  TAMA 임계값: M < {config.TAMA_THRESHOLD_MALE} cm², "
          f"F < {config.TAMA_THRESHOLD_FEMALE} cm²")
    print("=" * 60)

    results = {}

    # Step 1: Feature Selection
    if not SKIP_FEATURE_SELECTION:
        import code.feature_selection as feature_selection
        ok = run_step("Step 1 - Feature Selection (AEC 상관분석)", feature_selection.main)
        results['feature_selection'] = ok
        if ok:
            print("\n  ※ feature_selection_report.xlsx 를 확인한 후")
            print("     config.py 의 SELECTED_AEC_FEATURES 를 업데이트하고")
            print("     python run_analysis.py --skip-fs 로 재실행하세요.")
            print("     (현재 설정으로 계속 진행합니다...)")
    else:
        print("\n  [Step 1 건너뜀: --skip-fs 옵션 사용]")

    # Step 2: Linear Regression
    import code.linear_regression as linear_regression
    ok = run_step("Step 2 - Part 1: 선형 회귀 (Linear Regression)", linear_regression.main)
    results['linear'] = ok

    # Step 3: Logistic Regression
    import code.logistic_regression as logistic_regression
    ok = run_step("Step 3 - Part 2: 로지스틱 회귀 (Logistic Regression)", logistic_regression.main)
    results['logistic'] = ok

    # Step 4: Multivariable Analysis
    import code.multivariable_analysis as multivariable_analysis
    ok = run_step("Step 4 - Part 3: Multivariable Analysis (Case 1·2·3)", multivariable_analysis.main)
    results['multivariable'] = ok

    # Step 5: Generate Plots
    import code.generate_plots as generate_plots
    ok = run_step("Step 5 - 시각화 생성 (15개 그래프)", generate_plots.main)
    results['plots'] = ok

    # Step 6: Generate Report
    import code.generate_report as generate_report
    ok = run_step("Step 6 - Markdown 연구 보고서 생성", generate_report.generate_report)
    results['report'] = ok

    # 최종 요약
    print(f"\n{'=' * 60}")
    print("  실행 결과 요약")
    print(f"{'=' * 60}")
    labels = {
        'feature_selection': 'Step 1  Feature Selection',
        'linear':            'Step 2  Linear Regression',
        'logistic':          'Step 3  Logistic Regression',
        'multivariable':     'Step 4  Multivariable Analysis',
        'plots':             'Step 5  시각화 생성',
        'report':            'Step 6  Markdown 보고서',
    }
    all_ok = True
    for key, label in labels.items():
        if key not in results:
            status = '건너뜀'
        elif results[key]:
            status = '✔ 성공'
        else:
            status = '✘ 실패'
            all_ok = False
        print(f"  {label:35s}: {status}")

    print(f"\n  결과 파일 위치: {config.RESULTS_DIR}")
    print(f"  ├── feature_selection_report.xlsx")
    print(f"  ├── linear_results.xlsx")
    print(f"  ├── logistic_results.xlsx")
    print(f"  ├── multivariable_results.xlsx")
    print(f"  ├── figures/  (15개 PNG)")
    print(f"  └── research_report.md")

    if not all_ok:
        print("\n  일부 단계에서 오류가 발생했습니다. 위 오류 메시지를 확인하세요.")
    else:
        print("\n  모든 분석이 성공적으로 완료되었습니다.")
    print("=" * 60)


if __name__ == '__main__':
    main()
