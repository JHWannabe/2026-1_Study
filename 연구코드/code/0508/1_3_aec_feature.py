from core_aec_feature import (
    run_linear_regression,
    run_logistic_regression,
    run_linear_regression_selected,
    run_logistic_regression_selected,
    run_resnet1d,
)

TARGET2 = 'TAMA'
TARGET1 = 'SMI'

if __name__ == '__main__':
    # ── 전체 피처 사용 (aec_feature_only) ──────────────────
    # print(f'\n=== {TARGET1} Linear Regression ===')
    # run_linear_regression(TARGET1)
    # print(f'\n=== {TARGET1} Logistic Regression ===')
    # run_logistic_regression(TARGET1)
    # print(f'\n=== {TARGET2} Linear Regression ===')
    # run_linear_regression(TARGET2)
    # print(f'\n=== {TARGET2} Logistic Regression ===')
    # run_logistic_regression(TARGET2)

    # ── fold별 피처 선택 (aec_feature_selected) ────────────
    # print(f'\n=== {TARGET1} Linear Regression (selected) ===')
    # run_linear_regression_selected(TARGET1)
    # print(f'\n=== {TARGET1} Logistic Regression (selected) ===')
    # run_logistic_regression_selected(TARGET1)
    # print(f'\n=== {TARGET2} Linear Regression (selected) ===')
    # run_linear_regression_selected(TARGET2)
    # print(f'\n=== {TARGET2} Logistic Regression (selected) ===')
    # run_logistic_regression_selected(TARGET2)


    print(f'\n=== {TARGET1} ResNet1D ===')
    run_resnet1d(TARGET1)
    print(f'\n=== {TARGET2} ResNet1D ===')
    run_resnet1d(TARGET2)

