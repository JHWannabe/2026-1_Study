from core_clinic_aec_feature import run_linear_regression, run_logistic_regression, run_resnet1d

TARGET2 = 'TAMA'
TARGET1 = 'SMI'

if __name__ == '__main__':
    # print(f'=== {TARGET1} Linear Regression ===')
    # run_linear_regression(TARGET1)
    # print(f'\n=== {TARGET1} Logistic Regression ===')
    # run_logistic_regression(TARGET1)
    # print(f'=== {TARGET2} Linear Regression ===')
    # run_linear_regression(TARGET2)
    # print(f'\n=== {TARGET2} Logistic Regression ===')
    # run_logistic_regression(TARGET2)
    print(f'\n=== {TARGET1} ResNet1D ===')
    run_resnet1d(TARGET1)
    print(f'\n=== {TARGET2} ResNet1D ===')
    run_resnet1d(TARGET2)
