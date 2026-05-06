from clinic_core import run_linear_regression, run_logistic_regression

TARGET = 'TAMA'

if __name__ == '__main__':
    print(f'=== {TARGET} Linear Regression ===')
    run_linear_regression(TARGET)
    print(f'\n=== {TARGET} Logistic Regression ===')
    run_logistic_regression(TARGET)
