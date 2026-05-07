"""
4개 모델 전체 실행 + Primary comparison (항목 5)
  Model 1: Age + Sex + BMI
  Model 2: AEC-CNN only
  Model 3: Age + Sex + BMI + AEC-CNN score  (primary_model.py)
  Model 4: Age + Sex + BMI + handcrafted AEC

Primary comparison (§10 item 5): Model 1 vs Model 3
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from model1_clinic_only      import run_model1
from model2_aec_cnn_only     import run_model2
from primary_model           import run_primary_model   # Model 3
from model4_clinic_handcrafted import run_model4

TARGET = 'SMI'

if __name__ == '__main__':
    results = {}
    results['M1: Clinic only']       = run_model1(TARGET)
    results['M2: AEC-CNN only']      = run_model2(TARGET)
    results['M3: Clinic + AEC-CNN']  = run_primary_model(TARGET)
    results['M4: Clinic + HC-AEC']   = run_model4(TARGET)

    # ── 4개 모델 비교표 ────────────────────────────────────
    print(f'\n{"="*68}')
    print(f'[4-Model Comparison - {TARGET}]')
    print(f'{"="*68}')
    hdr = f"{'Model':<26} {'AUC':>8} {'AUPRC':>8} {'Brier':>8} {'Acc':>8}"
    print(hdr)
    print('-' * len(hdr))
    for name, m in results.items():
        print(f"{name:<26} {m['AUC']:>8.4f} {m['AUPRC']:>8.4f} "
              f"{m['Brier']:>8.4f} {m['Acc']:>8.4f}")
    print('=' * len(hdr))

    # ── Primary comparison: Model 1 vs Model 3 (§10 item 5) ──
    m1 = results['M1: Clinic only']
    m3 = results['M3: Clinic + AEC-CNN']
    delta_auc   = m3['AUC']   - m1['AUC']
    delta_auprc = m3['AUPRC'] - m1['AUPRC']
    delta_brier = m3['Brier'] - m1['Brier']   # negative = better
    print(f'\n[Primary Comparison - §10 item 5]')
    print(f'  Model 1 (Clinic only)    AUC={m1["AUC"]:.4f}  AUPRC={m1["AUPRC"]:.4f}  Brier={m1["Brier"]:.4f}')
    print(f'  Model 3 (Clinic+AEC-CNN) AUC={m3["AUC"]:.4f}  AUPRC={m3["AUPRC"]:.4f}  Brier={m3["Brier"]:.4f}')
    print(f'  Delta (M3 - M1)          dAUC={delta_auc:+.4f}  dAUPRC={delta_auprc:+.4f}  dBrier={delta_brier:+.4f}')

    if delta_auc > 0.01:
        print('  => AEC-CNN has incremental predictive value over Age/Sex/BMI.')
    else:
        print('  => AEC-CNN incremental value is marginal; may primarily proxy BMI/body size.')
