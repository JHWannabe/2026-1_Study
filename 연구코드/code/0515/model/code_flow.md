# Code Flow — `python main.py` 실행 워크플로우

## 진입점

```
if __name__ == "__main__":
    run_all_cases()          ← 유일한 진입점
```

---

## 1단계 — 데이터 로드 및 분할 (`run_all_cases`)

```
run_all_cases()
│
├─ load_data()                    → X (N,3), y, sex            # M1용 (Age, sex_enc, BMI)
├─ load_data_with_aec()           → X_clin, X_aec, y2, sex2   # M2용 (Clinic + AEC Matched)
├─ load_data_with_aec_unmatched() → X_clin_u, X_aec_u, ...    # M2_2용 (AEC 행 순서 셔플)
└─ load_data_with_aec_meta()      → X_clin3, X_aec3, X_mfr,   # M3용 (+ ManufacturerModelName)
                                     y3, sex3, n_mfr
```

**공통 전처리** (`_load_filtered_meta` 내부, 4개 로드 함수 모두 호출):
1. kVp == 100 필터 (다른 kVp는 AEC 신호 특성 상이)
2. 소수 제조사 제거 (비율 < `MIN_MFR_RATIO` = 5%)
3. SMI 이진 레이블 생성 (M: ≤40.96 → sarcopenia=1, F: ≤30.6 → sarcopenia=1)

**분할** (`stratify=y`, `TEST_SIZE=0.2`, `SEED=42`):
```
split_data()      → X_cv / X_te              (M1)
split_data_dual() → X_clin_cv/te, X_aec_cv/te (M2, M2_2)
split_data_quad() → X_clin_cv/te, X_aec_cv/te, X_mfr_cv/te (M3)
```

---

## 2단계 — 4개 모델 병렬 실행 (`ProcessPoolExecutor`, max_workers=4)

```
ProcessPoolExecutor
├─ fut1  = _run_model1(...)     # 별도 프로세스
├─ fut2  = _run_model2(...)     # 별도 프로세스
├─ fut2_2= _run_model2_2(...)   # 별도 프로세스
└─ fut3  = _run_model3(...)     # 별도 프로세스
```

각 워커는 stdout을 `io.StringIO`로 캡처 → 완료 후 `run.log`에 저장

---

## 3단계 — 각 모델 워커 내부 루프

### Model 1 (`_run_model1`)

```
for case_name, sc in CASES_M1:          # 1가지 케이스
    run_cross_validation(X_cv, y_cv, scale_X=sc)
    evaluate_test(...)
    save_all(...)
    results.append({...})
```

### Model 2 (`_run_model2`) / Model 2_2 (`_run_model2_2`)

```
for aec_var in AEC_VARIANTS:            # 7가지 AEC 변환
    aec_variant(X_aec_cv, aec_var)      → X_aec_cv_v, mask_cv
    aec_variant(X_aec_te, aec_var)      → X_aec_te_v, mask_te
    # excl_extreme의 경우 mask로 X_clin·y·sex 배열도 필터링
    for case_name, sc, sa in CASES_M2:  # 2가지 스케일링 케이스
        run_cross_validation_cross(...)
        evaluate_test_cross(...)
        save_all_cross(...)
        results.append({aec_var, case, ...})
```

> Model 2_2는 AEC 행 순서가 셔플된 `X_aec_u`를 입력으로 사용 (음성 대조군)

### Model 3 (`_run_model3`)

```
for aec_var in AEC_VARIANTS:            # 7가지 AEC 변환
    aec_variant(X_aec3_cv, aec_var)     → X_aec3_cv_v, mask_cv
    # mask로 X_clin3·X_mfr·y·sex 배열도 필터링
    for case_name, sc, sa in CASES_M3:  # 2가지 스케일링 케이스
        run_cross_validation_cross3(...)
        evaluate_test_cross3(...)
        save_all_cross(...)
        results.append({aec_var, case, ...})
```

---

## 4단계 — Cross-Validation 함수 내부

### `run_cross_validation` (M1)

```
StratifiedKFold(n_splits=5)
└─ for fold in 5 folds:
    ├─ _maybe_scale_clin(X_tr, X_val, scale_X)   # Age·BMI만 표준화 (sex_enc 제외)
    │
    ├─ [LR] LogisticRegression.fit / predict
    │   └─ group_metrics → lr_cv[fold]
    │
    └─ [ResNet1D]
        ├─ build_resnet(y_tr)      # BCEWithLogitsLoss(pos_weight) + Adam + CosineAnnealingLR
        ├─ for ep in 1..EPOCHS(200):
        │   ├─ train_one_epoch(model, tr_dl, crit, opt)
        │   └─ eval_loader(model, val_dl, crit)  → val_auc
        │       └─ if val_auc > best: 가중치 스냅샷 저장
        ├─ model.load_state_dict(best_state)      # best epoch 복원
        └─ group_metrics → rn_cv[fold]

return (lr_cv, rn_cv, lr_roc_folds, rn_roc_folds, rn_histories, best_epochs)
```

### `run_cross_validation_cross` (M2 / M2_2)

```
StratifiedKFold(n_splits=5)
└─ for fold in 5 folds:
    ├─ _maybe_scale_clin(X_clin, scale_clin)   # Age·BMI만 표준화
    ├─ _maybe_scale(X_aec,   scale_aec)        # AEC 전 컬럼 표준화
    │
    ├─ X_lr = hstack([X_clin_s, X_aec_s])     # LR / ResNet1D 공용 입력
    │
    ├─ [LR] LogisticRegression → lr_cv[fold]
    │
    ├─ [CrossAttn] ClinAECCrossAttn
    │   ├─ build_cross_attn(y_tr)
    │   ├─ for ep in 1..200: train_cross_epoch → eval_cross_loader → best epoch 저장
    │   └─ ca_cv[fold]
    │
    └─ [ResNet1D] ResNet1D (X_lr 입력)
        ├─ build_resnet(y_tr)
        ├─ for ep in 1..200: train_one_epoch → eval_loader → best epoch 저장
        └─ rn_cv[fold]

return (lr_cv, ca_cv, rn_cv, ..., ca_best_epochs, rn_best_epochs)
```

### `run_cross_validation_cross3` (M3)

```
StratifiedKFold(n_splits=5)
└─ for fold in 5 folds:
    ├─ _maybe_scale_clin(X_clin, scale_clin)
    ├─ _maybe_scale(X_aec,   scale_aec)
    │
    ├─ X_lr = hstack([X_clin_s, X_mfr.reshape(-1,1), X_aec_s])
    │
    ├─ [LR] LogisticRegression → lr_cv[fold]
    │
    ├─ [CrossAttn3] ClinAECScanCrossAttn
    │   ├─ build_cross_attn3(y_tr, n_manufacturers)
    │   │   └─ MfrTokenizer: ManufacturerModelName 정수 → Embedding 토큰
    │   ├─ for ep in 1..200: train_cross3_epoch → eval_cross3_loader → best epoch 저장
    │   └─ ca3_cv[fold]
    │
    └─ [ResNet1D] ResNet1D (X_lr 입력) → rn_cv[fold]

return (lr_cv, ca3_cv, rn_cv, ..., ca3_best_epochs, rn_best_epochs)
```

---

## 5단계 — Test Set 최종 평가 (`evaluate_test*`)

CV fold best epoch의 **중앙값(median)**을 사용해 전체 CV 세트로 재학습 → test set 예측

```
evaluate_test(X_cv, y_cv, X_te, y_te, sex_te, med_epoch, scale_X)
├─ _scale_clin_te(X_cv, X_te, scale_X)     # Age·BMI만 표준화
├─ LR: fit(X_cv_s) → predict(X_te_s)
└─ ResNet1D: for _ in range(med_epoch): train → eval(X_te)

evaluate_test_cross(... , med_epoch, rn_med_epoch, scale_clin, scale_aec)
├─ _scale_clin_te / _scale_or_copy
├─ LR: fit(hstack[clin, aec]) → predict
├─ CrossAttn: for _ in range(med_epoch): train_cross_epoch → eval_cross_loader
└─ ResNet1D: for _ in range(rn_med_epoch): train_one_epoch → eval_loader

evaluate_test_cross3(... , med_epoch, rn_med_epoch, n_manufacturers, ...)
└─ (CrossAttn3 포함, 구조 동일)
```

반환값:
- M1: `(lr_pred, lr_prob, rn_pred_te, rn_prob_te, rn_true_te)`
- M2/M2_2/M3: `(lr_pred, lr_prob, ca_pred_te, ca_prob_te, ca_true_te, rn_pred_te, rn_prob_te)`

---

## 6단계 — 시각화 및 보고서 저장 (`save_all` / `save_all_cross`)

```
save_all(out_dir=results/0515/model_1/{case})
├─ plot_data_distribution(...)      → data_distribution.png
├─ plot_roc_curves(...)             → cv_roc_curves.png
├─ plot_metric_distribution(...)    → cv_metric_distribution.png
├─ plot_confusion_matrices(...)     → confusion_matrices.png
├─ plot_training_curves(...)        → training_curves.png
├─ plot_test_roc(...)               → test_roc_curves.png
├─ plot_test_roc_by_sex(...)        → test_roc_by_sex.png
├─ plot_calibration(...)            → calibration.png
└─ save_report_md(...)              → results.md

save_all_cross(out_dir=results/0515/model_{2|2_2|3}/{aec_var}/{case})
└─ 동일한 8종 PNG + results.md (CrossAttn 레이블)
```

---

## 7단계 — 모델 간 비교 결과 저장 (`run_all_cases` 마지막)

```
_print_comparison(results_m1, results_m2, results_m2_2, results_m3)
└─ 콘솔 출력: 모델별 LR/CrossAttn/ResNet1D 성능 테이블 + best case 요약

_save_comparison_md(results_m1, results_m2, results_m2_2, results_m3)
├─ Best Cases 요약 테이블 (_best_cases_summary_md)
├─ 모델별 전체 케이스 성능 테이블 (_md_table)
├─ Fold-level 통계 검정
│   ├─ M1: LR vs ResNet1D        (paired t-test + Wilcoxon, _fold_stats)
│   ├─ M2: LR vs CrossAttn
│   ├─ M2: LR vs ResNet1D
│   ├─ M2_2: LR vs CrossAttn (Unmatched)
│   ├─ M3: LR vs CrossAttn3
│   └─ M3: LR vs ResNet1D
└─ Cross-model 비교 (_cross_model_md_block)
    ├─ LR: M1 vs M2(len256), M1 vs M3(len256)
    ├─ LR: M2 vs M2_2, M2 vs M3       → (aec_var, case) 키로 매칭
    ├─ Deep CrossAttn: M1→M2, M1→M3, M2→M2_2, M2→M3
    └─ Deep ResNet1D:  M1→M2, M1→M3, M2→M2_2, M2→M3

→ 저장: results/0515/scaling_comparison.md
```

---

## 전체 실행 흐름 요약

```
main.py
└─ run_all_cases()
    ├─ [데이터] load_data × 4 → split × 4
    ├─ [병렬]  ProcessPoolExecutor(4)
    │   ├─ _run_model1
    │   │   └─ for case(×1):
    │   │       run_cross_validation → evaluate_test → save_all
    │   │
    │   ├─ _run_model2
    │   │   └─ for aec_var(×7) × case(×2):
    │   │       aec_variant → run_cross_validation_cross → evaluate_test_cross → save_all_cross
    │   │
    │   ├─ _run_model2_2  (M2와 동일 구조, AEC unmatched 입력)
    │   │   └─ for aec_var(×7) × case(×2): ...
    │   │
    │   └─ _run_model3
    │       └─ for aec_var(×7) × case(×2):
    │           aec_variant → run_cross_validation_cross3 → evaluate_test_cross3 → save_all_cross
    │
    └─ [결과] _print_comparison + _save_comparison_md
```

### 총 실험 수

| 모델 | AEC variants | 스케일링 케이스 | 서브모델 | 총 실험 |
|------|:-----------:|:-----------:|:------:|:------:|
| M1   | 1           | 1           | LR + ResNet1D | 1 |
| M2   | 7           | 2           | LR + CrossAttn + ResNet1D | 14 |
| M2_2 | 7           | 2           | LR + CrossAttn + ResNet1D | 14 |
| M3   | 7           | 2           | LR + CrossAttn3 + ResNet1D | 14 |

> 각 실험마다 5-Fold CV + Test 평가 수행

---

## 출력 디렉토리 구조

```
results/0515/
├─ model_1/
│   └─ scale_clinic/            # 케이스별
│       ├─ *.png (8종)
│       ├─ results.md
│       └─ ../run.log
├─ model_2/
│   └─ {aec_var}/               # len064, len128, ..., excl_extreme
│       └─ {case}/              # scale_clinic, scale_both
│           ├─ *.png + results.md
│           └─ ../../run.log
├─ model_2_2/   (동일 구조)
├─ model_3/     (동일 구조)
└─ scaling_comparison.md        # 전 모델 비교 테이블
```

---

## 모델 아키텍처 참고

| 모델 | 입력 | 아키텍처 | 파일 |
|------|------|---------|------|
| LR (M1) | Age, sex_enc, BMI | LogisticRegression | `cross_val.py` |
| ResNet1D (M1) | Age, sex_enc, BMI | Conv1D ResNet → FC | `models.py` |
| LR (M2) | Clinic + AEC (hstack) | LogisticRegression | `cross_val.py` |
| CrossAttn (M2) | Clinic / AEC 분리 | Bidirectional Cross-Attention | `models.py` |
| ResNet1D (M2) | Clinic + AEC (hstack) | Conv1D ResNet → FC | `models.py` |
| LR (M3) | Clinic + MFR + AEC | LogisticRegression | `cross_val.py` |
| CrossAttn3 (M3) | Clinic / MFR Emb / AEC | Bidirectional Cross-Attention | `models.py` |
| ResNet1D (M3) | Clinic + MFR + AEC | Conv1D ResNet → FC | `models.py` |
