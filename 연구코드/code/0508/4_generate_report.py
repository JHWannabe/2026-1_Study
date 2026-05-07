# -*- coding: utf-8 -*-
"""
4_generate_report.py
────────────────────────────────────────────────────────────────
clinic_only / aec_only / aec_feature_only 분석 결과 MD 파일들을
읽어 하나의 요약 보고서를 생성한다.

출력: results/0508_research_report.md

실행: python 4_generate_report.py
────────────────────────────────────────────────────────────────
"""

import sys, io, re
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pathlib import Path

SCRIPT_DIR  = Path(__file__).parent
RESULT_ROOT = SCRIPT_DIR.parent.parent / "results"
BASE_DIR    = RESULT_ROOT / "0508"
OUT_REPORT  = RESULT_ROOT / "0508_research_report.md"

METHODS = [
    ("clinic_only",         "임상 변수 (Age, Sex, BMI)"),
    ("aec_only",            "AEC 원시 피처 256개"),
    ("aec_feature_only",    "AEC 추출 피처 전체"),
    ("aec_feature_selected","AEC 추출 피처 선택 (corr+VIF)"),
    ("clinic_aec_feature",  "임상 변수 + AEC 추출 피처 선택"),
]
METHODS_RESNET1D = [
    ("clinic_only",        "임상 변수"),
    ("aec_only",           "AEC 원시"),
    ("aec_feature_only",   "AEC 피처(fold 선택)"),
    ("clinic_aec_feature", "임상+AEC(fold 선택)"),
]
TARGETS = ["TAMA", "SMI"]


# ── MD 파싱 ───────────────────────────────────────────────────
def parse_resnet1d_md(path: Path) -> dict:
    if not path.exists():
        return {}
    text = path.read_text(encoding='utf-8')
    result = {}

    m = re.search(
        r'\|\s*\*\*Mean\*\*\s*\|\s*\*\*([-0-9.]+)\*\*\s*\|\s*\*\*([-0-9.]+)\*\*\s*\|',
        text)
    if m:
        result['mae_mean'], result['r2_mean'] = float(m.group(1)), float(m.group(2))

    m = re.search(
        r'\|\s*\*\*Std\*\*\s*\|\s*\*\*([-0-9.]+)\*\*\s*\|\s*\*\*([-0-9.]+)\*\*\s*\|',
        text)
    if m:
        result['mae_std'], result['r2_std'] = float(m.group(1)), float(m.group(2))

    m = re.search(r'Test R² = \*\*([-0-9.]+)\*\*', text)
    if m:
        result['test_r2'] = float(m.group(1))

    m = re.search(r'Test MAE = \*\*([-0-9.]+)\*\*', text)
    if m:
        result['test_mae'] = float(m.group(1))

    return result


def parse_linear_md(path: Path) -> dict:
    if not path.exists():
        return {}
    text = path.read_text(encoding='utf-8')
    result = {}

    m = re.search(
        r'\|\s*\*\*Mean\*\*\s*\|\s*\*\*([-0-9.]+)\*\*\s*\|\s*\*\*([-0-9.]+)\*\*\s*\|\s*\*\*([-0-9.]+)\*\*\s*\|',
        text)
    if m:
        result['mse_mean'], result['rmse_mean'], result['r2_mean'] = \
            float(m.group(1)), float(m.group(2)), float(m.group(3))

    m = re.search(
        r'\|\s*\*\*Std\*\*\s*\|\s*\*\*([-0-9.]+)\*\*\s*\|\s*\*\*([-0-9.]+)\*\*\s*\|\s*\*\*([-0-9.]+)\*\*\s*\|',
        text)
    if m:
        result['mse_std'], result['rmse_std'], result['r2_std'] = \
            float(m.group(1)), float(m.group(2)), float(m.group(3))

    m = re.search(
        r'Test Set 성능.*?\n\|[^\n]+\n\|---[^\n]+\n\|\s*\*\*([-0-9.]+)\*\*\s*\|\s*\*\*([-0-9.]+)\*\*\s*\|\s*\*\*([-0-9.]+)\*\*',
        text, re.DOTALL)
    if m:
        result['test_mse'], result['test_rmse'], result['test_r2'] = \
            float(m.group(1)), float(m.group(2)), float(m.group(3))

    m = re.search(r'\*\*Input features:\*\* (.+)', text)
    if m:
        result['features'] = m.group(1).strip()

    return result


def parse_logistic_md(path: Path) -> dict:
    if not path.exists():
        return {}
    text = path.read_text(encoding='utf-8')
    result = {}

    m = re.search(
        r'\|\s*\*\*Mean\*\*\s*\|\s*\*\*([0-9.]+)\*\*\s*\|\s*\*\*([0-9.]+)\*\*\s*\|',
        text)
    if m:
        result['acc_mean'], result['auc_mean'] = float(m.group(1)), float(m.group(2))

    m = re.search(
        r'\|\s*\*\*Std\*\*\s*\|\s*\*\*([0-9.]+)\*\*\s*\|\s*\*\*([0-9.]+)\*\*\s*\|',
        text)
    if m:
        result['acc_std'], result['auc_std'] = float(m.group(1)), float(m.group(2))

    m = re.search(
        r'Test Set 성능.*?\n\|[^\n]+\n\|---[^\n]+\n\|\s*\*\*([0-9.]+)\*\*\s*\|\s*\*\*([0-9.]+)\*\*',
        text, re.DOTALL)
    if m:
        result['test_acc'], result['test_auc'] = float(m.group(1)), float(m.group(2))

    # 성별 임계값
    thresholds = {}
    for sex, label in [('Male', 'Male'), ('Female', 'Female')]:
        t = re.search(rf'\|\s*{sex}\s*\|\s*([0-9.]+)\s*\|', text)
        if t:
            thresholds[label] = float(t.group(1))
    if thresholds:
        result['thresholds'] = thresholds

    return result


# ── 통합 데이터 수집 ──────────────────────────────────────────
def collect_all() -> dict:
    data = {}
    for method_key, method_label in METHODS:
        data[method_key] = {}
        for target in TARGETS:
            lin_path = BASE_DIR / method_key / target / "Linear_Report.md"
            log_path = BASE_DIR / method_key / target / "Logistic_Report.md"
            rn_path  = BASE_DIR / method_key / target / "ResNet1D_Report.md"
            data[method_key][target] = {
                'linear':    parse_linear_md(lin_path),
                'logistic':  parse_logistic_md(log_path),
                'resnet1d':  parse_resnet1d_md(rn_path),
                'lin_exists': lin_path.exists(),
                'log_exists': log_path.exists(),
                'rn_exists':  rn_path.exists(),
            }
    return data


# ── 마크다운 보고서 생성 ──────────────────────────────────────
def fmt(v, d=4):
    try:
        return f'{float(v):.{d}f}'
    except (TypeError, ValueError):
        return 'N/A'


def _interp_r2(r2: float) -> str:
    if r2 >= 0.7:
        return f'{r2:.4f} ✅ **우수** — 모델이 실제값 변동의 {r2*100:.0f}%를 설명함'
    if r2 >= 0.5:
        return f'{r2:.4f} 🟡 **보통** — 어느 정도 예측 가능하나 설명력 제한적'
    if r2 >= 0.3:
        return f'{r2:.4f} 🟠 **낮음** — 일부 경향 포착, 실용 수준 미달'
    return f'{r2:.4f} 🔴 **매우 낮음** — 예측 실패에 가까움'


def _interp_auc(auc: float) -> str:
    if auc >= 0.8:
        return f'{auc:.4f} ✅ **우수** (0.8 이상: 양호한 분류 능력)'
    if auc >= 0.7:
        return f'{auc:.4f} 🟡 **양호** (0.7~0.8: 임상 참고 수준)'
    if auc >= 0.6:
        return f'{auc:.4f} 🟠 **미흡** (0.6~0.7: 제한적 변별력)'
    return f'{auc:.4f} 🔴 **무의미** (0.6 미만: 무작위 분류와 유사)'


def make_report(data: dict) -> str:
    lines = [
        '# 0508 분석 연구 보고서',
        '',
        '> **데이터**: 강남세브란스병원 (`aec_feature_filtered` 시트)',
        '> **분석**: 5-Fold Cross Validation · Linear Regression · Logistic Regression',
        '> **자동 생성**: `4_generate_report.py`',
        '',
        '---',
        '',
        '## 1. 연구 배경 및 목적',
        '',
        '### 1.1 왜 근육량(TAMA, SMI)을 예측하는가?',
        '',
        '근감소증(Sarcopenia)은 근육량·근력·기능이 감소하는 노화 관련 증후군으로,',
        '낙상, 입원, 사망률 증가와 직접 연관됩니다.',
        '표준 진단법은 **CT 영상에서 L3 레벨의 골격근 면적을 측정**하는 것이며, 이를 통해',
        '두 가지 지표를 계산합니다:',
        '',
        '- **TAMA (Total Abdominal Muscle Area)**: L3 수준 복부 총 골격근 단면적 (cm²)',
        '  - 근육량의 절대적 크기를 반영; 키·체형이 유사한 집단 내 비교에 유리',
        '- **SMI (Skeletal Muscle Index)**: TAMA ÷ 키² (cm²/m²)',
        '  - 체형을 보정한 상대적 근육량; 성별·키가 다른 집단 간 비교에 필수',
        '',
        '그러나 CT 촬영은 **방사선 피폭·고비용** 문제가 있어,',
        '일상적으로 수집 가능한 데이터만으로 근육량을 예측하는 것이 연구 목표입니다.',
        '',
        '### 1.2 AEC 피처란 무엇인가?',
        '',
        '**AEC (Automatic Exposure Control)** 는 CT 촬영 시 환자 체형에 맞춰',
        '방사선량을 자동 조절하는 기능입니다. 이 과정에서 기기는 촬영 전 각 슬라이스의',
        '신체 감쇠(attenuation) 정보를 읽어 노출값을 결정하며,',
        '이 **노출 조절 곡선(AEC curve)** 에 신체 구성 정보가 간접적으로 담겨 있습니다.',
        '',
        '즉, AEC는 "CT를 찍기 위한 준비 데이터"인데,',
        '이 안에 근육·지방·뼈의 분포 패턴이 반영되어 있어 근육량 예측에 활용 가능합니다.',
        '',
        '`aec_feature_filtered` 시트의 피처들은 AEC 곡선에서 추출한',
        '통계(평균, 표준편차, 왜도, 첨도), 주파수(FFT), 웨이블릿(Wavelet) 특징값들입니다.',
        '',
        '### 1.3 분석 방법 비교의 의의',
        '',
        '본 연구에서는 다음 5가지 입력 조합을 비교하여',
        '"어떤 피처가 근육량 예측에 가장 효과적인가"를 검증합니다:',
        '',
        '| # | 방법 | 핵심 질문 |',
        '|---|---|---|',
        '| 1 | clinic_only | 나이·성별·BMI만으로 근육량 예측이 가능한가? (기준선) |',
        '| 2 | aec_only | AEC 원시 시계열(256개)에 예측 정보가 있는가? |',
        '| 3 | aec_feature_only | 추출된 AEC 피처(65개) 전체는 얼마나 유용한가? |',
        '| 4 | aec_feature_selected | 다중공선성 제거 후 선택된 피처가 더 좋은가? |',
        '| 5 | clinic_aec_feature | 임상 정보 + AEC 피처 결합이 시너지 효과를 내는가? |',
        '',
        '---',
        '',
        '## 2. 분석 설계',
        '',
        '### 2.1 회귀 모델 선택 이유',
        '',
        '**선형 회귀 (Linear Regression)**',
        '- 목적: TAMA·SMI 수치 자체를 예측 (연속값)',
        '- 지표: R² (결정계수), RMSE (평균제곱근오차)',
        '- 의의: "AEC 피처로 실제 근육량 수치를 얼마나 정확히 맞출 수 있는가?"',
        '',
        '**로지스틱 회귀 (Logistic Regression)**',
        '- 목적: 근감소증 위험군(하위 25%) 분류 (이진 분류)',
        '- 지표: AUC-ROC, Accuracy',
        '- 의의: "AEC 피처로 근감소증 고위험 환자를 선별할 수 있는가?"',
        '- 이진화 기준: 성별 내 하위 25% → Low (1), 나머지 → Normal (0)',
        '  (성별에 따라 근육량 기준이 다르므로 성별 내 분위수 적용)',
        '',
        '### 2.2 Train/Test Split + 5-Fold Cross Validation',
        '',
        '전체 데이터를 **80% Train / 20% Test** 로 분리합니다:',
        '',
        '- **Train 80%**: 5-Fold CV로 모델 학습 및 CV 성능 추정',
        '- **Test 20%**: 최종 모델의 완전 독립적인 성능 평가 (학습에 전혀 사용되지 않음)',
        '',
        '**5-Fold CV (Train 80% 내에서만 수행)**',
        '- 80% 데이터를 5개 부분으로 나눔',
        '- 4개로 학습 → 1개로 평가를 5번 반복',
        '- CV 평균±표준편차: 학습 안정성 평가',
        '- **보고서의 최종 성능 지표는 Test 20%** 기준 (data leakage 없음)',
        '',
        '### 2.3 피처 선택 방법 (aec_feature_selected, clinic_aec_feature)',
        '',
        '65개 AEC 피처 중 상당수가 서로 높은 상관관계를 보일 수 있습니다.',
        '이런 **다중공선성(Multicollinearity)** 문제는 회귀계수 추정을 불안정하게 만듭니다.',
        '이를 해결하기 위해 각 fold의 학습 데이터에서만 다음 절차를 수행합니다:',
        '',
        '**Step 1: 상관계수 필터 (|r| > 0.8)**',
        '- AEC 피처 간 절대 상관계수가 0.8 초과인 쌍에서 후자 제거',
        '- 이유: 두 피처가 거의 같은 정보를 담고 있으면 하나만으로 충분',
        '',
        '**Step 2: VIF (Variance Inflation Factor) 반복 제거 (VIF > 10)**',
        '- VIF: 한 피처가 다른 피처들에 의해 얼마나 예측되는지를 나타내는 지표',
        '- VIF = 1/(1-R²) — R²가 높을수록 다중공선성 심각',
        '- VIF > 10인 피처 중 최고값을 반복 제거',
        '- `clinic_aec_feature`에서는 임상 피처(Age, Sex, BMI)는 고정, AEC만 제거 대상',
        '',
        '**중요**: 피처 선택을 **Train 80% 데이터에서만** 수행하여',
        'Test 20% 정보가 선택 과정에 누출되는 것을 방지합니다 (data leakage 없음).',
        '',
        '---',
        '',
        '## 3. 성능 지표 해설',
        '',
        '### R² (결정계수) — 선형 회귀',
        '',
        '> **"모델이 실제값의 변동을 몇 % 설명하는가?"**',
        '',
        '- 범위: -∞ ~ 1.0 (1에 가까울수록 좋음)',
        '- R² = 0.7 → "실제 TAMA 값 변동의 70%를 이 모델로 설명 가능"',
        '- R² ≤ 0 → 평균값을 예측하는 것보다 성능이 나쁨 (모델 실패)',
        '- 해석 기준:',
        '  - ≥ 0.7: 우수 | 0.5~0.7: 보통 | 0.3~0.5: 낮음 | < 0.3: 매우 낮음',
        '',
        '### RMSE (Root Mean Squared Error) — 선형 회귀',
        '',
        '> **"예측값이 실제값과 평균적으로 얼마나 벗어나는가?"**',
        '',
        '- 단위: 타겟 변수와 동일 (TAMA의 경우 cm², SMI의 경우 cm²/m²)',
        '- 낮을수록 좋음; R²와 함께 해석',
        '- 예: RMSE = 5.0 → "예측한 TAMA값이 실제와 평균 ±5 cm² 차이"',
        '',
        '### AUC-ROC — 로지스틱 회귀',
        '',
        '> **"근감소증 위험군과 정상군을 얼마나 잘 구별하는가?"**',
        '',
        '- 범위: 0.5 ~ 1.0 (0.5 = 동전 던지기, 1.0 = 완벽한 분류)',
        '- ROC 곡선: 모든 임계값에서 민감도(TPR) vs 특이도(1-FPR) 그래프',
        '- AUC = ROC 곡선 아래 면적 → 임계값에 무관한 종합 분류 능력',
        '- 해석 기준:',
        '  - ≥ 0.8: 우수 | 0.7~0.8: 양호 | 0.6~0.7: 미흡 | < 0.6: 무의미',
        '',
        '---',
        '',
        '## 4. 분석 결과',
        '',
    ]

    for target in TARGETS:
        target_full = 'TAMA (총 복부 근육 단면적)' if target == 'TAMA' else 'SMI (골격근 지수)'
        lines += [
            f'### 4-{TARGETS.index(target)+1}. {target_full}',
            '',
            f'#### 선형 회귀 — {target} 예측 (5-Fold CV on Train 80%)',
            '',
            '| 방법 | 설명 | CV R² (mean±std) | CV RMSE (mean±std) | Test R² |',
            '|---|---|---|---|---|',
        ]
        for method_key, method_label in METHODS:
            d = data[method_key][target]['linear']
            if not d:
                lines.append(f'| {method_key} | {method_label} | N/A | N/A | N/A |')
                continue
            r2   = f"{fmt(d.get('r2_mean'))}±{fmt(d.get('r2_std'))}"
            rmse = f"{fmt(d.get('rmse_mean'))}±{fmt(d.get('rmse_std'))}"
            test = fmt(d.get('test_r2'))
            lines.append(f'| {method_key} | {method_label} | {r2} | {rmse} | {test} |')

        lines += [
            '',
            f'#### 로지스틱 회귀 — {target} 위험군 분류 (5-Fold CV on Train 80%)',
            '',
            '> 이진화: 성별 내 하위 25% = Low (근감소증 위험), 나머지 = Normal',
            '',
            '| 방법 | 설명 | CV Accuracy (mean±std) | CV AUC-ROC (mean±std) | Test AUC |',
            '|---|---|---|---|---|',
        ]
        for method_key, method_label in METHODS:
            d = data[method_key][target]['logistic']
            if not d:
                lines.append(f'| {method_key} | {method_label} | N/A | N/A | N/A |')
                continue
            acc  = f"{fmt(d.get('acc_mean'))}±{fmt(d.get('acc_std'))}"
            auc  = f"{fmt(d.get('auc_mean'))}±{fmt(d.get('auc_std'))}"
            test = fmt(d.get('test_auc'))
            lines.append(f'| {method_key} | {method_label} | {acc} | {auc} | {test} |')

        lines += [
            '',
            f'#### ResNet1D (1D CNN) — {target} 예측 (5-Fold CV on Train 80%)',
            '',
            '| 방법 | CV R² (mean±std) | CV MAE (mean±std) | Test R² | Test MAE |',
            '|---|---|---|---|---|',
        ]
        for method_key, method_label in METHODS_RESNET1D:
            d = data[method_key][target]['resnet1d']
            if not d:
                lines.append(f'| {method_key} | N/A | N/A | N/A | N/A |')
                continue
            r2       = f"{fmt(d.get('r2_mean'))}±{fmt(d.get('r2_std'))}"
            mae      = f"{fmt(d.get('mae_mean'))}±{fmt(d.get('mae_std'))}"
            test_r2  = fmt(d.get('test_r2'))
            test_mae = fmt(d.get('test_mae'))
            lines.append(f'| {method_key} | {r2} | {mae} | {test_r2} | {test_mae} |')

        lines += ['', f'#### {target} 성별 하위 25% 임계값', '',
                  '> 로지스틱 회귀에서 "Low" 레이블 기준값 (성별별 Q25)',
                  '',
                  '| 방법 | Male 임계값 | Female 임계값 |', '|---|---|---|']
        for method_key, _ in METHODS:
            thr    = data[method_key][target]['logistic'].get('thresholds', {})
            male   = fmt(thr.get('Male'), 4)
            female = fmt(thr.get('Female'), 4)
            lines.append(f'| {method_key} | {male} | {female} |')

        lines += ['', '---', '']

    # ── 성능 해석 섹션 ─────────────────────────────────────────
    lines += [
        '## 5. 핵심 결과 및 해석',
        '',
    ]
    for target in TARGETS:
        lines += [f'### {target}', '']
        for method_key, method_label in METHODS:
            lin  = data[method_key][target]['linear']
            logi = data[method_key][target]['logistic']
            r2_v    = lin.get('test_r2',   lin.get('r2_mean',  None))
            auc_v   = logi.get('test_auc', logi.get('auc_mean', None))
            r2_str  = _interp_r2(r2_v)   if r2_v  is not None else 'N/A'
            auc_str = _interp_auc(auc_v) if auc_v is not None else 'N/A'
            lines += [
                f'**{method_key}** ({method_label})',
                f'- 선형 Test R²:   {r2_str}',
                f'- 로지스틱 Test AUC: {auc_str}',
                '',
            ]

    # ── 종합 비교 요약 ─────────────────────────────────────────
    lines += [
        '## 6. 방법 간 비교 요약',
        '',
    ]
    for target in TARGETS:
        best_r2_m, best_r2 = None, -999.0
        best_auc_m, best_auc = None, -999.0
        worst_r2_m, worst_r2 = None, 999.0
        for method_key, _ in METHODS:
            lin  = data[method_key][target]['linear']
            logi = data[method_key][target]['logistic']
            r2  = lin.get('test_r2',   lin.get('r2_mean',  -999))
            auc = logi.get('test_auc', logi.get('auc_mean', -999))
            if r2 > best_r2:
                best_r2, best_r2_m = r2, method_key
            if r2 < worst_r2 and r2 > -999:
                worst_r2, worst_r2_m = r2, method_key
            if auc > best_auc:
                best_auc, best_auc_m = auc, method_key

        # AEC feature selection vs all-feature 비교
        r2_all  = data['aec_feature_only'][target]['linear'].get('test_r2',   data['aec_feature_only'][target]['linear'].get('r2_mean',   None))
        r2_sel  = data['aec_feature_selected'][target]['linear'].get('test_r2',  data['aec_feature_selected'][target]['linear'].get('r2_mean',  None))
        auc_all = data['aec_feature_only'][target]['logistic'].get('test_auc', data['aec_feature_only'][target]['logistic'].get('auc_mean', None))
        auc_sel = data['aec_feature_selected'][target]['logistic'].get('test_auc', data['aec_feature_selected'][target]['logistic'].get('auc_mean', None))

        lines += [f'#### {target}', '']
        lines.append(f'- 🏆 **최고 Linear R²**: `{best_r2_m}` — {_interp_r2(best_r2) if best_r2 > -999 else "N/A"}')
        lines.append(f'- 🏆 **최고 AUC-ROC**: `{best_auc_m}` — {_interp_auc(best_auc) if best_auc > -999 else "N/A"}')
        lines.append(f'- 📉 **최저 Linear R²**: `{worst_r2_m}` ({fmt(worst_r2) if worst_r2 < 999 else "N/A"})')

        if r2_all is not None and r2_sel is not None:
            diff_r2 = r2_sel - r2_all
            diff_auc = (auc_sel - auc_all) if (auc_all and auc_sel) else None
            sign = '+' if diff_r2 >= 0 else ''
            lines.append(
                f'- 📊 **피처 선택 효과 (aec_feature_only → aec_feature_selected)**: '
                f'R² {sign}{diff_r2:.4f}'
                + (f', AUC {("+" if diff_auc >= 0 else "")}{diff_auc:.4f}' if diff_auc is not None else '')
            )
            if diff_r2 > 0.01:
                lines.append('  - ✅ 피처 선택으로 성능 향상 → 다중공선성 제거 효과 있음')
            elif diff_r2 < -0.01:
                lines.append('  - ⚠️ 피처 선택 후 성능 저하 → 유용한 피처가 제거됐을 가능성')
            else:
                lines.append('  - ➡️ 피처 선택 전후 성능 거의 동일 → 과적합 위험 감소 효과 기대')

        # clinic vs clinic_aec 비교
        r2_clinic = data['clinic_only'][target]['linear'].get('test_r2',   data['clinic_only'][target]['linear'].get('r2_mean',   None))
        r2_caec   = data['clinic_aec_feature'][target]['linear'].get('test_r2', data['clinic_aec_feature'][target]['linear'].get('r2_mean', None))
        if r2_clinic is not None and r2_caec is not None:
            diff = r2_caec - r2_clinic
            sign = '+' if diff >= 0 else ''
            lines.append(
                f'- 📊 **AEC 추가 효과 (clinic_only → clinic_aec_feature)**: R² {sign}{diff:.4f}'
            )
            if diff > 0.05:
                lines.append('  - ✅ 임상 변수에 AEC 피처 추가 시 유의미한 성능 향상')
            elif diff > 0:
                lines.append('  - ➡️ AEC 피처가 소폭 도움이 됨')
            else:
                lines.append('  - ⚠️ AEC 피처 추가 효과 없음 (임상 변수만으로 충분할 수 있음)')

        lines.append('')

    # ── 한계 및 향후 방향 ─────────────────────────────────────
    lines += [
        '---',
        '',
        '## 7. 한계 및 향후 연구 방향',
        '',
        '### 한계',
        '',
        '1. **단일 기관 데이터**: 강남세브란스병원 데이터만 사용 — 외부 검증(신촌 데이터) 필요',
        '2. **소규모 샘플**: 데이터 수가 적어 교차검증 결과의 분산이 클 수 있음',
        '3. **선형 모델의 한계**: 근육량과 AEC 피처 간의 비선형 관계를 포착하지 못할 수 있음',
        '4. **피처 선택의 불안정성**: fold별로 다른 피처가 선택될 수 있어 해석 일관성 저하',
        '5. **클래스 불균형**: 하위 25% 기준 이진화 시 클래스 비율 고정(1:3) — 실제 근감소증 유병률과 다를 수 있음',
        '',
        '### 향후 연구 방향',
        '',
        '1. **비선형 모델 도입**: Random Forest, XGBoost, 1D-CNN 등으로 성능 비교',
        '2. **외부 검증**: 신촌세브란스 데이터로 일반화 가능성 확인',
        '3. **AEC 피처 최적화**: SHAP 분석으로 핵심 피처 규명 후 임상 해석 강화',
        '4. **클래스 임계값 조정**: 실제 임상 기준(EWGSOP2, AWGS 등)을 적용한 이진화',
        '5. **전처리 보강**: AEC 곡선 정규화, 이상치 처리 등으로 피처 품질 향상',
        '',
        '---',
        '',
        '*본 보고서는 `4_generate_report.py`에 의해 자동 생성됩니다.*',
    ]

    return '\n'.join(lines)


# ── 메인 ─────────────────────────────────────────────────────
def main():
    print('[1] MD 파일 파싱 중...')
    data = collect_all()

    for method_key, _ in METHODS:
        for target in TARGETS:
            td = data[method_key][target]
            lin_ok = '✔' if td['lin_exists'] else '✘'
            log_ok = '✔' if td['log_exists'] else '✘'
            rn_ok  = '✔' if td['rn_exists']  else '✘'
            print(f'    {method_key}/{target}  Linear:{lin_ok}  Logistic:{log_ok}  ResNet1D:{rn_ok}')

    print('[2] 보고서 생성 중...')
    report = make_report(data)

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(report, encoding='utf-8')
    print(f'[3] 저장 완료: {OUT_REPORT}')


if __name__ == '__main__':
    main()
