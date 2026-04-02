## 1. 전체 연구 로드맵 (핵심 구조)
① Pilot Study (지금 단계)
AEC가 body composition / bone biomarker와 연관 있는지 확인
분석:
AEC 분포 (센터 / 성별 / 연령)
회귀 분석 (AEC + age + sex → muscle / bone)
정규화 방법 검토 (z-score vs percentile)

👉 목적:
AEC가 의미 있는 신호인지 검증 (feasibility 확인)

② Main Study (모델 구축 단계)
예측 모델 단계적 확장
Model A: age + sex
Model B: + bone
Model C: + muscle
Model D: + AEC
추가 분석:
Cox model (time-to-event)
interaction (AEC × muscle / bone)
level별 분석 (L1~L4)

👉 목적:
AEC가 fracture prediction에 실제로 기여하는지 검증

③ 외부 검증 (Generalization)
강남 → 신촌 검증
핵심:
AEC 센터 간 이질성 문제
calibration slope 확인

👉 목적:
모델의 실제 임상 적용 가능성 확보

④ 딥러닝 확장 (멀티모달)
CNN + Tabular 결합

구조:

CNN: CT 이미지 feature
Tabular: AEC + 임상 변수
Late fusion

👉 목적:
성능 향상 + 해석 가능성 (Grad-CAM)

⑤ 장기 연구 (확장 방향)
BMI-independent 분석
Federated learning
Serial CT (longitudinal)
Osteosarcopenia phenotype

👉 목적:
연구 확장 및 논문 파이프라인 확보

## 2. 현재 코드 상태 평가

이미 잘 되어있는 부분 (Strong)
AEC curve 추출 (HSV 기반) ✅
성별 / 그룹별 비교 ✅
BMI/TAMA 분석 ✅
클러스터링 (KMeans, Agglo) ✅

👉 결론:
"탐색적 분석 + 패턴 발견" 단계는 충분히 완료됨

부족한 부분 (Critical Gap)

❌ 통계 분석 없음

p-value 없음

단순 시각화 수준

❌ 회귀 분석 없음

AEC → muscle / bone 관계 검증 없음

❌ 비선형 분석 없음

spline / LOWESS 없음

❌ 센터 보정 없음

normalization 없음

❌ outcome 없음

fracture 연결 안됨

❌ 예측 모델 없음

Model A~D 없음

👉 결론:
지금 코드는 "논문용 분석" 단계까지는 아직 부족

## 3. 가장 중요한 핵심 전환 포인트

현재:

Clustering 기반 탐색 연구

앞으로:

Statistical + Predictive modeling 연구

## 4. 우선순위 (실제 해야 할 것)

🔴 1순위 (지금 바로)

### 1. 회귀 분석 코드

muscle ~ AEC + age + sex

bone   ~ AEC + age + sex

👉 이게 Pilot Study의 핵심

### 2. p-value 추가

Mann-Whitney U

Bonferroni correction

🟠 2순위 (단기)

### 3. spline 분석

AEC vs attenuation (비선형 확인)

### 4. AEC 정규화

z-score (센터별)

percentile

👉 이걸 안 하면 다기관 비교 불가능

🟡 3순위 (중기)

### 5. fracture 연결

label 추가

logistic / Cox

### 6. Model A → D 구축

AUC 비교

calibration

🟢 4순위 (장기)

외부 검증 (강남 → 신촌)

CNN multimodal

longitudinal 분석

## 5. 한 줄 핵심 요약

👉 지금 상태
→ "AEC 패턴은 봤다"

👉 다음 단계
→ "AEC가 진짜 임상적으로 의미 있는지 증명해야 한다"

## 6. 실전 조언 (중요)

현재 흐름에서 가장 위험한 포인트:

❌ clustering만 계속 하는 것

이건 논문 임팩트가 약함

반대로 가장 중요한 방향:

✅ regression + outcome 연결

결론

이 연구의 핵심 축은 하나다:

"AEC가 기존 변수(age/sex/bone/muscle)보다 추가적인 정보를 주는가?"