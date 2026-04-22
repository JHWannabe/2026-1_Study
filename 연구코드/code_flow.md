# 전체 code Flow

1. 이미지 필터링
2. 그룹 비교
3. BMI / TAMA 분포 확인
4. 군집 분석
5. Rule-based 4분면 분석
6. ML-based silhouette 기반 분석
7. train/test 기반 K-means 평가

## 1. 이미지 필터링

### 0_image_filter.py

Excel에 있는 PatientID를 기준으로 raw/Image 폴더에서 해당 환자의 PNG만 선택해 Image_Filter 폴더로 복사

### code flow

경로 설정 -> normalize_patient_id -> load_patient_ids -> copy_matching_pngs -> save_missing_report -> remove_missing_ids_from_excel

### 입력

- Excel: Result_Filter.xlsx
- 이미지 폴더: raw/Image
- 기준 컬럼: PatientID

### 출력

- Image_Filter/ : 분석 가능한 PNG만 복사
- missing_patient_ids.csv
- Result_Filter_filtered.xlsx

## 2. 그룹 비교

### 1_aec_group_comparison.py

AEC curve를 환자별로 추출해서 그룹별 평균 shape을 비교 분석하는 코드

### 목적

환자를 성별, BMI, TAMA 기준으로 나눠서 AEC 그래프 차이를 봄

### Code flow

CFG 설정 load -> discover_default_paths / parse_args -> AEC Curve 추출 단계 extract_aec (extract_blue_curve_mask -> detect_plot_bounds -> mask_to_raw_aec -> smooth_1d) -> extract_summary_features -> load_filtered_dataset -> 그룹별 시각화/통계 저장

### AEC Cureve 추출 단계 세부 내용

1. extract_blue_curve_mask

- PNG를 HSV 색공간으로 바꾼 뒤 blue 범위를 thresholding 해서 파란선만 마스크로 뽑음, 그리고 morphological close로 선이 끊긴 부분을 메움

2. detect_plot_bounds

- 그래프 축 영역을 찾고, 회색조 변환 후 어두운 축/프레임을 이용해 left, right, top, bottom plot영역 검출
- 이 단계로 단순 픽셀 y좌표를 실제 상대 높이로 변환 가능

3. mask_to_raw_aec

- 검출된 blue mask를 실제 AEC 값으로 변환
- 각 x열마다 curve의 y값 계산
- 비어있는 x열은 interpolation으로 보간
- 전체 curve를 n_points 길이로 resample
- plot top/bottom 기준으로 0~1 상대 높이 계싼
- 그 값을 y_min ~ y_max 범위의 mA로 변환

즉, 이미지 속 파란선 -> 균일 길이의 수치 백터로 바꿔주는 핵심 함수

4. smooth_1d

- moving average로 노이즈를 줄임
- curve shape은 유지하면서 작은 흔들림을 완화하는 역할

### 입력

- Excel: Result_Filter.xlsx
- 이미지: Image_Filter/*.png

### 출력

- mean_aec.png: 평균 및 표준편차
- overlay.png: 개별 곡선 다중 overlay
- mean_differnce.png: 두 그룹 평균 차이
- feature boxplots
- group_statistics.csv
- group_members.csv
- mean_curves.csv

## 3. BMI / TAMA 분포 확인

### 2_bmi_tama.py

BMI와 TAMA 분포를 사분면 형태로 시각화하는 코드
환자들을 BMI와 TAMA 기준으로 나눠서 전체 분포, 성별별 분포를 scatter plot으로 확인

### code flow

load_bmi_tama_data -> 기준점 계산 -> build_quadrants -> draw_scatter -> save_scatter_plot -> save_scatter_plot_by_sex

### 입력

- Result_Filter.xlsx: PatientID, SRC_Report, BMI

### 출력

- bmi_tama_scatter.png
- bmi_tama_scatter_by_sex.png

## 4. 군집 분석

### 3_aec_cluster_analysis copy.py

Agglomerative Clustering 기반 AEC 군집 분석 1차 버전

### 목적

- AEC curve shape와 summary feature를 합쳐 비지도 군집화하고, 최적 cluster 수를 silhouette 기준으로 선택

### code flow

load_clinical_data -> extract_aec -> compute_aec_features -> build_aec_dataset -> prepare_feature_matrix -> select_cluster_count -> cluster_aecs

### 출력

- silhouette/inertia 곡선
- dendrogram
- detailed silhouette plot
- cluster mean AEC
- sex ratio stacked bar
- BMI/TAMA boxplot
- cluster member/summary CSV
- markdown report

### 핵심 의미 및 특징

- AEC Shape 자체가 자연스럽게 몇 개 패턴으로 나뉘는지 보는 비지도 학습 코드
- 특징
  - 계층적 군집
  - 자동 k 선택
  - 시각화가 풍부
  - train/test 개념은 없음

## 5. Rule-based 4분면 분석

### 3_aec_quadrant_shape_analysis.py

완전 비지도 군집이 아니라, BMI/TAMA rule-based로 4개의 군집을 먼저 정의하고, 그룹별 AEC shape 차이를 보는 코드

### 목적

- 환자를 먼저 임상 기준으로 4개 그룹으로 나눈뒤, 그룹별 AEC shape이 정말 다른지 확인

### 그룹 정의

- BMI cutoff = 25
- TAMA cutoff = 전체 평균

### code flow

clinical data load -> AEC 추출 -> feature 계산 -> assign_quadrant_groups -> run_statistical_tests(Kruskal-Wallis, pairwise Mann-Whitney U, Bonferroni correction)

### 출력

- 그룹별 mean AEC
- 그룹 count bar plot
- 여러 feature boxplot
- group member csv
- summary csv
- markdown report

BMI와 TAMA 기준으로 사전 정의한 4분면 그룹 간 AEC shpae feature 차이를 비교하고, 비모수 검정을 통해 유의성을 평가

## 6. ML-based silhouette 기반 분석

### 3_aec_cluster_analysis.py

위 copy 버전의 확장판/ 정리판

### copy 버전과 차이

- 최적 k 하나만 보는 게 아니라 상위 silhouette 결과 여러 개를 비교할 수 있게 확장

### 추가된 핵심 flow

- get_top_cluster_rankings
  - silhouette 점수 기준으로 상위 3개의 k를 뽑는다.
  - main에서 상위 3개의 k에 대해 반복 실행

### 출력

- detailed silhouette
- cluster mean AEC
- sex ratio
- BMI/TAMA boxplot
- cluster summary
- report

즉, 최적 k 하나만 믿지 말고, 상위 몇 개 후보를 같이 비교하는 구조

## 7. train/test 기반 K-means 평가

### 4_aec_k_means.py

train/test split을 포함한 K-means 평가 코드
이전 Agglomerative 코드와 가장 큰 차이는 train set에서 중심을 학습하고, test set은 그 중심과의 거리로 배정된다는 점

### 목적

- 클러스터링이 단지 한 번의 전체 데이터 분할에서만 보이는 현상인지, 아니면 새 데이터(test)에도 어느 정도 재현되는지 확인

### code flow

데이터 로드 -> AEC 추출 -> feature 계산 -> split_dataset -> scaling -> fit_kmeans_numpy -> train label 부여 -> **test label 부여**

### test label 부여 세부 내용

test_labels = np.argmin(compute_squared_distances(x_test, centers), axis=1)

- train에서 centroid 학습
- test는 cnetroid distance 기반 assignment
- test silhouette/inertia로 평가

### 출력

- train/test 각각에 대해:
  - silhouette plot
  - cluster mean AEC
  - sex ratio
  - BMI/TAMA boxplot
  - cluster member csv
  - cluster summary csv

### 핵심

- 군집 패턴이 train에서만 우연히 생긴 게 아닌지 확인하려는 검증 코드
- train set에서 K-means cnetroid를 추정한 뒤, test set는 centroid distance를 통해 cluster를 할당하여 군집 구조의 재현성을 평가함