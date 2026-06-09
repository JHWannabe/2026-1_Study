from fpdf import FPDF
import os

FONT_REG = r"C:\Windows\Fonts\malgun.ttf"
FONT_BOLD = r"C:\Windows\Fonts\malgunbd.ttf"
OUTPUT = os.path.join(os.path.dirname(__file__), "AUC향상_개선방안.pdf")

GRAY_BG   = (245, 245, 245)
CODE_BG   = (30,  30,  30)
BLUE      = (30,  80,  160)
ORANGE    = (200, 90,  0)
GREEN     = (20,  120, 60)
RED_SOFT  = (180, 40,  40)
WHITE     = (255, 255, 255)
BLACK     = (30,  30,  30)
LIGHT_BLUE= (220, 232, 250)
BORDER    = (180, 180, 180)


class PDF(FPDF):
    def header(self):
        pass

    def footer(self):
        self.set_y(-13)
        self.set_font("reg", size=8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 8, f"{self.page_no()}", align="C")

    def add_fonts(self):
        self.add_font("reg",  "", FONT_REG,  uni=True)
        self.add_font("bold", "", FONT_BOLD, uni=True)

    # ── helpers ──────────────────────────────────────────────
    def _rgb(self, color): return color[0], color[1], color[2]

    def h1(self, text):
        self.set_font("bold", size=18)
        self.set_text_color(*BLUE)
        self.ln(2)
        self.multi_cell(0, 9, text)
        self.set_draw_color(*BLUE)
        self.set_line_width(0.6)
        self.line(self.l_margin, self.get_y()+1, self.w - self.r_margin, self.get_y()+1)
        self.ln(5)

    def h2(self, text):
        self.set_font("bold", size=13)
        self.set_text_color(*ORANGE)
        self.ln(4)
        self.multi_cell(0, 7, text)
        self.ln(1)

    def h3(self, text):
        self.set_font("bold", size=11)
        self.set_text_color(*BLACK)
        self.ln(3)
        self.multi_cell(0, 6, text)
        self.ln(1)

    def body(self, text, size=10):
        self.set_font("reg", size=size)
        self.set_text_color(*BLACK)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def bullet(self, text, indent=5):
        self.set_font("reg", size=10)
        self.set_text_color(*BLACK)
        x0 = self.get_x()
        self.set_x(self.l_margin + indent)
        self.cell(4, 5.5, "•")
        self.multi_cell(0, 5.5, text)
        self.set_x(x0)

    def code_block(self, lines):
        self.ln(2)
        x, y = self.l_margin, self.get_y()
        w = self.w - self.l_margin - self.r_margin
        line_h = 5.0
        total_h = len(lines) * line_h + 6
        self.set_fill_color(*CODE_BG)
        self.rect(x, y, w, total_h, "F")
        self.set_font("reg", size=8.5)
        self.set_text_color(*WHITE)
        self.set_y(y + 3)
        for line in lines:
            self.set_x(x + 4)
            self.cell(w - 8, line_h, line, ln=1)
        self.ln(3)
        self.set_text_color(*BLACK)

    def info_box(self, text, bg=LIGHT_BLUE):
        self.ln(2)
        x, y = self.l_margin, self.get_y()
        w = self.w - self.l_margin - self.r_margin
        self.set_font("reg", size=9.5)
        # measure height
        n = self.get_string_width(text) / (w - 10) + 1
        h = max(12, int(n) * 5.5 + 8)
        self.set_fill_color(*bg)
        self.set_draw_color(*BORDER)
        self.set_line_width(0.3)
        self.rect(x, y, w, h, "FD")
        self.set_text_color(*BLACK)
        self.set_xy(x + 5, y + 4)
        self.multi_cell(w - 10, 5.5, text)
        self.ln(3)

    def table_row(self, cols, widths, bold=False, bg=WHITE):
        self.set_fill_color(*bg)
        self.set_font("bold" if bold else "reg", size=9)
        self.set_text_color(*BLACK)
        x0 = self.l_margin
        self.set_x(x0)
        for txt, w in zip(cols, widths):
            self.set_x(self.get_x())
            self.cell(w, 7, txt, border=1, fill=True)
        self.ln()

    def priority_badge(self, text, color):
        self.set_font("bold", size=9)
        self.set_fill_color(*color)
        self.set_text_color(*WHITE)
        self.set_draw_color(*color)
        tw = self.get_string_width(text) + 8
        self.cell(tw, 7, text, fill=True, border=0)
        self.set_text_color(*BLACK)
        self.ln(2)


def make_pdf():
    pdf = PDF(orientation="P", unit="mm", format="A4")
    pdf.add_fonts()
    pdf.set_margins(18, 18, 18)
    pdf.set_auto_page_break(auto=True, margin=18)

    # ════════════════════════════════════════════
    # COVER PAGE
    # ════════════════════════════════════════════
    pdf.add_page()
    pdf.set_fill_color(*BLUE)
    pdf.rect(0, 0, 210, 60, "F")

    pdf.set_y(14)
    pdf.set_font("bold", size=22)
    pdf.set_text_color(*WHITE)
    pdf.cell(0, 12, "AUC 0.9+ 달성을 위한 모델 개선 방안", align="C", ln=1)
    pdf.set_font("reg", size=12)
    pdf.cell(0, 8, "근감소증 예측 모델 (Sarcopenia Detection) 연구", align="C", ln=1)
    pdf.set_font("reg", size=10)
    pdf.cell(0, 7, "Model1 AUC 0.803 → Model2/3 AUC 0.9+ 목표", align="C", ln=1)

    pdf.set_y(70)
    pdf.set_text_color(*BLACK)

    # 현황 요약 박스
    pdf.h1("현황 분석")
    pdf.body("현재 5개 모델의 성능은 아래와 같으며, AEC 추가 모델이 베이스라인(M1)을 유의미하게 능가하지 못하고 있습니다.")
    pdf.ln(2)

    # 현황 테이블
    widths = [18, 38, 42, 22, 22, 24]
    headers = ["모델", "입력 피처", "아키텍처", "Test AUC", "AUPRC", "문제점"]
    rows = [
        ("M1", "Age / Sex / BMI", "Logistic Regression", "0.803", "0.312", "피처 3개뿐"),
        ("M2", "Clinic + AEC (matched)", "Cross-Attention", "0.802", "0.346", "M1보다 낮음"),
        ("M2_2", "Clinic + AEC (unmatched)", "Cross-Attention", "0.841", "0.354", "Negative control 역전"),
        ("M3", "Clinic + Scanner + AEC", "3-way CrossAttn", "0.830", "0.337", "유의 향상 없음"),
        ("M4", "AEC Only", "ResNet1D", "0.604", "0.208", "AEC 단독 신호 부족"),
    ]
    row_bgs = [LIGHT_BLUE, (255,230,230), (255,230,230), (230,245,230), GRAY_BG]
    pdf.table_row(headers, widths, bold=True, bg=(50,80,160))
    # header white text hack
    for i, (row, bg) in enumerate(zip(rows, row_bgs)):
        pdf.table_row(list(row), widths, bg=bg)
    pdf.ln(3)

    pdf.info_box(
        "핵심 문제: AEC 시퀀스 raw 사용 시 정보량 부족(M4 AUC=0.60), "
        "Cross-Attention 융합이 임상 피처 3개를 보완하지 못함. "
        "M2_2(unmatched)가 M2(matched)를 능가하는 이상 현상 존재.",
        bg=(255, 245, 210)
    )

    # ════════════════════════════════════════════
    # PAGE 2 — 1순위
    # ════════════════════════════════════════════
    pdf.add_page()
    pdf.h1("개선 방안")
    pdf.priority_badge("1순위 — 피처 확장  (가장 빠른 AUC 향상 기대)", GREEN)
    pdf.ln(2)

    pdf.h2("[A]  AEC Hand-Crafted 피처 추출 후 M1에 Concat")
    pdf.body(
        "M4(AEC Only)의 AUC가 0.60에 불과한 것은 128점 raw 시퀀스를 그대로 "
        "신경망에 입력하면 유용한 패턴이 소실됨을 의미합니다. "
        "통계적 요약 피처를 추출해 임상 피처와 concat하면 deep model 없이도 AUC 향상이 가능합니다."
    )

    pdf.code_block([
        "import numpy as np, scipy.stats",
        "",
        "def extract_aec_features(aec_seq):  # shape: (128,)",
        "    return {",
        "        'aec_mean'      : aec_seq.mean(),",
        "        'aec_std'       : aec_seq.std(),",
        "        'aec_max'       : aec_seq.max(),",
        "        'aec_min'       : aec_seq.min(),",
        "        'aec_peak_pos'  : aec_seq.argmax() / 128,   # 정규화 peak 위치",
        "        'aec_auc'       : np.trapz(aec_seq),        # 곡선 아래 면적",
        "        'aec_skew'      : scipy.stats.skew(aec_seq),",
        "        'aec_kurt'      : scipy.stats.kurtosis(aec_seq),",
        "        # 구간별 통계 (early / mid / late attenuation)",
        "        'aec_early_mean': aec_seq[:43].mean(),",
        "        'aec_mid_mean'  : aec_seq[43:86].mean(),",
        "        'aec_late_mean' : aec_seq[86:].mean(),",
        "    }",
        "",
        "# 임상 피처와 concat → Logistic Regression / XGBoost에 입력",
        "aec_feats = extract_aec_features(aec_seq)   # 11개 피처",
        "X_combined = np.concatenate([clin_feats, list(aec_feats.values())])",
    ])

    pdf.body("추출되는 11개 피처의 의미:")
    items = [
        "aec_mean / std : 전체 감쇠 수준 및 변동성",
        "aec_max / min  : 피크 감쇠값 및 기저 감쇠값",
        "aec_peak_pos   : 최대 감쇠가 발생하는 시점 (조영제 도달 속도)",
        "aec_auc        : 총 조영제 통과량 (심박출량 대리 지표)",
        "aec_skew / kurt: 감쇠 곡선의 비대칭도 및 첨도",
        "early/mid/late : 조영 전기·중기·후기 평균 (동맥상·문맥상·평형상)",
    ]
    for it in items:
        pdf.bullet(it)
    pdf.ln(2)

    pdf.info_box(
        "기대 효과: 임상 피처(3개) + AEC 통계 피처(11개) = 총 14개 피처로 LR/XGBoost 학습 시 "
        "AUC 0.84~0.88 수준 달성 가능 (문헌 유사 연구 기준). "
        "구현이 단순하고 해석 가능성이 높아 논문 기여도도 큼.",
        bg=(220, 245, 220)
    )

    # ════════════════════════════════════════════
    # PAGE 3 — 2순위
    # ════════════════════════════════════════════
    pdf.add_page()
    pdf.priority_badge("2순위 — 모델 구조 개선", (100, 60, 180))
    pdf.ln(2)

    pdf.h2("[B]  Cross-Attention 대신 1D CNN + Late Fusion")
    pdf.body(
        "현재 Cross-Attention의 문제: AEC 시퀀스(128점) vs 임상 피처(3개)의 "
        "극단적 차원 불균형으로 attention이 수렴하지 못합니다. "
        "1D CNN으로 지역 패턴을 먼저 압축한 후 임상 피처와 late fusion하는 구조가 더 안정적입니다."
    )

    pdf.code_block([
        "class ImprovedFusionModel(nn.Module):",
        "    def __init__(self, n_clin=3):",
        "        super().__init__()",
        "        # AEC 인코더: 128 → 지역 패턴 압축",
        "        self.aec_encoder = nn.Sequential(",
        "            nn.Conv1d(1, 32, kernel_size=7, padding=3),",
        "            nn.BatchNorm1d(32), nn.ReLU(),",
        "            nn.MaxPool1d(2),                   # 128 → 64",
        "            nn.Conv1d(32, 64, kernel_size=5, padding=2),",
        "            nn.BatchNorm1d(64), nn.ReLU(),",
        "            nn.AdaptiveAvgPool1d(8),           # → 64 x 8 = 512-dim",
        "            nn.Flatten()",
        "        )",
        "        # 임상 피처 MLP",
        "        self.clin_encoder = nn.Sequential(",
        "            nn.Linear(n_clin, 32), nn.ReLU(), nn.Dropout(0.3),",
        "            nn.Linear(32, 64),     nn.ReLU()",
        "        )",
        "        # Late Fusion 분류기",
        "        self.classifier = nn.Sequential(",
        "            nn.Linear(512 + 64, 128), nn.ReLU(), nn.Dropout(0.4),",
        "            nn.Linear(128, 1)",
        "        )",
        "",
        "    def forward(self, aec, clin):",
        "        aec_feat  = self.aec_encoder(aec.unsqueeze(1))  # (B, 512)",
        "        clin_feat = self.clin_encoder(clin)             # (B, 64)",
        "        return self.classifier(torch.cat([aec_feat, clin_feat], dim=1))",
    ])

    pdf.ln(2)
    pdf.h2("[C]  Contrastive Pre-training (데이터가 충분할 경우)")
    pdf.body(
        "동일 환자의 다회 촬영 AEC를 positive pair로, 다른 환자를 negative pair로 설정하여 "
        "SimCLR / SupCon 방식으로 AEC 인코더를 사전학습합니다. "
        "이후 frozen encoder + classifier fine-tuning으로 소규모 레이블 데이터에서도 "
        "강건한 표현 학습이 가능합니다."
    )
    pdf.bullet("적용 조건: 동일 환자 2회 이상 촬영 데이터 존재 여부 확인 필요")
    pdf.bullet("대안: AEC augmentation(noise/mask)을 positive pair로 사용하는 self-supervised 방식")
    pdf.ln(2)

    pdf.info_box(
        "구조 개선 기대 효과: 1D CNN Late Fusion → AUC 0.85~0.88 (Cross-Attention 대비 +0.03~0.05). "
        "Contrastive Pre-training → 데이터 크기에 따라 추가 +0.02~0.04.",
        bg=(235, 225, 250)
    )

    # ════════════════════════════════════════════
    # PAGE 4 — 3순위
    # ════════════════════════════════════════════
    pdf.add_page()
    pdf.priority_badge("3순위 — 학습 전략 개선", (160, 80, 0))
    pdf.ln(2)

    pdf.h2("[D]  하이퍼파라미터 조정")
    pdf.body("현재 config.py의 설정과 권장 변경값:")

    widths2 = [45, 40, 40, 45]
    pdf.table_row(["파라미터", "현재값", "권장값", "이유"], widths2, bold=True, bg=(50,80,160))
    param_rows = [
        ("LR_RATE",     "1e-5",  "3e-4",      "너무 낮아 수렴 속도 저하"),
        ("WEIGHT_DECAY","기본값", "1e-3",      "AdamW 정규화 강화"),
        ("BATCH_SIZE",  "32",    "16",         "소규모 데이터 시 더 많은 업데이트"),
        ("EPOCHS",      "500",   "300 + ES",   "Early Stopping으로 과적합 방지"),
        ("N_HEADS",     "1",     "2 또는 4",   "Multi-head attention 활성화"),
    ]
    for i, row in enumerate(param_rows):
        bg = GRAY_BG if i % 2 == 0 else WHITE
        pdf.table_row(list(row), widths2, bg=bg)
    pdf.ln(4)

    pdf.code_block([
        "# config.py 권장 수정",
        "LR_RATE     = 3e-4       # 1e-5 → 3e-4",
        "WEIGHT_DECAY = 1e-3      # AdamW L2 정규화",
        "BATCH_SIZE  = 16         # 소규모 데이터 최적화",
        "N_HEADS     = 4          # Multi-head attention",
        "",
        "# train_eval.py — Early Stopping 추가",
        "patience, best_val_auc, wait = 30, 0, 0",
        "for epoch in range(EPOCHS):",
        "    ...  # train",
        "    if val_auc > best_val_auc:",
        "        best_val_auc, wait = val_auc, 0",
        "        torch.save(model.state_dict(), ckpt_path)",
        "    else:",
        "        wait += 1",
        "        if wait >= patience: break",
    ])

    pdf.h2("[E]  클래스 불균형 처리 강화")
    pdf.body(
        "현재 AUPRC=0.31은 심각한 클래스 불균형을 의미합니다. "
        "Focal Loss는 이미 사용 중이나 pos_weight가 명시적으로 설정되지 않았을 수 있습니다."
    )

    pdf.code_block([
        "# train_eval.py — pos_weight 명시 계산",
        "n_neg = (y_train == 0).sum()",
        "n_pos = (y_train == 1).sum()",
        "pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32).to(device)",
        "",
        "# FocalLossWithLogits에 전달",
        "criterion = FocalLossWithLogits(gamma=2.0, pos_weight=pos_weight)",
        "",
        "# 추가: SMOTE 적용 (선택적)",
        "from imblearn.over_sampling import SMOTE",
        "X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)",
    ])

    # ════════════════════════════════════════════
    # PAGE 5 — 앙상블 + 실행 순서
    # ════════════════════════════════════════════
    pdf.add_page()
    pdf.priority_badge("4순위 — 앙상블 (마무리 성능 향상)", RED_SOFT)
    pdf.ln(2)

    pdf.h2("[F]  모델 앙상블")
    pdf.body(
        "각 모델의 예측 확률을 가중 평균하면 단일 모델 대비 AUC를 0.01~0.03 추가로 향상시킬 수 있습니다. "
        "특히 M1(LR)과 M3(Deep)처럼 서로 다른 귀납 편향을 가진 모델의 앙상블이 효과적입니다."
    )

    pdf.code_block([
        "# 테스트 시 예측 확률 앙상블",
        "prob_m1 = model1.predict_proba(X_test)[:, 1]   # Logistic Regression",
        "prob_m3 = torch.sigmoid(model3(aec, clin)).cpu().numpy().flatten()",
        "",
        "# 가중 평균 (AUC 기반 가중치)",
        "w1, w3 = 0.35, 0.65",
        "prob_ensemble = w1 * prob_m1 + w3 * prob_m3",
        "",
        "# 또는 Stacking (Meta-learner)",
        "from sklearn.linear_model import LogisticRegression",
        "meta_X = np.column_stack([prob_m1, prob_m3])",
        "meta_clf = LogisticRegression().fit(meta_X_val, y_val)",
    ])

    pdf.ln(4)
    pdf.h1("권장 실행 순서 (로드맵)")

    steps = [
        ("Step 1", "Excel 원본 컬럼 확인",
         "강남_liver_merged_features_ok.xlsx의 전체 컬럼 목록 확인.\n"
         "SMI 외 추가 임상/lab 수치 존재 시 CLIN_COLS에 추가."),
        ("Step 2", "AEC Hand-Crafted 피처 추출",
         "extract_aec_features() 구현 → M1(LR)에 concat하여 AUC 확인.\n"
         "목표: 0.84+ 달성 여부 판단."),
        ("Step 3", "학습 전략 수정",
         "LR=3e-4, pos_weight 명시, Early Stopping 추가.\n"
         "기존 M2/M3 재학습 후 성능 비교."),
        ("Step 4", "1D CNN Late Fusion 모델 구현",
         "ImprovedFusionModel 코드 작성 → 5-Fold CV 비교.\n"
         "목표: AUC 0.87+ 달성."),
        ("Step 5", "앙상블로 마무리",
         "Step 2 모델 + Step 4 모델 앙상블.\n"
         "목표: AUC 0.90+ 달성."),
    ]

    colors_step = [BLUE, GREEN, ORANGE, (100,60,180), RED_SOFT]
    for (tag, title, desc), color in zip(steps, colors_step):
        x, y = pdf.l_margin, pdf.get_y()
        pdf.set_fill_color(*color)
        pdf.rect(x, y, 22, 14, "F")
        pdf.set_font("bold", size=9)
        pdf.set_text_color(*WHITE)
        pdf.set_xy(x + 1, y + 4)
        pdf.cell(20, 6, tag, align="C")

        pdf.set_font("bold", size=10)
        pdf.set_text_color(*color)
        pdf.set_xy(x + 25, y + 1)
        pdf.cell(0, 6, title, ln=1)

        pdf.set_font("reg", size=9)
        pdf.set_text_color(*BLACK)
        pdf.set_x(x + 25)
        pdf.multi_cell(pdf.w - pdf.r_margin - x - 25, 5, desc)
        pdf.ln(3)

    # ── 최종 요약 ──
    pdf.ln(2)
    pdf.set_fill_color(*LIGHT_BLUE)
    pdf.set_draw_color(*BLUE)
    pdf.set_line_width(0.5)
    x, y = pdf.l_margin, pdf.get_y()
    pdf.rect(x, y, pdf.w - pdf.l_margin - pdf.r_margin, 28, "FD")
    pdf.set_xy(x + 5, y + 4)
    pdf.set_font("bold", size=11)
    pdf.set_text_color(*BLUE)
    pdf.cell(0, 7, "최종 목표 시나리오", ln=1)
    pdf.set_x(x + 5)
    pdf.set_font("reg", size=10)
    pdf.set_text_color(*BLACK)
    pdf.multi_cell(
        pdf.w - pdf.l_margin - pdf.r_margin - 10, 5.5,
        "Step 1~2 완료 후 AUC 0.85+ 달성 → Step 3~4로 0.87~0.88 → "
        "Step 5 앙상블로 0.90+ 목표 달성.\n"
        "각 Step별 5-Fold CV + DeLong Test로 통계적 유의성 검증 필수."
    )

    pdf.output(OUTPUT)
    print(f"PDF saved: {OUTPUT}")


if __name__ == "__main__":
    make_pdf()
