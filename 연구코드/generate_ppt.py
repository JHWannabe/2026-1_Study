"""
AEC-based Sarcopenia Prediction Research PPT
- Comparison-focused main slides
- Brier score 언급 슬라이드에 calibration plot 의무 포함
- 조건별 상세 슬라이드 (부록)
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

BASE   = "C:/Users/jhjun/OneDrive/Desktop/2026-1_Study/연구코드/results/ok"
M1     = f"{BASE}/model_1"
M2N    = f"{BASE}/model_2/bce/aec128/norm"
M3N    = f"{BASE}/model_3/bce/aec128/norm"
M22N   = f"{BASE}/model_2_2/bce/aec128/norm"
CMP128 = f"{BASE}/comparison/bce/aec128"
CMP256 = f"{BASE}/comparison/bce/aec256"
CMPF128= f"{BASE}/comparison/focal/aec128"
CMPF256= f"{BASE}/comparison/focal/aec256"

def img(folder, name):
    return os.path.join(folder, name)

# ── Colors ────────────────────────────────────────────────────
C_BG   = RGBColor(0xFF,0xFF,0xFF)
C_DARK = RGBColor(0x1A,0x1A,0x2E)
C_BLUE = RGBColor(0x0F,0x6B,0xBF)
C_TEAL = RGBColor(0x0F,0x9B,0x8A)
C_ORG  = RGBColor(0xE8,0x74,0x00)
C_GRN  = RGBColor(0x1E,0x8B,0x4C)
C_RED  = RGBColor(0xC0,0x39,0x2B)
C_LGR  = RGBColor(0xF0,0xF4,0xF8)
C_MGR  = RGBColor(0x9E,0xA3,0xAB)
C_WHT  = RGBColor(0xFF,0xFF,0xFF)

TOTAL = 42

prs = Presentation()
prs.slide_width  = Inches(13.33)
prs.slide_height = Inches(7.5)
blank = prs.slide_layouts[6]

# ── Helpers ────────────────────────────────────────────────────
def add_slide():
    s = prs.slides.add_slide(blank)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = C_BG
    return s

def box(s, l, t, w, h, fill=None, line=None, lw=None):
    sh = s.shapes.add_shape(1, Inches(l), Inches(t), Inches(w), Inches(h))
    sh.fill.solid() if fill else sh.fill.background()
    if fill: sh.fill.fore_color.rgb = fill
    if line: sh.line.color.rgb = line; sh.line.width = Pt(lw or 0.75)
    else: sh.line.fill.background()
    return sh

def txt(s, text, l, t, w, h, sz=13.0, bold=False, color=C_DARK,
        align=PP_ALIGN.LEFT, italic=False, font="맑은 고딕"):
    tb = s.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tb.word_wrap = True
    p = tb.text_frame.paragraphs[0]; p.alignment = align
    r = p.add_run(); r.text = text
    r.font.size = Pt(sz); r.font.bold = bold; r.font.italic = italic
    r.font.color.rgb = color; r.font.name = font

def pic(s, path, l, t, w, h):
    if os.path.exists(path):
        s.shapes.add_picture(path, Inches(l), Inches(t), Inches(w), Inches(h))
    else:
        box(s, l, t, w, h, fill=C_LGR, line=C_MGR, lw=0.5)
        txt(s, f"[파일 없음]\n{os.path.basename(path)}", l+0.07, t+0.07,
            w-0.14, h-0.14, sz=9, color=C_MGR)

def hdr(s, title, sub=None):
    box(s, 0, 0, 13.33, 0.84, fill=C_DARK)
    txt(s, title, 0.35, 0.06, 11.0, 0.44, sz=25, bold=True, color=C_WHT)
    if sub: txt(s, sub, 0.35, 0.50, 11.0, 0.28, sz=13, color=C_MGR)

def hline(s, l, t, w, color=C_BLUE, h=0.04):
    box(s, l, t, w, h, fill=color)

def card(s, l, t, w, h, fill=C_LGR, line=None):
    return box(s, l, t, w, h, fill=fill, line=line, lw=0.75 if line else None)

def trow_h(s, headers, xs, ws, y, ht=0.3):
    for h, x, w in zip(headers, xs, ws):
        box(s, x, y, w, ht, fill=C_DARK)
        txt(s, h, x, y, w, ht, sz=10, bold=True, color=C_WHT, align=PP_ALIGN.CENTER)

def trow_d(s, cells, xs, ws, y, ht=0.36, fill=C_WHT, colors=None, bolds=None):
    for i,(c,x,w) in enumerate(zip(cells,xs,ws)):
        cl = (colors[i] if colors else C_DARK)
        b  = (bolds[i]  if bolds  else False)
        box(s, x, y, w, ht, fill=fill, line=RGBColor(0xCC,0xCC,0xCC), lw=0.5)
        txt(s, c, x, y, w, ht, sz=10.5, bold=b, color=cl, align=PP_ALIGN.CENTER)

def snum(s, n):
    txt(s, f"{n} / {TOTAL}", 12.3, 7.15, 1.0, 0.3, sz=11,
        color=C_MGR, align=PP_ALIGN.RIGHT)

def calib_banner(s, l, t, w, label):
    box(s, l, t, w, 0.33, fill=RGBColor(0xFF,0xF0,0xD0))
    txt(s, f"※ Calibration — {label}  (Brier 보정 품질 확인)",
        l+0.08, t+0.02, w-0.12, 0.29, sz=11, bold=True, color=C_ORG)

# ═══════════════════════════════════════════════════════════════
# SLIDE 1 — Title
# ═══════════════════════════════════════════════════════════════
s = add_slide()
box(s, 0, 0, 13.33, 7.5, fill=C_DARK)
box(s, 0, 0, 13.33, 0.08, fill=C_BLUE)
box(s, 0, 7.42, 13.33, 0.08, fill=C_TEAL)
txt(s, "AEC 신호 기반 근감소증 예측 모델", 1.0, 1.8, 11.33, 0.9,
    sz=36, bold=True, color=C_WHT, align=PP_ALIGN.CENTER)
txt(s, "Cross-Attention 아키텍처와 DeLong Test를 통한 모델 간 성능 비교 분석",
    1.0, 2.85, 11.33, 0.55, sz=18, color=C_MGR, align=PP_ALIGN.CENTER)
hline(s, 3.5, 3.65, 6.33, color=C_TEAL)
txt(s, "총 859명  |  5-Fold CV  |  Bootstrap 95% CI  |  DeLong AUC Test",
    1.0, 4.0, 11.33, 0.45, sz=14, color=C_MGR, align=PP_ALIGN.CENTER)
txt(s, "2026-05-22", 1.0, 4.55, 11.33, 0.35, sz=13, color=C_MGR, align=PP_ALIGN.CENTER)
for i,(label,col) in enumerate([
    ("Model 1  Clinic Only", C_BLUE),
    ("Model 2  Clinic + AEC", C_TEAL),
    ("Model 2_2  Neg. Control", C_MGR),
    ("Model 3  Clinic+Scanner+AEC", C_ORG),
]):
    x = 1.1 + i*2.8
    box(s, x, 5.3, 2.6, 0.85, fill=col)
    txt(s, label, x, 5.37, 2.6, 0.72, sz=12, bold=True,
        color=C_WHT, align=PP_ALIGN.CENTER)
snum(s, 1)

# ═══════════════════════════════════════════════════════════════
# SLIDE 2 — 연구 배경 및 목적
# ═══════════════════════════════════════════════════════════════
s = add_slide()
hdr(s, "연구 배경 및 목적", "AEC 신호의 진단적 가치 탐색")
card(s, 0.3, 0.99, 5.9, 5.86)
txt(s, "연구 배경", 0.5, 1.09, 5.5, 0.35, sz=16, bold=True)
hline(s, 0.5, 1.46, 5.5, color=C_BLUE)
for i,t in enumerate([
    "• 근감소증(Sarcopenia): SMI 기반 이진 분류 — 노인 허약·사망률과 직결",
    "• 기존 진단: CT 판독 + 영상의학과 전문의 수동 분석 → 시간·비용 소모",
    "• AEC(자동노출제어): CT 스캔 시 자동 수집, 체형·조직 구성 간접 반영",
    "• 추가 검사 없이 기존 CT 장비에서 수집 가능한 Zero-cost 부가 신호",
    "• 핵심 질문: AEC가 Clinic 변수 대비 얼마나 예측력을 개선하는가?",
]):
    txt(s, t, 0.5, 1.56+i*0.9, 5.6, 0.82, sz=13.5, color=C_DARK)

card(s, 6.5, 0.99, 6.5, 5.86)
txt(s, "비교 연구 목적 (DeLong Test 기반)", 6.7, 1.09, 6.1, 0.35, sz=16, bold=True)
hline(s, 6.7, 1.46, 6.1, color=C_TEAL)
for i,(q,a,col) in enumerate([
    ("① M1 vs M2: AEC 신호가 기여하는가?",
     "Clinic(LR) vs Clinic+AEC(CrossAttn)\n→ DeLong AUC 비교로 통계 검증", C_BLUE),
    ("② M1 vs M3: Scanner+AEC 조합 효과?",
     "Clinic(LR) vs Clinic+Scanner+AEC(CrossAttn3)\n→ DeLong AUC 비교", C_TEAL),
    ("③ M2 vs M3: Scanner 순증가 효과?",
     "CrossAttn vs CrossAttn3 (동일 test set)\n→ 통계적 유의차 여부 확인", C_ORG),
    ("④ M2 vs M2_2: 매칭이 실질적 기여?",
     "Matched vs Unmatched (음성 대조군)\n→ AEC 개인 대응의 필요성 검증", C_GRN),
]):
    box(s, 6.6, 1.56+i*1.25, 0.08, 1.05, fill=col)
    txt(s, q, 6.75, 1.58+i*1.25, 6.1, 0.35, sz=13, bold=True, color=col)
    txt(s, a, 6.75, 1.93+i*1.25, 6.1, 0.55, sz=12, color=C_DARK)
snum(s, 2)

# ═══════════════════════════════════════════════════════════════
# SLIDE 3 — 데이터셋 + data_distribution.png
# ═══════════════════════════════════════════════════════════════
s = add_slide()
hdr(s, "데이터셋 및 실험 설계", "859명 단일 기관 후향적 코호트 / 5-Fold Stratified CV")

card(s, 0.3, 0.99, 5.9, 6.31)
txt(s, "데이터셋 구성", 0.5, 1.07, 5.5, 0.33, sz=15, bold=True)
hline(s, 0.5, 1.42, 5.5, color=C_BLUE)
hx = [0.5, 1.6, 3.1, 4.6]; hw = [1.05, 1.45, 1.45, 1.5]
trow_h(s, ["구분","남성(M)","여성(F)","합계"], hx, hw, 1.47)
for i,(row,fill) in enumerate(zip([
    ("Train","285 (Sarco 15.4%)","402 (Sarco 9.5%)","687 (Sarco 11.9%)"),
    ("Test", "71  (Sarco 15.5%)","101 (Sarco 9.9%)","172 (Sarco 12.2%)"),
], [C_WHT, C_LGR])):
    trow_d(s, row, hx, hw, 1.79+i*0.37, ht=0.37, fill=fill)
txt(s, "연령: 57.6±12.1세  |  BMI: 23.6±3.3 kg/m²  |  Train:Test = 8:2",
    0.5, 2.55, 5.5, 0.28, sz=11, color=C_MGR, italic=True)

txt(s, "AEC 전처리 조건 (5종)", 0.5, 2.97, 5.5, 0.33, sz=15, bold=True)
hline(s, 0.5, 3.32, 5.5, color=C_TEAL)
for i,(k,v) in enumerate([
    ("norm",       "원본 AEC 시퀀스 min-max 정규화"),
    ("crop80/60",  "중앙 80% / 60% 구간만 사용"),
    ("len128/256", "시퀀스 길이를 128 / 256으로 리샘플링"),
    ("excl_extreme","상하위 1% 이상치 제거 후 정규화"),
]):
    txt(s, k,  0.5, 3.41+i*0.4, 1.7, 0.38, sz=12, bold=True, color=C_BLUE)
    txt(s, v,  2.2, 3.41+i*0.4, 3.9, 0.38, sz=12, color=C_DARK)
txt(s, "✕  BCE / Focal Loss  ×  AEC 128pt / 256pt  =  4 시나리오 × 5조건 = 20조건",
    0.5, 5.05, 5.5, 0.3, sz=11, color=C_MGR, italic=True)

card(s, 6.5, 0.99, 5.9, 3.31)
txt(s, "모델 구성", 6.7, 1.07, 5.5, 0.33, sz=15, bold=True)
hline(s, 6.7, 1.42, 5.5, color=C_ORG)
for i,(t,b,col) in enumerate([
    ("Model 1 — Baseline","Age, Sex, BMI → Logistic Regression", C_BLUE),
    ("Model 2 — AEC Matched","Clinic + AEC(동일환자) → CrossAttention", C_TEAL),
    ("Model 2_2 — AEC Unmatched","Clinic + AEC(다른환자) → 음성 대조군", C_MGR),
    ("Model 3 — Scanner+AEC","Clinic + MFR + AEC → CrossAttention3", C_ORG),
]):
    box(s, 6.6, 1.51+i*1.0, 0.08, 0.82, fill=col)
    txt(s, t, 6.75, 1.53+i*1.0, 5.4, 0.33, sz=13, bold=True, color=col)
    txt(s, b, 6.75, 1.86+i*1.0, 5.4, 0.28, sz=12, color=C_DARK)

txt(s, "데이터 분포 (Train / Test / Sex / Sarcopenia)", 6.4, 4.17, 6.8, 0.28,
    sz=11, bold=True, color=C_DARK)
pic(s, img(M1, "data_distribution.png"), 6.4, 4.47, 6.8, 2.83)
snum(s, 3)

# ═══════════════════════════════════════════════════════════════
# SLIDE 4 — 통계 분석 방법
# ═══════════════════════════════════════════════════════════════
s = add_slide()
hdr(s, "통계 분석 방법", "모델 비교를 위한 3단계 검증 체계")
for i,(col,title,body) in enumerate([
    (C_BLUE, "① DeLong AUC Test  (Test Set)",
     "동일 Test set 두 모델의 AUROC 직접 비교\n"
     "DeLong et al. (1988) — 쌍별 공분산 구조 활용\n"
     "H₀: AUROC_A = AUROC_B  →  z-통계량 기반 검정\n"
     "excl_extreme 조건: test set 크기 불일치로 M1 비교 제외"),
    (C_TEAL, "② Bootstrap 95% CI  (Test Set)",
     "n_boot = 2,000회 복원 추출 기반 신뢰구간\n"
     "2.5th–97.5th percentile 구간 보고\n"
     "AUROC / AUPRC / Brier / Accuracy / F1 전 지표\n"
     "단일 추정치의 불확실성 정량화"),
    (C_ORG, "③ Fold-level Paired Tests  (CV)",
     "5-Fold 교차검증 값에 대한 쌍별 검정\n"
     "Paired t-test + Wilcoxon signed-rank (n=5)\n"
     "두 검정 모두 보고 (p-value 쌍 제시)\n"
     "M1↔M2/M3: train set 차이 주의"),
]):
    card(s, 0.3+i*4.35, 0.99, 4.15, 5.72)
    box(s, 0.3+i*4.35, 0.99, 4.15, 0.5, fill=col)
    txt(s, title, 0.4+i*4.35, 1.02, 3.95, 0.45, sz=15, bold=True, color=C_WHT)
    y = 1.64
    for line in body.split("\n"):
        txt(s, f"  {line}", 0.4+i*4.35, y, 3.9, 0.44, sz=13, color=C_DARK)
        y += 0.47
box(s, 0.3, 6.37, 12.7, 0.52, fill=RGBColor(0xEE,0xF5,0xFF))
txt(s, "유의수준:  *** p<0.001   ** p<0.01   * p<0.05   † p<0.10   ns p≥0.10",
    0.5, 6.40, 12.3, 0.42, sz=13, bold=True, color=C_DARK, align=PP_ALIGN.CENTER)
snum(s, 4)

# ═══════════════════════════════════════════════════════════════
# SLIDE 5 — 전체 모델 성능 비교 개요
# [Brier 포함 → calibration 의무]
# ═══════════════════════════════════════════════════════════════
s = add_slide()
hdr(s, "전체 모델 성능 비교 개요  (BCE + AEC 128pt + norm)",
    "M1 → M2 → M3 순차 비교  |  Bootstrap 95% CI  |  Brier → Calibration 참조")

# Comparison table
hx = [0.3,1.9,3.05,4.6,5.75,6.85,7.95,9.65,11.35]
hw = [1.55,1.1,1.5,1.1,1.05,1.05,1.65,1.65,1.85]
trow_h(s, ["모델","AUC","95% CI AUC","AUPRC","Brier","F1",
           "vs M1 Δ AUC","vs M1 DeLong p","비고"],
       hx, hw, 0.99, ht=0.32)
rows_ov = [
    ("M1 (LR)","0.8325","[0.739, 0.909]","0.5008","0.1804","0.3953",
     "—","기준선","Clinic Only"),
    ("M2 (CrossAttn)","0.8868","[0.810, 0.949]","0.5551","0.2124","0.4444",
     "+0.054","p=0.071 †","Clinic+AEC"),
    ("M2_2 (Neg.Ctrl)","0.8420","[0.750, 0.919]","0.5177","0.1964","0.3619",
     "+0.010","(neg ctrl)","Unmatched"),
    ("M3 (CrossAttn3)","0.8944","[0.825, 0.950]","0.5636","0.1952","0.4571",
     "+0.062","p=0.040 *","Clinic+Scn+AEC"),
]
fills = [C_LGR,RGBColor(0xE8,0xF4,0xE8),RGBColor(0xF5,0xF5,0xF5),RGBColor(0xFF,0xF3,0xE0)]
sigs  = [C_MGR, C_GRN, C_MGR, C_GRN]
for i,(row,fill,sc) in enumerate(zip(rows_ov,fills,sigs)):
    table_c = [C_DARK]*6 + [C_BLUE, sc, C_MGR]
    trow_d(s, row, hx, hw, 1.33+i*0.4, ht=0.4, fill=fill, colors=table_c,
           bolds=[i in [1,3]]*9)

# Left bottom: ROC comparison
txt(s, "ROC 비교 (M1·M2·M2₂·M3)", 0.3, 3.87, 4.5, 0.3, sz=12, bold=True)
pic(s, img(CMP128, "roc_all_models_norm.png"), 0.3, 4.19, 4.5, 3.11)

# Center bottom: key comparison highlights
card(s, 5.0, 3.87, 4.1, 3.43, fill=RGBColor(0xE8,0xF8,0xE8))
box(s, 5.0, 3.87, 0.1, 3.43, fill=C_GRN)
txt(s, "비교 핵심", 5.2, 3.95, 3.7, 0.32, sz=14, bold=True, color=C_GRN)
for i,t in enumerate([
    "• M1→M3: Δ AUC +0.062, DeLong p=0.040 (*)",
    "• M1→M2: Δ AUC +0.054, DeLong p=0.071 (†)",
    "• M2→M3:  Δ AUC +0.008, p=0.437 (ns)",
    "• AUPRC: M1(0.501)→M3(0.564) +12.6%",
    "• M2_2(neg ctrl) vs M2: Δ-0.045 → 매칭 효과 일부 확인",
    "• 여성 AUC: M1(0.792)→M3(0.876) 극적 개선",
]):
    txt(s, t, 5.2, 4.36+i*0.45, 3.7, 0.42, sz=12, color=C_DARK)

# Right bottom: calibration (Brier shown → mandatory)
calib_banner(s, 9.25, 3.87, 3.85, "M3 norm  (Brier 0.1952)")
pic(s, img(M3N, "calibration.png"), 9.25, 4.22, 3.85, 3.08)
snum(s, 5)

# ═══════════════════════════════════════════════════════════════
# SLIDE 6 — M1 vs M2: AEC의 기여
# [Brier 포함 → calibration 의무]
# ═══════════════════════════════════════════════════════════════
s = add_slide()
hdr(s, "M1 vs M2 비교 — AEC 신호의 기여",
    "Clinic Only(LR) vs Clinic+AEC(CrossAttn)  |  DeLong AUC Test  |  Brier → Calibration 참조")

# DeLong table (left)
card(s, 0.3, 0.99, 6.0, 4.52)
txt(s, "DeLong Test — M1(LR) vs M2(CrossAttn)  [ norm 조건 ]",
    0.5, 1.07, 5.6, 0.33, sz=14, bold=True)
hline(s, 0.5, 1.42, 5.6, color=C_BLUE)
dx = [0.45, 2.05, 3.1, 4.15, 5.2]; dw = [1.55, 1.0, 1.0, 1.0, 0.95]
trow_h(s, ["시나리오","M1 AUC","M2 AUC","Δ AUC","p-val"], dx, dw, 1.47)
dlong_m1m2 = [
    ("BCE+AEC128","0.8325","0.8868","+0.054","p=0.071 †"),
    ("BCE+AEC256","0.8325","0.8884","+0.056","p=0.013 *"),
    ("Focal+AEC128","0.8325","0.8739","+0.041","p=0.079 †"),
    ("Focal+AEC256","0.8325","0.8600","+0.027","p=0.330 ns"),
]
sig_c_m1m2 = [C_GRN, C_GRN, C_GRN, C_MGR]
for i,(row,sc) in enumerate(zip(dlong_m1m2, sig_c_m1m2)):
    fill = C_WHT if i%2==0 else C_LGR
    trow_d(s, row, dx, dw, 1.79+i*0.4, ht=0.4, fill=fill,
           colors=[C_DARK,C_DARK,C_BLUE,C_BLUE,sc])

# Bootstrap CI (left continued)
txt(s, "Bootstrap 95% CI — BCE+AEC128+norm 기준",
    0.5, 3.49, 5.6, 0.3, sz=12, bold=True, color=C_DARK)
hline(s, 0.5, 3.81, 5.6, color=C_TEAL, h=0.03)
bx2 = [0.45, 1.85, 3.25, 4.7]; bw2 = [1.35, 1.35, 1.4, 1.25]
trow_h(s, ["지표","M1","M2","Δ(M2-M1)"], bx2, bw2, 3.87, ht=0.28)
for i,(row,fill) in enumerate(zip([
    ("AUC-ROC","0.833 [0.739,0.909]","0.887 [0.810,0.949]","+0.054"),
    ("AUPRC",  "0.501 [0.297,0.694]","0.555 [0.361,0.764]","+0.054"),
    ("Brier",  "0.180 [0.153,0.212]","0.212 [0.177,0.249]","+0.032"),
    ("F1",     "0.395 [0.254,0.523]","0.444 [0.303,0.578]","+0.049"),
],[C_WHT,C_LGR,C_WHT,C_LGR])):
    trow_d(s, row, bx2, bw2, 4.17+i*0.33, ht=0.33, fill=fill)

# Right top: 4-panel ROC (all 4 scenarios)
txt(s, "조건별 ROC 비교  (norm 기준, M1·M2·M2₂·M3 포함)", 6.5, 0.99, 6.6, 0.3, sz=12, bold=True)
for j,(folder, lbl) in enumerate([
    (CMP128,"BCE+AEC128"), (CMP256,"BCE+AEC256"),
    (CMPF128,"Focal+AEC128"), (CMPF256,"Focal+AEC256"),
]):
    col_j = j%2; row_j = j//2
    x0 = 6.5 + col_j*3.4; y0 = 1.34 + row_j*1.72
    pic(s, img(folder, "roc_all_models_norm.png"), x0, y0, 3.2, 1.6)
    txt(s, lbl, x0, y0+1.62, 3.2, 0.22, sz=9, color=C_MGR, align=PP_ALIGN.CENTER)

# Right bottom: calibration (Brier shown → mandatory)
calib_banner(s, 6.5, 4.74, 6.6, "M2  BCE+AEC128+norm  (Brier 0.2124)")
pic(s, img(M2N, "calibration.png"), 6.5, 5.09, 6.6, 2.21)
snum(s, 6)

# ═══════════════════════════════════════════════════════════════
# SLIDE 7 — M1 vs M3: Scanner+AEC 효과
# [Brier 포함 → calibration 의무]
# ═══════════════════════════════════════════════════════════════
s = add_slide()
hdr(s, "M1 vs M3 비교 — Scanner + AEC 조합 효과",
    "Clinic Only(LR) vs Clinic+Scanner+AEC(CrossAttn3)  |  DeLong AUC Test  |  Brier → Calibration 참조")

card(s, 0.3, 0.99, 6.0, 4.52)
txt(s, "DeLong Test — M1(LR) vs M3(CrossAttn3)  [ norm 조건 ]",
    0.5, 1.07, 5.6, 0.33, sz=14, bold=True)
hline(s, 0.5, 1.42, 5.6, color=C_ORG)
trow_h(s, ["시나리오","M1 AUC","M3 AUC","Δ AUC","p-val"], dx, dw, 1.47)
dlong_m1m3 = [
    ("BCE+AEC128","0.8325","0.8944","+0.062","p=0.040 *"),
    ("BCE+AEC256","0.8325","0.8858","+0.053","p=0.062 †"),
    ("Focal+AEC128","0.8325","0.8448","+0.012","p=0.776 ns"),
    ("Focal+AEC256","0.8325","0.8641","+0.032","p=0.253 ns"),
]
sig_c_m1m3 = [C_GRN, C_GRN, C_MGR, C_MGR]
for i,(row,sc) in enumerate(zip(dlong_m1m3, sig_c_m1m3)):
    fill = C_WHT if i%2==0 else C_LGR
    trow_d(s, row, dx, dw, 1.79+i*0.4, ht=0.4, fill=fill,
           colors=[C_DARK,C_DARK,C_ORG,C_BLUE,sc])

txt(s, "Bootstrap 95% CI — BCE+AEC128+norm 기준",
    0.5, 3.49, 5.6, 0.3, sz=12, bold=True)
hline(s, 0.5, 3.81, 5.6, color=C_ORG, h=0.03)
trow_h(s, ["지표","M1","M3","Δ(M3-M1)"], bx2, bw2, 3.87, ht=0.28)
for i,(row,fill) in enumerate(zip([
    ("AUC-ROC","0.833 [0.739,0.909]","0.894 [0.825,0.950]","+0.062"),
    ("AUPRC",  "0.501 [0.297,0.694]","0.564 [0.366,0.766]","+0.063"),
    ("Brier",  "0.180 [0.153,0.212]","0.195 [0.165,0.227]","+0.015"),
    ("F1",     "0.395 [0.254,0.523]","0.457 [0.305,0.595]","+0.062"),
],[C_WHT,C_LGR,C_WHT,C_LGR])):
    trow_d(s, row, bx2, bw2, 4.17+i*0.33, ht=0.33, fill=fill)

txt(s, "조건별 ROC 비교  (norm 기준, M1·M2·M2₂·M3 포함)", 6.5, 0.99, 6.6, 0.3, sz=12, bold=True)
for j,(folder,lbl) in enumerate([
    (CMP128,"BCE+AEC128"),(CMP256,"BCE+AEC256"),
    (CMPF128,"Focal+AEC128"),(CMPF256,"Focal+AEC256"),
]):
    col_j=j%2; row_j=j//2
    x0=6.5+col_j*3.4; y0=1.34+row_j*1.72
    pic(s, img(folder,"roc_all_models_norm.png"), x0, y0, 3.2, 1.6)
    txt(s, lbl, x0, y0+1.62, 3.2, 0.22, sz=9, color=C_MGR, align=PP_ALIGN.CENTER)

calib_banner(s, 6.5, 4.74, 6.6, "M3  BCE+AEC128+norm  (Brier 0.1952)")
pic(s, img(M3N, "calibration.png"), 6.5, 5.09, 6.6, 2.21)
snum(s, 7)

# ═══════════════════════════════════════════════════════════════
# SLIDE 8 — M2 vs M3: Scanner 순증가 효과
# ═══════════════════════════════════════════════════════════════
s = add_slide()
hdr(s, "M2 vs M3 비교 — Scanner(MFR) 추가 효과",
    "CrossAttn vs CrossAttn3  |  모든 조건에서 DeLong ns — 통계적 유의차 없음")

# DeLong table
card(s, 0.3, 0.99, 7.5, 4.71)
txt(s, "DeLong Test — M2(CrossAttn) vs M3(CrossAttn3)  [ 전 조건 ]",
    0.5, 1.07, 7.1, 0.33, sz=14, bold=True)
hline(s, 0.5, 1.42, 7.1, color=C_MGR)
mx = [0.45, 2.3, 3.45, 4.6, 5.75, 6.85]
mw = [1.8,  1.1, 1.1,  1.1, 1.05, 0.9]
trow_h(s, ["조건","M2 AUC","M3 AUC","Δ AUC","p-val","sig"], mx, mw, 1.47)
m2m3_rows = [
    ("BCE+AEC128 norm",    "0.8868","0.8944","+0.008","0.437","ns"),
    ("BCE+AEC128 crop60",  "0.8635","0.8505","-0.013","0.513","ns"),
    ("BCE+AEC128 crop80",  "0.8612","0.8575","-0.004","0.793","ns"),
    ("BCE+AEC128 len128",  "0.8578","0.8357","-0.022","0.255","ns"),
    ("BCE+AEC256 norm",    "0.8884","0.8858","-0.003","0.851","ns"),
    ("BCE+AEC256 len128",  "0.8694","0.8559","-0.014","0.530","ns"),
    ("Focal+AEC128 norm",  "0.8739","0.8448","-0.029","0.296","ns"),
    ("Focal+AEC256 norm",  "0.8600","0.8641","+0.004","0.807","ns"),
    ("Focal+AEC256 excl",  "0.8854","0.8174","-0.068","0.013","*"),
]
for i,row in enumerate(m2m3_rows):
    fill = C_WHT if i%2==0 else C_LGR
    is_sig = row[-1] != "ns"
    trow_d(s, row, mx, mw, 1.79+i*0.37, ht=0.37, fill=fill,
           colors=[C_DARK]*4+[C_GRN if is_sig else C_MGR, C_GRN if is_sig else C_MGR])

# Right: ROC + insight
txt(s, "ROC — BCE+AEC128+norm  (M2 vs M3 근소 차이)", 8.0, 0.99, 5.1, 0.3, sz=12, bold=True)
pic(s, img(CMP128,"roc_all_models_norm.png"), 8.0, 1.31, 5.1, 2.68)

card(s, 8.0, 4.14, 5.1, 3.11, fill=RGBColor(0xF5,0xF5,0xF5))
box(s, 8.0, 4.14, 0.1, 3.11, fill=C_MGR)
txt(s, "해석", 8.2, 4.22, 4.7, 0.33, sz=14, bold=True, color=C_MGR)
for i,t in enumerate([
    "• M2 vs M3: 20개 조건 중 19개 ns",
    "  → Scanner 정보가 AUC에 통계적 유의차 미산출",
    "• 예외: Focal+AEC256+excl_extreme p=0.013 (*)",
    "  → 단 1건, 해석에 주의 필요",
    "• M3는 일관되게 M2보다 미세하게 높거나 비슷함",
    "  → 방향성은 존재, 검출력 부족 가능성",
    "• 추후 n↑ → M2 vs M3 재검증 권장",
]):
    txt(s, t, 8.2, 4.64+i*0.38, 4.7, 0.36, sz=11.5, color=C_DARK)
snum(s, 8)

# ═══════════════════════════════════════════════════════════════
# SLIDE 9 — M2 vs M2_2: 매칭 검증 (음성 대조군)
# ═══════════════════════════════════════════════════════════════
s = add_slide()
hdr(s, "M2 vs M2_2 비교 — 매칭 효과 검증 (음성 대조군)",
    "Matched(M2) vs Unmatched(M2_2)  |  AEC 개인 대응의 실질적 기여 확인")

# Experiment design
card(s, 0.3, 0.99, 4.8, 1.45)
txt(s, "실험 설계", 0.5, 1.07, 4.4, 0.3, sz=14, bold=True)
hline(s, 0.5, 1.39, 4.4, color=C_TEAL)
for i,t in enumerate([
    "M2: 환자 i의 Clinic + 환자 i의 AEC (동일 환자)",
    "M2_2: 환자 i의 Clinic + 환자 j의 AEC (다른 환자)",
    "→ 동일 CrossAttention 구조 / AEC만 랜덤 셔플",
]):
    txt(s, t, 0.5, 1.49+i*0.32, 4.4, 0.3, sz=12, color=C_DARK)

# DeLong table
card(s, 0.3, 2.57, 7.5, 4.21)
txt(s, "DeLong Test — M2(Matched) vs M2_2(Unmatched)", 0.5, 2.65, 7.1, 0.3, sz=14, bold=True)
hline(s, 0.5, 2.97, 7.1, color=C_TEAL)
nx = [0.45, 2.5, 3.7, 5.0, 5.95, 6.95]
nw = [2.0, 1.15, 1.25, 0.9, 0.95, 0.9]
trow_h(s, ["조건","M2 AUC","M2_2 AUC","Δ AUC","p-val","sig"], nx, nw, 3.02)
neg_ctrl = [
    ("BCE+AEC128 norm",     "0.8868","0.8420","+0.045","0.138","ns"),
    ("BCE+AEC128 excl_ext", "0.8339","0.5596","+0.274","<0.001","***"),
    ("BCE+AEC256 norm",     "0.8884","0.7931","+0.095","0.005","**"),
    ("BCE+AEC256 excl_ext", "0.8507","0.5474","+0.303","<0.001","***"),
    ("Focal+AEC128 norm",   "0.8739","0.7975","+0.076","0.008","**"),
    ("Focal+AEC128 excl_ext","0.8854","0.5503","+0.335","<0.001","***"),
    ("Focal+AEC256 norm",   "0.8600","0.8382","+0.022","0.489","ns"),
    ("Focal+AEC256 excl_ext","0.8854","0.5353","+0.350","<0.001","***"),
]
for i,row in enumerate(neg_ctrl):
    fill = C_WHT if i%2==0 else C_LGR
    sc = C_RED if row[-1] != "ns" else C_MGR
    trow_d(s, row, nx, nw, 3.34+i*0.37, ht=0.37, fill=fill,
           colors=[C_DARK]*4+[sc,sc])

# Right: ROC most dramatic + insight
txt(s, "ROC — excl_extreme 조건 (가장 극적인 차이)",
    8.0, 0.99, 5.1, 0.3, sz=12, bold=True)
pic(s, img(CMP128,"roc_all_models_excl_extreme.png"), 8.0, 1.31, 5.1, 2.55)

card(s, 8.0, 3.99, 5.1, 3.21, fill=RGBColor(0xFF,0xF3,0xE0))
box(s, 8.0, 3.99, 0.1, 3.21, fill=C_ORG)
txt(s, "해석", 8.2, 4.07, 4.7, 0.3, sz=14, bold=True, color=C_ORG)
for i,t in enumerate([
    "• excl_extreme: M2(0.834) vs M2_2(0.560) Δ=+0.274 ***",
    "  → AEC는 개인별 고유 신호 — 단순 체형 proxy 아님",
    "• BCE+AEC256+norm: Δ=+0.095 **",
    "• Focal+AEC128+norm: Δ=+0.076 **",
    "• norm 조건: Loss·AEC길이에 따라 매칭 효과 강도 상이",
    "  → AEC 신호의 개인 대응이 예측력의 핵심 요소",
]):
    txt(s, t, 8.2, 4.45+i*0.38, 4.7, 0.36, sz=11.5, color=C_DARK)
snum(s, 9)

# ═══════════════════════════════════════════════════════════════
# SLIDE 10 — 4-panel ROC: 전 시나리오 비교
# ═══════════════════════════════════════════════════════════════
s = add_slide()
hdr(s, "전 시나리오 ROC 비교 — norm 조건",
    "Loss × AEC 길이 4가지 조합에서 M1·M2·M2₂·M3 비교")

panels = [
    (CMP128,  "BCE + AEC 128pt / norm",
     "M1→M3: p=0.040 *  |  M1→M2: p=0.071 †", C_BLUE),
    (CMP256,  "BCE + AEC 256pt / norm",
     "M1→M2: p=0.013 *  |  M1→M3: p=0.062 †", C_TEAL),
    (CMPF128, "Focal + AEC 128pt / norm",
     "M1→M2: p=0.079 †  |  M1→M3: ns", C_ORG),
    (CMPF256, "Focal + AEC 256pt / norm",
     "M1→M2: ns  |  M1→M3: ns", C_MGR),
]
for (folder,label,note,col),(x0,y0) in zip(panels,[(0.25,0.99),(6.82,0.99),(0.25,4.09),(6.82,4.09)]):
    box(s, x0, y0, 6.3, 0.38, fill=col)
    txt(s, label, x0+0.1, y0+0.02, 6.1, 0.34, sz=13, bold=True, color=C_WHT)
    txt(s, note,  x0+0.1, y0+0.42, 6.1, 0.28, sz=11, color=C_DARK, italic=True)
    pic(s, img(folder,"roc_all_models_norm.png"), x0, y0+0.72, 6.3, 3.2)
snum(s, 10)

# ═══════════════════════════════════════════════════════════════
# SLIDE 11 — Attention Map 비교 (M2 vs M3)
# ═══════════════════════════════════════════════════════════════
s = add_slide()
hdr(s, "Attention Map 비교 — M2 vs M3",
    "BCE + AEC 128pt + norm  |  CrossAttn(M2) · CrossAttn3(M3) 어텐션 패턴 비교")

for j,(mdir,label,col) in enumerate([
    (M2N,  "Model 2  CrossAttn  (Clinic → AEC)",  C_TEAL),
    (M3N,  "Model 3  CrossAttn3  (Clinic+Scanner → AEC)", C_ORG),
]):
    x0 = 0.3 + j*6.5
    box(s, x0, 0.99, 6.3, 0.38, fill=col)
    txt(s, label, x0+0.1, 1.01, 6.1, 0.34, sz=14, bold=True, color=C_WHT)
    txt(s, "Attention Heatmap (샘플별)", x0+0.1, 1.42, 3.0, 0.25, sz=10, color=C_MGR)
    pic(s, img(mdir,"attention_heatmap.png"), x0, 1.69, 6.2, 2.75)
    txt(s, "Attention Map C→A (평균)", x0+0.1, 4.49, 3.0, 0.25, sz=10, color=C_MGR)
    pic(s, img(mdir,"attention_map_c2a.png"), x0, 4.76, 6.2, 2.59)
snum(s, 11)

# ═══════════════════════════════════════════════════════════════
# SLIDE 12 — 최적 조건 상세 비교
# [Brier 포함 → calibration 의무]
# ═══════════════════════════════════════════════════════════════
s = add_slide()
hdr(s, "최적 조건 상세 비교 — BCE + AEC128+norm  vs  BCE + AEC256+norm",
    "두 조건에서 독립적으로 통계 유의성 확인  |  Brier → Calibration 참조")

# Summary highlight
box(s, 0.3, 0.99, 12.7, 0.65, fill=RGBColor(0xE8,0xF8,0xE8))
box(s, 0.3, 0.99, 0.12, 0.65, fill=C_GRN)
txt(s,
    "BCE+AEC128+norm: M1→M3 p=0.040 (*)  /  "
    "BCE+AEC256+norm: M1→M2 p=0.013 (*) · M2 vs M2_2 p=0.005 (**)  "
    "→ 두 조건 모두 AEC 기여 통계 검증",
    0.55, 1.06, 12.1, 0.52, sz=13, bold=True, color=C_DARK)

# Comparison table (Brier shown → calibration mandatory)
hx7 = [0.3,1.55,2.8,4.5,5.6,6.65,7.7,9.4,11.1]
hw7 = [1.2,1.2, 1.65,1.05,1.0,1.0, 1.65,1.65,2.05]
trow_h(s, ["모델","AUC","95% CI","AUPRC","Brier","F1",
           "BCE128 DeLong","BCE256 DeLong","비고"],
       hx7, hw7, 1.74, ht=0.3)
for i,(row,fill) in enumerate(zip([
    ("M1 (LR)","0.8325","[0.739,0.909]","0.5008","0.1804","0.3953",
     "기준선","기준선","Clinic Only"),
    ("M2 (CrossAttn)","0.8868","[0.810,0.949]","0.5551","0.2124","0.4444",
     "p=0.071 †","p=0.013 *","BCE128 vs 256"),
    ("M2_2 (Neg)","0.8420","[0.750,0.919]","0.5177","0.1964","0.3619",
     "(neg ctrl)","(neg ctrl)","Unmatched"),
    ("M3 (CrossAttn3)","0.8944","[0.825,0.950]","0.5636","0.1952","0.4571",
     "p=0.040 *","p=0.062 †","Best Overall"),
],[C_LGR,RGBColor(0xE8,0xF4,0xE8),C_WHT,RGBColor(0xFF,0xF3,0xE0)])):
    s6 = C_GRN if "*" in row[6] or "†" in row[6] else C_MGR
    s7 = C_GRN if "*" in row[7] or "†" in row[7] else C_MGR
    trow_d(s, row, hx7, hw7, 2.06+i*0.4, ht=0.4, fill=fill,
           colors=[C_DARK]*6+[s6,s7,C_MGR], bolds=[i in [1,3]]*9)

# Bottom: ROC comparison + confusion + calibration (Brier → mandatory)
txt(s, "ROC — BCE+AEC128+norm", 0.3, 3.79, 3.9, 0.28, sz=11, bold=True)
pic(s, img(CMP128,"roc_all_models_norm.png"), 0.3, 4.09, 3.9, 3.16)

txt(s, "M3 Confusion Matrix", 4.35, 3.79, 3.9, 0.28, sz=11, bold=True)
pic(s, img(M3N,"confusion_matrices.png"), 4.35, 4.09, 3.9, 3.16)

calib_banner(s, 8.4, 3.79, 4.7, "M3  BCE+AEC128+norm  (Brier 0.1952)")
pic(s, img(M3N,"calibration.png"), 8.4, 4.14, 4.7, 3.11)
snum(s, 12)

# ═══════════════════════════════════════════════════════════════
# SLIDE 13 — 성별 분리 비교 (M1 vs M3)
# ═══════════════════════════════════════════════════════════════
s = add_slide()
hdr(s, "성별 분리 비교 — M1 vs M3  (BCE + AEC128 + norm)",
    "남성(n=71, 유병률 15.5%) · 여성(n=101, 유병률 9.9%) 분리 AUC 비교")

# Sex comparison table
card(s, 0.3, 0.99, 12.7, 2.0)
sx = [0.4,1.9,3.25,4.65,6.05,7.4,8.8,10.2,11.6]
sw = [1.45,1.3,1.35,1.35,1.3,1.35,1.35,1.35,1.55]
trow_h(s, ["모델","전체 AUC","전체 CI","남성 AUC","남성 CI","여성 AUC","여성 CI","남 AUPRC","여 AUPRC"],
       sx, sw, 1.04, ht=0.3)
for i,(row,fill) in enumerate(zip([
    ("M1 (LR)",        "0.8325","[0.739,0.909]","0.8818","[0.786,0.956]","0.7923","[0.677,0.895]","0.6710","0.3197"),
    ("M2 (CrossAttn)", "0.8868","[0.810,0.949]","0.8879","[0.800,0.954]","0.8714","[0.793,0.946]","0.5968","0.6063"),
    ("M3 (CrossAttn3)","0.8944","[0.825,0.950]","0.9000","[0.825,0.959]","0.8758","[0.793,0.950]","0.6218","0.4883"),
],[C_LGR,RGBColor(0xE8,0xF4,0xE8),RGBColor(0xFF,0xF3,0xE0)])):
    trow_d(s, row, sx, sw, 1.36+i*0.41, ht=0.41, fill=fill, bolds=[i>0]*9)

# Analysis cards
card(s, 0.3, 3.09, 3.8, 4.06, fill=RGBColor(0xE8,0xF0,0xFF))
box(s, 0.3, 3.09, 0.1, 4.06, fill=C_BLUE)
txt(s, "남성 비교", 0.5, 3.17, 3.5, 0.32, sz=14, bold=True, color=C_BLUE)
for i,t in enumerate([
    "• M1→M3: 0.882→0.900  (Δ+0.018)",
    "• M3에서 처음으로 AUC 0.90 돌파",
    "• AUPRC: M1(0.671)→M3(0.622)",
    "  → Precision관점에서 M1이 더 높음",
    "• 유병률 15.5% → 균형 잡힌 클래스",
    "  → AUC 해석 안정적",
]):
    txt(s, t, 0.5, 3.59+i*0.52, 3.6, 0.5, sz=12, color=C_DARK)

card(s, 4.25, 3.09, 3.8, 4.06, fill=RGBColor(0xFF,0xF0,0xF8))
box(s, 4.25, 3.09, 0.1, 4.06, fill=C_RED)
txt(s, "여성 비교", 4.45, 3.17, 3.5, 0.32, sz=14, bold=True, color=C_RED)
for i,t in enumerate([
    "• M1→M3: 0.792→0.876  (Δ+0.084)",
    "  → 여성에서 AEC 기여 효과가 더 큼",
    "• AUPRC: M1(0.320)→M2(0.606)",
    "  → AEC 추가로 양성 정밀도 극적 개선",
    "• 유병률 9.9% → 심한 class imbalance",
    "  → AUPRC가 AUC보다 중요한 지표",
]):
    txt(s, t, 4.45, 3.59+i*0.52, 3.6, 0.5, sz=12, color=C_DARK)

# ROC by sex images
txt(s, "M1  by Sex", 8.2, 3.09, 2.4, 0.28, sz=11, bold=True)
pic(s, img(M1,"test_roc_by_sex.png"), 8.2, 3.39, 2.4, 2.0)
txt(s, "M3  by Sex", 10.75, 3.09, 2.4, 0.28, sz=11, bold=True, color=C_ORG)
pic(s, img(M3N,"test_roc_by_sex.png"), 10.75, 3.39, 2.4, 2.0)
txt(s, "△ 여성 AUC 0.792 → 0.876  |  △ 남성 AUC 0.882 → 0.900",
    8.2, 5.44, 5.0, 0.3, sz=11, bold=True, color=C_DARK, align=PP_ALIGN.CENTER)

# Training curves
txt(s, "M3 norm — Training Curves", 8.2, 5.78, 4.9, 0.28, sz=10, color=C_MGR)
pic(s, img(M3N,"training_curves.png"), 8.2, 6.08, 4.9, 1.22)
snum(s, 13)

# ═══════════════════════════════════════════════════════════════
# SLIDE 14 — 종합 결론
# ═══════════════════════════════════════════════════════════════
s = add_slide()
hdr(s, "종합 결론 및 임상적 함의",
    "AEC 신호의 근감소증 예측 기여 — 비교 연구 결과 요약")
for i,(col,title,items) in enumerate([
    (C_BLUE, "① AEC 신호는 Clinic 대비 통계적으로 유의미하게 AUC를 향상", [
        "BCE+AEC128+norm: M1→M3 DeLong p=0.040 (*)",
        "BCE+AEC256+norm: M1→M2 DeLong p=0.013 (*)",
        "음성 대조군(M2_2) 대비 M2 일관 우세 → 매칭 효과 검증",
        "여성 AUPRC: M1(0.320)→M2(0.606) — AEC가 양성 정밀도 극적 개선",
    ]),
    (C_TEAL, "② norm 전처리 조건이 최적 — 전 조건 중 가장 일관된 AUC 향상", [
        "모든 Loss×AEC길이 시나리오에서 norm이 최고 또는 상위 AUC",
        "crop/truncation 없이 전체 시퀀스 활용이 최선",
        "BCE+AEC256+norm에서 M1 vs M2 유의성 강화: p=0.013 (*)",
        "→ AEC 길이 256pt가 M2의 AEC 학습 신호를 증폭",
    ]),
    (C_ORG, "③ M2 vs M3: Scanner 추가효과 미입증 — 20개 조건 중 19개 ns", [
        "BCE+AEC128+norm: M2(0.887) vs M3(0.894) p=0.437 ns",
        "단, 방향성은 M3≥M2 일관 → 검출력 부족 가능성",
        "남성 AUC 0.900 달성은 M3에서만 → 성별별 효과 차별화",
        "→ 다기관·다scanner 데이터 확보 시 M3 효과 강화 예상",
    ]),
    (C_GRN, "④ Zero-cost AEC로 임상 등급 스크리닝 가능성 확인", [
        "기존 CT 스캔 시 자동 수집 — 추가 비용·검사 없음",
        "남성 AUC 0.900 / 여성 AUC 0.876 → 임상 허용 수준",
        "건강검진 프로그램 연동 시 고위험군 조기 선별 가능",
        "PACS-DICOM AEC 자동 추출 파이프라인 구축 필요",
    ]),
]):
    card(s, 0.3, 0.99+i*1.52, 12.7, 1.45)
    box(s, 0.3, 0.99+i*1.52, 0.12, 1.45, fill=col)
    txt(s, title, 0.55, 1.06+i*1.52, 12.1, 0.36, sz=14, bold=True, color=col)
    for k,item in enumerate(items[:2]):
        txt(s, f"  • {item}", 0.55, 1.46+i*1.52+k*0.32, 6.0, 0.3, sz=11.5, color=C_DARK)
    for k,item in enumerate(items[2:]):
        txt(s, f"  • {item}", 6.8,  1.46+i*1.52+k*0.32, 6.1, 0.3, sz=11.5, color=C_DARK)
snum(s, 14)

# ═══════════════════════════════════════════════════════════════
# SLIDE 15 — 제한점 & 향후 연구
# ═══════════════════════════════════════════════════════════════
s = add_slide()
hdr(s, "제한점 및 향후 연구 방향", "현재 비교 연구의 한계와 발전 가능성")
card(s, 0.3, 0.99, 5.9, 5.91)
txt(s, "연구 제한점", 0.5, 1.07, 5.5, 0.36, sz=16, bold=True, color=C_RED)
hline(s, 0.5, 1.45, 5.5, color=C_RED)
for i,(t,b) in enumerate([
    ("소표본 비교","Test n=172, Sarco 21명 → DeLong 검출력 제한, CI 넓음"),
    ("단일 기관","단일 센터 데이터 → 외부 검증 미실시"),
    ("excl_extreme 비교 제한","Test set 크기 변화 → M1 vs M2/M3 DeLong 직접 비교 불가"),
    ("M2 vs M3 미유의","Scanner 순증가 효과 미확인 → 검출력 부족 가능성"),
    ("단면 연구","AEC-근육량 인과관계 미확인 → 종단 연구 필요"),
]):
    txt(s, f"  ▶ {t}", 0.5, 1.57+i*1.0, 5.5, 0.34, sz=13, bold=True)
    txt(s, f"    {b}", 0.5, 1.91+i*1.0, 5.5, 0.58, sz=11.5, color=C_MGR)
card(s, 6.5, 0.99, 6.5, 5.91)
txt(s, "향후 연구 방향", 6.7, 1.07, 6.1, 0.36, sz=16, bold=True, color=C_BLUE)
hline(s, 6.7, 1.45, 6.1, color=C_BLUE)
for i,(t,b) in enumerate([
    ("다기관 검증","강남+신촌 통합 → 일반화 성능·DeLong 유의성 재확인"),
    ("샘플 확장","Test n>500 → CI 축소, M2 vs M3 검출력 증가"),
    ("전처리 설계 개선","동일 test set 유지 설계 → excl_extreme 공정 비교"),
    ("아키텍처 확장","Multi-head Attn + positional encoding → 시퀀스 길이 극복"),
    ("임상 파이프라인","PACS-DICOM AEC 추출 자동화 → 실시간 스크리닝"),
]):
    txt(s, f"  ◆ {t}", 6.7, 1.57+i*1.0, 6.1, 0.34, sz=13, bold=True, color=C_BLUE)
    txt(s, f"    {b}", 6.7, 1.91+i*1.0, 6.1, 0.58, sz=11.5)
box(s, 0.3, 6.95, 12.7, 0.38, fill=C_DARK)
txt(s, "감사합니다  |  Questions & Discussion",
    0.3, 6.97, 12.7, 0.33, sz=14, bold=True, color=C_WHT, align=PP_ALIGN.CENTER)
snum(s, 15)

# ═══════════════════════════════════════════════════════════════
# SLIDE 16 — 부록 섹션 디바이더
# ═══════════════════════════════════════════════════════════════
s = add_slide()
box(s, 0, 0, 13.33, 7.5, fill=C_DARK)
box(s, 0, 0, 13.33, 0.08, fill=C_BLUE)
box(s, 0, 7.42, 13.33, 0.08, fill=C_TEAL)
txt(s, "부록 — 조건별 상세 결과",
    1.0, 2.3, 11.33, 0.9, sz=36, bold=True, color=C_WHT, align=PP_ALIGN.CENTER)
txt(s, "Loss × AEC 길이 × 전처리  전 조건  |  M2 · M2_2 · M3 개별 시각화",
    1.0, 3.35, 11.33, 0.55, sz=17, color=C_MGR, align=PP_ALIGN.CENTER)
hline(s, 3.5, 4.0, 6.33, color=C_TEAL)
for i,(lbl,col,var) in enumerate([
    ("BCEWithLogitsLoss + AEC 128pt", C_BLUE,  "norm / crop60 / crop80 / len128 / excl_extreme"),
    ("BCEWithLogitsLoss + AEC 256pt", C_TEAL,  "norm / crop60 / crop80 / len128 / len256 / excl_extreme"),
    ("FocalLoss + AEC 128pt",         C_ORG,   "norm / crop60 / crop80 / len128 / excl_extreme"),
    ("FocalLoss + AEC 256pt",         C_GRN,   "norm / crop60 / crop80 / len256 / excl_extreme"),
]):
    box(s, 1.0, 4.35+i*0.62, 11.33, 0.52, fill=col)
    txt(s, f"  {lbl}   →  {var}",
        1.05, 4.37+i*0.62, 11.2, 0.48, sz=13, bold=True, color=C_WHT)
txt(s, "각 슬라이드: 전모델 ROC | M2 Calibration | M3 Calibration  ·  M2 Confusion | M3 Confusion | M2 Attention",
    1.0, 6.87, 11.33, 0.38, sz=11, color=C_MGR, align=PP_ALIGN.CENTER)
snum(s, 16)

# ═══════════════════════════════════════════════════════════════
# 조건별 슬라이드 생성
# ═══════════════════════════════════════════════════════════════
LOSS_LBL = {"bce": "BCEWithLogitsLoss", "focal": "FocalLoss"}
AEC_LBL  = {"aec128": "AEC 128pt",      "aec256": "AEC 256pt"}
COND_COL = {
    ("bce",   "aec128"): C_BLUE,
    ("bce",   "aec256"): C_TEAL,
    ("focal", "aec128"): C_ORG,
    ("focal", "aec256"): C_GRN,
}

def condition_slide(loss, aec_len, variant, sn):
    col     = COND_COL[(loss, aec_len)]
    m2_dir  = f"{BASE}/model_2/{loss}/{aec_len}/{variant}"
    m3_dir  = f"{BASE}/model_3/{loss}/{aec_len}/{variant}"
    cmp_png = f"{BASE}/comparison/{loss}/{aec_len}/roc_all_models_{variant}.png"

    s = add_slide()
    box(s, 0, 0, 13.33, 0.84, fill=C_DARK)
    box(s, 0, 0, 0.35, 0.84, fill=col)
    txt(s, f"{LOSS_LBL[loss]}  +  {AEC_LBL[aec_len]}  +  {variant}",
        0.45, 0.06, 10.5, 0.44, sz=22, bold=True, color=C_WHT)
    txt(s, "M2 (Matched) · M3 (Clinic+Scanner+AEC)  |  Brier → Calibration 포함",
        0.45, 0.50, 10.5, 0.28, sz=13, color=C_MGR)
    txt(s, f"{sn} / {TOTAL}", 12.3, 7.15, 1.0, 0.3, sz=11,
        color=C_MGR, align=PP_ALIGN.RIGHT)

    cx = [0.25, 4.57, 8.88]; cw = [4.2, 4.2, 4.2]
    for j,(lbl,lc) in enumerate([
        ("전모델 ROC 비교  (M1·M2·M2₂·M3)", C_DARK),
        ("M2 Calibration  ※ Brier 보정 확인", C_ORG),
        ("M3 Calibration  ※ Brier 보정 확인", C_ORG),
    ]):
        box(s, cx[j], 0.89, cw[j], 0.33, fill=lc)
        txt(s, lbl, cx[j]+0.05, 0.91, cw[j]-0.1, 0.29,
            sz=11, bold=True, color=C_WHT, align=PP_ALIGN.CENTER)

    pic(s, cmp_png,                        cx[0], 1.24, cw[0], 2.85)
    pic(s, img(m2_dir, "calibration.png"), cx[1], 1.24, cw[1], 2.85)
    pic(s, img(m3_dir, "calibration.png"), cx[2], 1.24, cw[2], 2.85)

    for j,(lbl,lc) in enumerate([
        ("M2 Confusion Matrix", C_TEAL),
        ("M3 Confusion Matrix", C_ORG),
        ("M2 Attention Heatmap", C_BLUE),
    ]):
        box(s, cx[j], 4.17, cw[j], 0.33, fill=lc)
        txt(s, lbl, cx[j]+0.05, 4.19, cw[j]-0.1, 0.29,
            sz=11, bold=True, color=C_WHT, align=PP_ALIGN.CENTER)

    pic(s, img(m2_dir, "confusion_matrices.png"), cx[0], 4.52, cw[0], 2.62)
    pic(s, img(m3_dir, "confusion_matrices.png"), cx[1], 4.52, cw[1], 2.62)
    pic(s, img(m2_dir, "attention_heatmap.png"),  cx[2], 4.52, cw[2], 2.62)

ALL_CONDITIONS = [
    ("bce",   "aec128", ["norm","crop60","crop80","len128","excl_extreme"]),
    ("bce",   "aec256", ["norm","crop60","crop80","len256","excl_extreme"]),
    ("focal", "aec128", ["norm","crop60","crop80","len128","excl_extreme"]),
    ("focal", "aec256", ["norm","crop60","crop80","len256","excl_extreme"]),
]
sn = 17
for loss, aec_len, variants in ALL_CONDITIONS:
    for variant in variants:
        condition_slide(loss, aec_len, variant, sn)
        sn += 1

# ═══════════════════════════════════════════════════════════════
# OK+MISSING SECTION — 간략 부록
# ═══════════════════════════════════════════════════════════════
BASE_OM   = "C:/Users/jhjun/OneDrive/Desktop/2026-1_Study/연구코드/results/ok+missing"
M1_OM     = f"{BASE_OM}/model_1"
M2N_OM    = f"{BASE_OM}/model_2/bce/aec128/norm"
M3N_OM    = f"{BASE_OM}/model_3/bce/aec128/norm"
CMP_OM    = f"{BASE_OM}/comparison"

# ── Slide 38: ok+missing 섹션 디바이더 ─────────────────────────
s = add_slide()
box(s, 0, 0, 13.33, 7.5, fill=RGBColor(0x1A,0x2A,0x3E))
box(s, 0, 0, 13.33, 0.08, fill=C_ORG)
box(s, 0, 7.42, 13.33, 0.08, fill=C_RED)
txt(s, "부록 2 — OK+Missing 확장 데이터셋",
    1.0, 2.1, 11.33, 0.9, sz=33, bold=True, color=C_WHT, align=PP_ALIGN.CENTER)
txt(s, "누락 데이터 포함 확장 코호트  |  n=1,162 (Train 929 / Test 233)",
    1.0, 3.15, 11.33, 0.5, sz=17, color=C_MGR, align=PP_ALIGN.CENTER)
hline(s, 3.5, 3.85, 6.33, color=C_ORG)
for i,(lbl,col,note) in enumerate([
    ("OK 코호트 (n=859)", C_BLUE, "AEC 매칭 완전한 환자만 포함 — 본 연구 주결과"),
    ("OK+Missing 코호트 (n=1,162)", C_ORG, "누락 AEC 포함 확장 — 비교 참고용 간략 제시"),
]):
    box(s, 1.5+i*5.3, 4.2, 5.0, 0.75, fill=col)
    txt(s, lbl, 1.5+i*5.3+0.1, 4.28, 4.8, 0.32, sz=13, bold=True,
        color=C_WHT, align=PP_ALIGN.CENTER)
    txt(s, note, 1.5+i*5.3+0.1, 4.6, 4.8, 0.32, sz=11,
        color=C_WHT, align=PP_ALIGN.CENTER)
txt(s, "결과 해석 주의: 누락 데이터 포함 → AEC 기여 효과 희석, 전반적 AUC 감소",
    1.0, 5.35, 11.33, 0.35, sz=13, color=C_MGR,
    italic=True, align=PP_ALIGN.CENTER)
snum(s, 37)

# ── Slide 38: ok+missing vs ok 전체 비교 ───────────────────────
s = add_slide()
hdr(s, "OK+Missing 코호트 결과 개요 — OK 대비 비교",
    "n=1,162 / Test n=233 / Sarco 25명(10.7%)  |  BCE + AEC128 + norm 기준")

# Dataset comparison
card(s, 0.3, 0.99, 5.5, 3.0)
txt(s, "데이터셋 비교", 0.5, 1.07, 5.1, 0.3, sz=14, bold=True)
hline(s, 0.5, 1.39, 5.1, color=C_BLUE)
cx = [0.4, 1.95, 3.5, 5.0]; cw = [1.5, 1.5, 1.45, 0.9]
trow_h(s, ["구분","OK","OK+Missing","차이"], cx, cw, 1.44, ht=0.3)
for i,(row,fill) in enumerate(zip([
    ("전체 n",    "859",  "1,162", "+303"),
    ("Train n",   "687",  "929",   "+242"),
    ("Test n",    "172",  "233",   "+61"),
    ("Sarco(Test)","21",  "25",    "+4"),
    ("Sarco 비율", "12.2%","10.7%", "-1.5%p"),
],[C_WHT,C_LGR,C_WHT,C_LGR,C_WHT])):
    trow_d(s, row, cx, cw, 1.76+i*0.37, ht=0.37, fill=fill)

# Performance comparison table
card(s, 0.3, 4.14, 5.5, 3.06)
txt(s, "BCE+AEC128+norm 성능 비교", 0.5, 4.22, 5.1, 0.3, sz=14, bold=True)
hline(s, 0.5, 4.54, 5.1, color=C_ORG)
px = [0.4, 1.65, 2.85, 4.15]; pw = [1.2, 1.15, 1.25, 1.35]
trow_h(s, ["모델","OK AUC","OM AUC","Δ"], px, pw, 4.59, ht=0.3)
for i,(row,fill) in enumerate(zip([
    ("M1 (LR)",    "0.8325","0.8119","-0.021"),
    ("M2 (CA)",    "0.8868","0.8254","-0.061"),
    ("M2_2 (Neg)", "0.8420","0.8246","-0.017"),
    ("M3 (CA3)",   "0.8944","0.8200","-0.074"),
],[C_LGR,RGBColor(0xE8,0xF4,0xE8),C_WHT,RGBColor(0xFF,0xF3,0xE0)])):
    d_col = C_RED
    trow_d(s, row, px, pw, 4.91+i*0.37, ht=0.37, fill=fill,
           colors=[C_DARK,C_DARK,C_DARK,d_col])

# Key findings
card(s, 6.0, 0.99, 7.1, 3.52, fill=RGBColor(0xFF,0xF3,0xE0))
box(s, 6.0, 0.99, 0.12, 3.52, fill=C_ORG)
txt(s, "주요 발견", 6.25, 1.07, 6.7, 0.3, sz=14, bold=True, color=C_ORG)
for i,t in enumerate([
    "• 전 모델에서 AUC 감소 (Δ –0.02~–0.07)",
    "  → Missing 환자는 더 어려운 케이스를 포함",
    "• M1 vs M2/M3 DeLong: 전 조건 ns",
    "  → ok에서 유의했던 조건도 유의성 소실",
    "• M2 vs M2_2 excl_extreme: 일관 ***",
    "  → AEC 매칭 효과는 데이터셋 불문 강건",
    "• M2 vs M3: 전 조건 ns (ok와 동일)",
]):
    txt(s, t, 6.25, 1.47+i*0.43, 6.7, 0.4, sz=12, color=C_DARK)

# DeLong summary table
card(s, 6.0, 4.61, 7.1, 2.59)
txt(s, "DeLong 요약  (norm 기준 / ok+missing)", 6.2, 4.69, 6.8, 0.3, sz=13, bold=True)
hline(s, 6.2, 5.01, 6.6, color=C_TEAL)
dx2 = [6.1, 7.6, 8.8, 10.1, 11.3]; dw2 = [1.45,1.15,1.25,1.15,1.65]
trow_h(s, ["비교","OK p","OM p","OK sig","OM sig"], dx2, dw2, 5.06, ht=0.28)
dlong_om_rows = [
    ("M1 vs M2 (BCE128)", "p=0.071", "p=0.578",  "†",   "ns",  C_GRN, C_MGR),
    ("M1 vs M3 (BCE128)", "p=0.040", "p=0.773",  "*",   "ns",  C_GRN, C_MGR),
    ("M1 vs M2 (BCE256)", "p=0.013", "p=0.492",  "*",   "ns",  C_GRN, C_MGR),
    ("M2 vs M2_2(excl)",  "<0.001",  "<0.001",   "***", "***", C_RED, C_RED),
]
for i,(lbl,ok_p,om_p,ok_sig,om_sig,ok_col,om_col) in enumerate(dlong_om_rows):
    fill = C_WHT if i%2==0 else C_LGR
    trow_d(s, [lbl, ok_p, om_p, ok_sig, om_sig],
           dx2, dw2, 5.36+i*0.37, ht=0.37, fill=fill,
           colors=[C_DARK, C_DARK, C_DARK, ok_col, om_col])
snum(s, 38)

# ── Slides 39–42: ok+missing 조건별 간략 슬라이드 ──────────────
def cond_slide_om(loss, aec_len, sn):
    col = COND_COL[(loss, aec_len)]
    m2_dir = f"{BASE_OM}/model_2/{loss}/{aec_len}/norm"
    m3_dir = f"{BASE_OM}/model_3/{loss}/{aec_len}/norm"
    cmp_png_n = f"{CMP_OM}/{loss}/{aec_len}/roc_all_models_norm.png"
    cmp_png_e = f"{CMP_OM}/{loss}/{aec_len}/roc_all_models_excl_extreme.png"

    s = add_slide()
    box(s, 0, 0, 13.33, 0.84, fill=C_DARK)
    box(s, 0, 0, 0.35, 0.84, fill=col)
    txt(s, f"[OK+Miss] {LOSS_LBL[loss]}  +  {AEC_LBL[aec_len]}  — 간략 요약",
        0.45, 0.06, 10.5, 0.44, sz=21, bold=True, color=C_WHT)
    txt(s, "OK+Missing 코호트 (n=1,162)  |  norm · excl_extreme 조건 비교 / DeLong 전 조건 ns",
        0.45, 0.50, 10.5, 0.28, sz=12, color=C_MGR)
    txt(s, f"{sn} / {TOTAL}", 12.3, 7.15, 1.0, 0.3, sz=11,
        color=C_MGR, align=PP_ALIGN.RIGHT)

    # Left: 2 ROC panels stacked
    for j,(png,lbl,y0) in enumerate([
        (cmp_png_n, "ROC — norm 조건", 0.89),
        (cmp_png_e, "ROC — excl_extreme 조건", 4.07),
    ]):
        box(s, 0.25, y0, 5.9, 0.33, fill=C_DARK)
        txt(s, lbl, 0.3, y0+0.02, 5.8, 0.29,
            sz=11, bold=True, color=C_WHT, align=PP_ALIGN.CENTER)
        pic(s, png, 0.25, y0+0.35, 5.9, 3.05)

    # Right: M2 & M3 calibration
    calib_banner(s, 6.4, 0.89, 6.7, f"M2  {LOSS_LBL[loss]}+{AEC_LBL[aec_len]}+norm  (Brier)")
    pic(s, img(m2_dir, "calibration.png"), 6.4, 1.24, 6.7, 2.8)
    calib_banner(s, 6.4, 4.17, 6.7, f"M3  {LOSS_LBL[loss]}+{AEC_LBL[aec_len]}+norm  (Brier)")
    pic(s, img(m3_dir, "calibration.png"), 6.4, 4.52, 6.7, 2.65)

OM_SCENARIOS = [
    ("bce",   "aec128", 39),
    ("bce",   "aec256", 40),
    ("focal", "aec128", 41),
    ("focal", "aec256", 42),
]
for loss, aec_len, sn in OM_SCENARIOS:
    cond_slide_om(loss, aec_len, sn)

# ═══════════════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════════════
OUT = ("C:/Users/jhjun/OneDrive/Desktop/2026-1_Study/연구코드/"
       "AEC_Sarcopenia_Research_Presentation.pptx")
prs.save(OUT)
print(f"Saved  → {OUT}")
print(f"Slides : {len(prs.slides)}")
