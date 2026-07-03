"""06_cross_attention_model.py / 07_cross_attention_v2.py의 CrossAttentionNet 구조를
발표자료용 flow diagram으로 그린다. 데이터 분석 파이프라인이 아니라 순수 시각화용
스크립트라 code/ 번호 시퀀스에는 포함하지 않았다.

출력: figures/diagrams/cross_attention_architecture.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patches import ConnectionStyle

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(BASE_DIR, "..", "figures", "diagrams")
os.makedirs(FIGURES_DIR, exist_ok=True)

COLOR_INPUT = "#DCE9F9"
COLOR_NORMALIZE = "#FBEEDC"
COLOR_EMBED = "#DDF2E4"
COLOR_ATTN = "#FCE4C2"
COLOR_NORM = "#EDEBF7"
COLOR_SKIP = "#F2F2F2"
COLOR_OUT = "#F9D9DD"
EDGE = "#4A4A4A"


def box(ax, xy, w, h, text, color, fontsize=11.5, weight="normal", style="round,pad=0.02,rounding_size=0.02"):
    x, y = xy
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=style, linewidth=1.3, edgecolor=EDGE, facecolor=color, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, weight=weight,
             linespacing=1.35, zorder=3)
    return (x, y, w, h)


def anchor(b, side):
    x, y, w, h = b
    return {
        "top": (x, y + h / 2), "bottom": (x, y - h / 2),
        "left": (x - w / 2, y), "right": (x + w / 2, y),
    }[side]


def arrow(ax, b1, b2, start="top", end="bottom", style="-|>", lw=1.4, color=EDGE,
          connectionstyle=None, ls="solid"):
    p1, p2 = anchor(b1, start), anchor(b2, end)
    kw = dict(arrowstyle=style, mutation_scale=14, linewidth=lw, color=color, zorder=1, linestyle=ls)
    if connectionstyle:
        kw["connectionstyle"] = connectionstyle
    ax.add_patch(FancyArrowPatch(p1, p2, **kw))


def elbow(ax, points, lw=1.4, color=EDGE, ls="dashed"):
    """여러 waypoint를 직각/직선 구간으로 잇는 화살표. 마지막 구간에만 화살촉을 붙인다."""
    for i in range(len(points) - 1):
        style = "-|>" if i == len(points) - 2 else "-"
        ax.add_patch(FancyArrowPatch(points[i], points[i + 1], arrowstyle=style, mutation_scale=14,
                                      linewidth=lw, color=color, linestyle=ls, zorder=1))


def main():
    fig, ax = plt.subplots(figsize=(14.5, 10.6))
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 11.2)
    ax.axis("off")

    # ---- Row 1: 입력(raw) ----
    clinic_in = box(ax, (3.2, 9.6), 4.6, 0.8,
                     "clinic (4개 스칼라)\nAge, Sex_M, Height, Weight", COLOR_INPUT, weight="bold")
    aec_in = box(ax, (10.2, 9.6), 4.6, 0.8,
                 "aec (128개 스칼라)\naec_1 ~ aec_128 원곡선", COLOR_INPUT, weight="bold")

    # ---- Row 2: 정규화 (run_cross_attention, 매 fold마다 train fold 기준으로 재적합) ----
    clinic_norm = box(ax, (3.2, 8.2), 5.0, 0.9,
                       "StandardScaler(변수별)\ntrain fold 기준 fit → 정규화된 clinic",
                       COLOR_NORMALIZE, fontsize=11)
    aec_norm = box(ax, (10.2, 8.2), 5.6, 1.05,
                    "scale_aec(): global(레벨 보존)\n또는 patient_wise(레벨 제거, 모양만)\n→ 정규화된 aec (128)",
                    COLOR_NORMALIZE, fontsize=11)
    arrow(ax, clinic_in, clinic_norm)
    arrow(ax, aec_in, aec_norm)

    # ---- Row 3: 토큰화(임베딩) ----
    clinic_tok = box(ax, (3.2, 6.7), 5.0, 1.0,
                      "clinic_value_proj: Linear(1→16)\n+ clinic_id_embed(변수별)\n→ clinic_tok (4토큰 × 16)",
                      COLOR_EMBED)
    aec_tok = box(ax, (10.2, 6.7), 5.0, 1.0,
                  "aec_value_proj: Linear(1→16)\n+ aec_pos_embed(위치별)\n→ aec_tok (128토큰 × 16)",
                  COLOR_EMBED)
    arrow(ax, clinic_norm, clinic_tok)
    arrow(ax, aec_norm, aec_tok)

    # ---- Row 4: Cross-Attention ----
    attn = box(ax, (6.7, 5.1), 7.6, 1.05,
                "nn.MultiheadAttention(d_model=16, n_heads=2, dropout=0.2)\n"
                "query = clinic_tok (4)  |  key = value = aec_tok (128)\n→ attn_out (4토큰 × 16), attn_weights (4×128)",
                COLOR_ATTN, weight="bold")
    arrow(ax, clinic_tok, attn, start="bottom", end="left",
          connectionstyle=ConnectionStyle("Arc3", rad=-0.12))
    arrow(ax, aec_tok, attn, start="bottom", end="right",
          connectionstyle=ConnectionStyle("Arc3", rad=0.12))

    # ---- Row 5: Residual + LayerNorm ----
    norm = box(ax, (6.7, 3.6), 6.6, 0.9,
                "h = LayerNorm( clinic_tok + attn_out )   (residual)\n→ mean pool(4토큰) → pooled (16)",
                COLOR_NORM)
    arrow(ax, attn, norm)
    # clinic_tok -> norm residual: attn 왼쪽 여백을 따라 내려가는 우회 경로(질의 화살표와 구분되도록)
    cl_left = anchor(clinic_tok, "left")
    norm_left = anchor(norm, "left")
    elbow(ax, [cl_left, (0.35, cl_left[1]), (0.35, norm_left[1]), norm_left],
          color="#5b7fb5", lw=1.3)

    # ---- Row 6: MLP head ----
    head = box(ax, (6.7, 2.2), 6.6, 0.9,
                "MLP head: Linear(16→8) → ReLU → Dropout(0.2) → Linear(8→1)\n→ head_logit",
                COLOR_EMBED)
    arrow(ax, norm, head)

    # ---- Clinic-skip 분기 (raw clinic -> skip_logit, DAFT류) ----
    skip = box(ax, (12.6, 3.6), 3.2, 1.05,
                "clinic-skip (DAFT류)\nLinear(4→1)(clinic)\n→ skip_logit\n(Polsterl+, MICCAI 2021)",
                COLOR_SKIP, fontsize=10.5)
    # 다른 박스와 겹치지 않도록 맨 위/맨 오른쪽 여백을 따라 우회
    clinic_in_top = anchor(clinic_in, "top")
    skip_top = anchor(skip, "top")
    elbow(ax, [
        (5.0, clinic_in_top[1]), (5.0, 10.5), (13.6, 10.5), (13.6, skip_top[1]),
    ], color="#888888", lw=1.2)

    # ---- 합산 및 출력 ----
    out = box(ax, (8.6, 0.8), 9.4, 0.8,
               "logit = head_logit + skip_logit  →  sigmoid  →  P(저SMI)",
               COLOR_OUT, weight="bold")
    arrow(ax, head, out, start="bottom", end="left",
          connectionstyle=ConnectionStyle("Arc3", rad=-0.1))
    arrow(ax, skip, out, start="bottom", end="right",
          connectionstyle=ConnectionStyle("Arc3", rad=0.15))

    # 범례성 라벨(원본 raw clinic이 skip으로 들어간다는 것을 텍스트로 명시)
    ax.text(5.0, clinic_in_top[1], "  raw clinic →", ha="left", va="bottom", fontsize=9.5, color="#666666")

    # ---- 범례 ----
    ax.plot([0.3, 1.0], [1.4, 1.4], color=EDGE, lw=1.4, ls="solid")
    ax.text(1.15, 1.4, "순전파(feed-forward)", ha="left", va="center", fontsize=9.5, color="#333333")
    ax.plot([0.3, 1.0], [1.0, 1.0], color="#5b7fb5", lw=1.3, ls="dashed")
    ax.text(1.15, 1.0, "residual (clinic_tok → LayerNorm)", ha="left", va="center", fontsize=9.5, color="#333333")
    ax.plot([0.3, 1.0], [0.6, 0.6], color="#888888", lw=1.2, ls="dashed")
    ax.text(1.15, 0.6, "clinic-skip (raw clinic → skip_logit)", ha="left", va="center", fontsize=9.5, color="#333333")

    ax.set_title("Cross-Attention 아키텍처 흐름도 (CrossAttentionNet, 06/07_cross_attention_*.py)",
                  fontsize=13.5, weight="bold", pad=16)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "cross_attention_architecture.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print("저장:", path)


if __name__ == "__main__":
    main()
