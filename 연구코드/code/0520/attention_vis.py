"""
CrossAttention 가중치 시각화 모듈 (0520).

주요 함수:
  extract_attention_weights   — ClinAECCrossAttn에서 배치 단위 attention 추출
  plot_attention_heatmap      — Sarcopenia vs Normal × 임상 피처 × AEC 토큰 히트맵
  plot_aec_attention_overlay  — AEC 신호 위에 attention 가중치 오버레이
  plot_sex_attention_compare  — 남성/여성 attention 패턴 비교
  run_attention_analysis      — 전체 파이프라인
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

from config import DEVICE, EPOCHS, RESULTS_DIR_ATTN
from models import (ClinAECCrossAttn, build_cross_attn,
                    DualDataset, make_dual_loaders, train_cross_epoch)
from data import split_data_dual

CLINICAL_FEAT_NAMES = ["Age", "Sex", "BMI"]


# ── Attention 추출 ────────────────────────────────────────────

@torch.no_grad()
def extract_attention_weights(model: ClinAECCrossAttn,
                               X_clin: np.ndarray,
                               X_aec: np.ndarray,
                               batch_size: int = 64):
    """
    Returns
    -------
    attn_c2a : (N, n_clin_tokens, n_aec_tokens)   Clinical → AEC
    attn_a2c : (N, n_aec_tokens,  n_clin_tokens)  AEC → Clinical
    """
    model.eval()
    c2a_list, a2c_list = [], []
    N = len(X_clin)
    xc_t = torch.tensor(X_clin, dtype=torch.float32)
    xa_t = torch.tensor(X_aec,  dtype=torch.float32)
    for start in range(0, N, batch_size):
        xc = xc_t[start:start + batch_size].to(DEVICE)
        xa = xa_t[start:start + batch_size].to(DEVICE)
        _, attn = model(xc, xa, return_attention=True)
        c2a_list.append(attn["clinical_to_aec"].cpu().numpy())
        a2c_list.append(attn["aec_to_clinical"].cpu().numpy())
    return np.concatenate(c2a_list, axis=0), np.concatenate(a2c_list, axis=0)


# ── 시각화 함수 ───────────────────────────────────────────────

def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_attention_heatmap(attn_c2a: np.ndarray, y: np.ndarray,
                           out_dir: str, suffix: str = ""):
    """
    Clinical features × AEC tokens attention heatmap.

    attn_c2a : (N, n_clin=3, n_aec_tokens=32)
    y        : (N,) binary label — 0=Normal, 1=Sarcopenia
    """
    groups = {0: "Normal", 1: "Sarcopenia"}
    n_aec_tokens = attn_c2a.shape[-1]
    token_pos = np.linspace(0, 100, n_aec_tokens)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    vmin = attn_c2a.min()
    vmax = attn_c2a.max()

    for ax, (label, name) in zip(axes, groups.items()):
        mask = y == label
        if mask.sum() == 0:
            ax.set_title(f"{name} (n=0)")
            continue
        mean_attn = attn_c2a[mask].mean(axis=0)  # (3, 32)
        im = ax.imshow(mean_attn, aspect="auto",
                       extent=[0, 100, 2.5, -0.5],
                       cmap="YlOrRd", vmin=vmin, vmax=vmax)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(CLINICAL_FEAT_NAMES)
        ax.set_xlabel("AEC token position (% scan length)")
        ax.set_title(f"{name}  (n={mask.sum()})")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Attention weight")

    fig.suptitle("Clinical → AEC Attention  (mean per group)", fontsize=13, y=1.02)
    fname = f"attn_heatmap_c2a{suffix}.png"
    _save(fig, os.path.join(out_dir, fname))
    print(f"  [Attn] saved {fname}")


def plot_aec_attention_overlay(X_aec: np.ndarray,
                                attn_c2a: np.ndarray,
                                y: np.ndarray,
                                sex: np.ndarray,
                                out_dir: str):
    """
    Mean AEC curve overlaid with aggregated Clinical→AEC attention per group.

    Creates a 2×2 grid: rows = {Normal, Sarcopenia}, cols = {Male, Female}.
    AEC amplitude (left y-axis) + mean attention (right y-axis, shaded).
    """
    n_aec = X_aec.shape[1]
    n_tokens = attn_c2a.shape[-1]
    aec_pos = np.linspace(0, 100, n_aec)
    tok_pos = np.linspace(0, 100, n_tokens)

    groups = [(0, "M", "Normal / Male"),   (0, "F", "Normal / Female"),
              (1, "M", "Sarco / Male"),    (1, "F", "Sarco / Female")]
    colors = ["#2196F3", "#E91E63", "#FF5722", "#9C27B0"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    axes = axes.flatten()

    for ax, (lbl, sx, title), color in zip(axes, groups, colors):
        mask = (y == lbl) & (sex == sx)
        n = mask.sum()
        ax2 = ax.twinx()
        if n == 0:
            ax.set_title(f"{title}  (n=0)")
            continue

        mean_aec = X_aec[mask].mean(axis=0)
        std_aec  = X_aec[mask].std(axis=0)
        mean_attn_agg = attn_c2a[mask].mean(axis=0).mean(axis=0)  # mean over clinical, then batch → (n_tokens,)

        ax.plot(aec_pos, mean_aec, color=color, lw=1.5, label="AEC signal")
        ax.fill_between(aec_pos, mean_aec - std_aec, mean_aec + std_aec,
                        alpha=0.2, color=color)
        ax.set_ylabel("AEC amplitude", color=color)
        ax.tick_params(axis='y', labelcolor=color)

        ax2.fill_between(tok_pos, 0, mean_attn_agg,
                         alpha=0.45, color="gray", label="Attention")
        ax2.set_ylabel("Mean attention weight", color="gray")
        ax2.tick_params(axis='y', labelcolor="gray")
        ax2.set_ylim(bottom=0)

        ax.set_title(f"{title}  (n={n})")
        ax.set_xlabel("Scan position (%)")

    fig.suptitle("AEC Signal + Clinical→AEC Attention Overlay", fontsize=13)
    fig.tight_layout()
    fname = "attn_aec_overlay.png"
    _save(fig, os.path.join(out_dir, fname))
    print(f"  [Attn] saved {fname}")


def plot_sex_attention_compare(attn_c2a: np.ndarray,
                                attn_a2c: np.ndarray,
                                sex: np.ndarray,
                                out_dir: str):
    """
    Side-by-side M vs F attention heatmaps for both directions.

    attn_c2a : (N, 3, 32)   Clinical → AEC
    attn_a2c : (N, 32, 3)   AEC → Clinical
    """
    n_tokens = attn_c2a.shape[-1]
    tok_pos = np.linspace(0, 100, n_tokens)

    fig = plt.figure(figsize=(16, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    titles = [("M", "Male"), ("F", "Female")]

    # Row 0: Clinical → AEC
    vmin_c2a = attn_c2a.min(); vmax_c2a = attn_c2a.max()
    for col, (sx, label) in enumerate(titles):
        ax = fig.add_subplot(gs[0, col])
        mask = sex == sx
        if mask.sum() == 0:
            ax.set_title(f"Clin→AEC  {label} (n=0)")
            continue
        mean_attn = attn_c2a[mask].mean(axis=0)  # (3, 32)
        im = ax.imshow(mean_attn, aspect="auto",
                       extent=[0, 100, 2.5, -0.5],
                       cmap="YlOrRd", vmin=vmin_c2a, vmax=vmax_c2a)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(CLINICAL_FEAT_NAMES)
        ax.set_xlabel("AEC token pos (%)")
        ax.set_title(f"Clin→AEC  {label}  (n={mask.sum()})")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 1: AEC → Clinical (bar chart — mean attention each clinical feature receives)
    for col, (sx, label) in enumerate(titles):
        ax = fig.add_subplot(gs[1, col])
        mask = sex == sx
        if mask.sum() == 0:
            ax.set_title(f"AEC→Clin  {label} (n=0)")
            continue
        mean_a2c = attn_a2c[mask].mean(axis=0)  # (32, 3)
        mean_per_feat = mean_a2c.mean(axis=0)    # (3,) — avg attention per clinical feature
        bar_colors = ["#2196F3", "#4CAF50", "#FF9800"]
        ax.bar(CLINICAL_FEAT_NAMES, mean_per_feat, color=bar_colors, alpha=0.8)
        ax.set_ylabel("Mean attention weight")
        ax.set_title(f"AEC→Clin  {label}  (n={mask.sum()})")
        ax.set_ylim(0, max(mean_per_feat) * 1.3)
        for i, v in enumerate(mean_per_feat):
            ax.text(i, v + max(mean_per_feat) * 0.03, f"{v:.4f}",
                    ha="center", fontsize=9)

    fig.suptitle("Sex-stratified Attention Patterns", fontsize=13)
    fname = "attn_sex_compare.png"
    _save(fig, os.path.join(out_dir, fname))
    print(f"  [Attn] saved {fname}")


def plot_attention_profile(attn_c2a: np.ndarray,
                            X_aec: np.ndarray,
                            y: np.ndarray,
                            out_dir: str):
    """
    Line plot: mean attention profile per clinical feature, Sarco vs Normal.
    Shows which AEC scan positions each clinical feature attends to.
    """
    n_tokens = attn_c2a.shape[-1]
    tok_pos = np.linspace(0, 100, n_tokens)
    linestyles = ["-", "--", ":"]
    group_colors = {0: "#2196F3", 1: "#F44336"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for fi, feat_name in enumerate(CLINICAL_FEAT_NAMES):
        ax = axes[fi]
        for lbl, gname in [(0, "Normal"), (1, "Sarcopenia")]:
            mask = y == lbl
            if mask.sum() == 0:
                continue
            profile = attn_c2a[mask, fi, :].mean(axis=0)
            std_p   = attn_c2a[mask, fi, :].std(axis=0)
            color = group_colors[lbl]
            ax.plot(tok_pos, profile, color=color, lw=2, label=gname)
            ax.fill_between(tok_pos, profile - std_p, profile + std_p,
                            alpha=0.15, color=color)
        ax.set_title(f"Feature: {feat_name}")
        ax.set_xlabel("AEC token pos (%)")
        if fi == 0:
            ax.set_ylabel("Attention weight")
        ax.legend(fontsize=8)

    fig.suptitle("Clinical Feature Attention Profiles  (Sarco vs Normal)", fontsize=13)
    fig.tight_layout()
    fname = "attn_profile_by_feat.png"
    _save(fig, os.path.join(out_dir, fname))
    print(f"  [Attn] saved {fname}")


# ── 학습 헬퍼 ─────────────────────────────────────────────────

def _train_model_for_attention(X_clin_cv, X_aec_cv, y_cv,
                                med_epoch: int,
                                scale_clin: bool = True,
                                scale_aec:  bool = False):
    """CV 데이터 전체로 ClinAECCrossAttn을 med_epoch epoch 학습 후 반환."""
    from sklearn.preprocessing import StandardScaler
    X_c = X_clin_cv.copy()
    X_a = X_aec_cv.copy()
    if scale_clin:
        sc = StandardScaler()
        X_c[:, [0, 2]] = sc.fit_transform(X_c[:, [0, 2]])
    if scale_aec:
        sa = StandardScaler()
        X_a = sa.fit_transform(X_a)

    model, crit, opt, sched = build_cross_attn(y_cv)
    from torch.utils.data import DataLoader
    from models import DualDataset
    loader = DataLoader(DualDataset(X_c, X_a, y_cv),
                        batch_size=64, shuffle=True)
    for ep in range(med_epoch):
        train_cross_epoch(model, loader, crit, opt)
        sched.step()
    return model, X_c, X_a


# ── 전체 파이프라인 ───────────────────────────────────────────

def run_attention_analysis(X_clin_cv, X_aec_cv, y_cv, sex_cv,
                            X_clin_te, X_aec_te, y_te, sex_te,
                            med_epoch: int,
                            scale_clin: bool = True,
                            scale_aec:  bool = False,
                            out_dir: str = RESULTS_DIR_ATTN):
    """
    ClinAECCrossAttn을 CV 데이터로 학습 후 test set attention을 추출·시각화.

    Saves:
      attn_heatmap_c2a.png     — group-stratified Clinical→AEC heatmap
      attn_aec_overlay.png     — AEC + attention overlay (2×2 sex × class grid)
      attn_sex_compare.png     — M vs F attention patterns
      attn_profile_by_feat.png — per-feature attention profiles
    """
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[Attention] Training ClinAECCrossAttn for {med_epoch} epochs ...")
    model, X_c_cv, X_a_cv = _train_model_for_attention(
        X_clin_cv, X_aec_cv, y_cv, med_epoch, scale_clin, scale_aec
    )

    # Test set preprocessing (same scaler state not tracked — use raw for overlay,
    # but scale for model inference to match training distribution)
    from sklearn.preprocessing import StandardScaler
    X_c_te = X_clin_te.copy()
    X_a_te = X_aec_te.copy()
    if scale_clin:
        sc = StandardScaler()
        sc.fit(X_clin_cv[:, [0, 2]])
        X_c_te[:, [0, 2]] = sc.transform(X_c_te[:, [0, 2]])
    if scale_aec:
        sa = StandardScaler()
        sa.fit(X_aec_cv)
        X_a_te = sa.transform(X_a_te).astype(np.float32)

    print("[Attention] Extracting attention weights on test set ...")
    attn_c2a, attn_a2c = extract_attention_weights(model, X_c_te, X_a_te)

    print(f"[Attention] attn_c2a shape: {attn_c2a.shape}")
    print(f"[Attention] attn_a2c shape: {attn_a2c.shape}")

    plot_attention_heatmap(attn_c2a, y_te, out_dir)
    plot_aec_attention_overlay(X_aec_te, attn_c2a, y_te, sex_te, out_dir)
    plot_sex_attention_compare(attn_c2a, attn_a2c, sex_te, out_dir)
    plot_attention_profile(attn_c2a, X_aec_te, y_te, out_dir)

    _save_attention_report_md(attn_c2a, attn_a2c, y_te, sex_te, med_epoch, out_dir)
    print(f"[Attention] All plots saved → {out_dir}")


def _save_attention_report_md(attn_c2a, attn_a2c, y_te, sex_te,
                               med_epoch, out_dir):
    """Attention 통계 요약 markdown 저장."""
    lines = [
        "# Attention Weight Analysis",
        "",
        f"Trained ClinAECCrossAttn for **{med_epoch}** epochs on CV set.",
        f"Attention extracted on test set (N={len(y_te)}).",
        "",
        "## Clinical → AEC  (mean attention weight per feature)",
        "",
        "| Feature | Normal | Sarcopenia | Male | Female |",
        "|---------|-------:|-----------:|-----:|-------:|",
    ]
    for fi, fname in enumerate(CLINICAL_FEAT_NAMES):
        val = {
            "Normal":     attn_c2a[y_te == 0, fi, :].mean() if (y_te == 0).any() else float("nan"),
            "Sarco":      attn_c2a[y_te == 1, fi, :].mean() if (y_te == 1).any() else float("nan"),
            "Male":       attn_c2a[sex_te == "M", fi, :].mean() if (sex_te == "M").any() else float("nan"),
            "Female":     attn_c2a[sex_te == "F", fi, :].mean() if (sex_te == "F").any() else float("nan"),
        }
        lines.append(
            f"| {fname} | {val['Normal']:.5f} | {val['Sarco']:.5f} "
            f"| {val['Male']:.5f} | {val['Female']:.5f} |"
        )

    lines += [
        "",
        "## AEC → Clinical  (mean attention each clinical feature receives, averaged over AEC tokens)",
        "",
        "| Feature | Overall | Normal | Sarcopenia | Male | Female |",
        "|---------|--------:|-------:|-----------:|-----:|-------:|",
    ]
    for fi, fname in enumerate(CLINICAL_FEAT_NAMES):
        # attn_a2c: (N, n_tokens, 3) → mean over tokens → (N, 3)
        per_sample = attn_a2c[:, :, fi].mean(axis=1)
        val = {
            "all":    per_sample.mean(),
            "Normal": per_sample[y_te == 0].mean() if (y_te == 0).any() else float("nan"),
            "Sarco":  per_sample[y_te == 1].mean() if (y_te == 1).any() else float("nan"),
            "Male":   per_sample[sex_te == "M"].mean() if (sex_te == "M").any() else float("nan"),
            "Female": per_sample[sex_te == "F"].mean() if (sex_te == "F").any() else float("nan"),
        }
        lines.append(
            f"| {fname} | {val['all']:.5f} | {val['Normal']:.5f} | {val['Sarco']:.5f} "
            f"| {val['Male']:.5f} | {val['Female']:.5f} |"
        )

    lines += [
        "",
        "## Figures",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `attn_heatmap_c2a.png` | Clinical→AEC attention heatmap (Normal vs Sarcopenia) |",
        "| `attn_aec_overlay.png` | AEC signal + attention overlay (2×2: class × sex) |",
        "| `attn_sex_compare.png` | M vs F attention patterns (Clin→AEC heatmap + AEC→Clin bar) |",
        "| `attn_profile_by_feat.png` | Per-feature attention profile (Sarco vs Normal) |",
    ]

    md_path = os.path.join(out_dir, "attention_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [Attn] saved attention_report.md")
