"""
Secondary Model (성능 중심 방식 — Late-fusion CNN)
AEC curve(256pt) → 1D CNN → feature vector
feature vector + Age/Sex/BMI → FC head → Low SMI 예측 (end-to-end)

ref: 이홍선교수님_260506.docx §7
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    brier_score_loss, confusion_matrix, classification_report,
    roc_curve,
)
from sklearn.calibration import calibration_curve

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_gangnam as load_and_prepare, train_test_idx, AEC_COLS, CLIN_COLS

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.normpath(
    os.path.join(_SCRIPT_DIR, '..', '..', 'results', '0508_2', 'secondary'))

BATCH   = 64
EPOCHS  = 300
LR      = 1e-3
SEED    = 42
CNN_DIM = 64   # AEC branch 출력 차원
torch.manual_seed(SEED)
np.random.seed(SEED)


# ── Dataset ───────────────────────────────────────────────
class _FusionDataset(Dataset):
    def __init__(self, X_aec: np.ndarray, X_clin: np.ndarray, y: np.ndarray):
        self.aec  = torch.tensor(X_aec,  dtype=torch.float32).unsqueeze(1)  # (N,1,256)
        self.clin = torch.tensor(X_clin, dtype=torch.float32)                # (N,3)
        self.y    = torch.tensor(y,      dtype=torch.float32).unsqueeze(1)

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.aec[i], self.clin[i], self.y[i]


def _loader(X_aec, X_clin, y, shuffle):
    return DataLoader(_FusionDataset(X_aec, X_clin, y),
                      batch_size=BATCH, shuffle=shuffle)


# ── Late-fusion CNN 모델 ───────────────────────────────────
class LateFusionCNN(nn.Module):
    """
    AEC branch: (1,256) → Conv1D × 4 → AdaptiveAvgPool → Linear(CNN_DIM)
    Fusion    : [CNN_DIM + n_clinical] → FC → 1 (logit)
    """
    def __init__(self, n_clinical: int = 3, cnn_dim: int = CNN_DIM):
        super().__init__()
        self.aec_branch = nn.Sequential(
            nn.Conv1d(1,   32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32,  64, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(256, cnn_dim), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.fusion = nn.Sequential(
            nn.Linear(cnn_dim + n_clinical, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, aec, clin):
        feat = self.aec_branch(aec)
        return self.fusion(torch.cat([feat, clin], dim=1))


# ── 학습 루프 ──────────────────────────────────────────────
def _train_model(X_aec_tr, X_clin_tr, y_tr, device) -> LateFusionCNN:
    model  = LateFusionCNN().to(device)
    crit   = nn.BCEWithLogitsLoss()
    opt    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    loader = _loader(X_aec_tr, X_clin_tr, y_tr, shuffle=True)

    best_loss, best_state = float('inf'), None
    patience, no_improve  = 30, 0

    for ep in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for aec_b, clin_b, y_b in loader:
            aec_b, clin_b, y_b = aec_b.to(device), clin_b.to(device), y_b.to(device)
            opt.zero_grad()
            loss = crit(model(aec_b, clin_b), y_b)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(y_b)
        sched.step()
        epoch_loss /= len(loader.dataset)
        if epoch_loss < best_loss:
            best_loss  = epoch_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def _predict(model: LateFusionCNN, X_aec, X_clin, device) -> np.ndarray:
    model.eval()
    ds = _FusionDataset(X_aec, X_clin, np.zeros(len(X_aec)))
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    probs = []
    for aec_b, clin_b, _ in loader:
        logit = model(aec_b.to(device), clin_b.to(device)).cpu()
        probs.append(torch.sigmoid(logit).numpy())
    return np.concatenate(probs).ravel()


# ── 평가 지표 ──────────────────────────────────────────────
def _metrics(y_true, y_prob) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        'AUC'  : roc_auc_score(y_true, y_prob),
        'AUPRC': average_precision_score(y_true, y_prob),
        'Brier': brier_score_loss(y_true, y_prob),
        'Acc'  : (y_pred == y_true).mean(),
    }


# ── 시각화 ────────────────────────────────────────────────
def _save_plots(y_true, y_prob, tag, subdir):
    os.makedirs(subdir, exist_ok=True)
    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 3, wspace=0.35)

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    ax = fig.add_subplot(gs[0])
    ax.plot(fpr, tpr, lw=2, label=f'AUC={auc_val:.3f}')
    ax.plot([0,1],[0,1],'k--',lw=1)
    ax.set(xlabel='FPR', ylabel='TPR', title=f'ROC — {tag}')
    ax.legend()

    # Calibration
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    ax = fig.add_subplot(gs[1])
    ax.plot(prob_pred, prob_true, 's-', lw=2, label='Model')
    ax.plot([0,1],[0,1],'k--',lw=1,label='Perfect')
    ax.set(xlabel='Mean predicted prob', ylabel='Fraction positive',
           title=f'Calibration — {tag}')
    ax.legend()

    # Confusion matrix
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    ax = fig.add_subplot(gs[2])
    im = ax.imshow(cm, cmap='Blues')
    fig.colorbar(im, ax=ax)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                    color='black', fontsize=12)
    ax.set(xticks=[0,1], xticklabels=['Normal','Low'],
           yticks=[0,1], yticklabels=['Normal','Low'],
           xlabel='Predicted', ylabel='Actual',
           title=f'Confusion Matrix — {tag}')

    path = os.path.join(subdir, f'{tag}_eval.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → 시각화 저장: {path}')


# ── 보고서 저장 ────────────────────────────────────────────
def _save_report(cv_metrics, test_metrics, cr_text, target, subdir):
    os.makedirs(subdir, exist_ok=True)
    keys = ['AUC', 'AUPRC', 'Brier', 'Acc']
    lines = [
        f'# Secondary Model Report — {target}',
        '',
        '**구조**: Late-fusion CNN — AEC branch + Age/Sex/BMI → end-to-end binary classification',
        '',
        '## 5-Fold CV 성능 (Train 80%)',
        '',
        '| Fold | AUC | AUPRC | Brier | Acc |',
        '|---|---|---|---|---|',
    ]
    for i, m in enumerate(cv_metrics, 1):
        lines.append(f'| {i} | {m["AUC"]:.4f} | {m["AUPRC"]:.4f} '
                     f'| {m["Brier"]:.4f} | {m["Acc"]:.4f} |')
    means = {k: np.mean([m[k] for m in cv_metrics]) for k in keys}
    stds  = {k: np.std([m[k] for m in cv_metrics])  for k in keys}
    lines += [
        f'| **Mean** | **{means["AUC"]:.4f}** | **{means["AUPRC"]:.4f}** '
        f'| **{means["Brier"]:.4f}** | **{means["Acc"]:.4f}** |',
        f'| **Std**  | **{stds["AUC"]:.4f}**  | **{stds["AUPRC"]:.4f}**  '
        f'| **{stds["Brier"]:.4f}**  | **{stds["Acc"]:.4f}**  |',
        '',
        '## Test Set 성능 (Test 20%)',
        '',
        '| AUC | AUPRC | Brier | Acc |',
        '|---|---|---|---|',
        f'| **{test_metrics["AUC"]:.4f}** | **{test_metrics["AUPRC"]:.4f}** '
        f'| **{test_metrics["Brier"]:.4f}** | **{test_metrics["Acc"]:.4f}** |',
        '',
        '## Classification Report (Test)',
        '',
        '```',
        cr_text.strip(),
        '```',
    ]
    path = os.path.join(subdir, 'Secondary_Report.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  → 보고서 저장: {path}')


# ── 메인 실행 ─────────────────────────────────────────────
def run_secondary_model(target: str = 'SMI'):
    print(f'\n{"="*60}')
    print(f'[Secondary Model] target={target}')
    print(f'{"="*60}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'  device: {device}')

    df = load_and_prepare(target)
    tr_idx, te_idx = train_test_idx(df, target)

    df_train = df.iloc[tr_idx].reset_index(drop=True)
    df_test  = df.iloc[te_idx].reset_index(drop=True)
    y_train  = df_train[f'{target}_bin'].values
    y_test   = df_test[f'{target}_bin'].values

    # ── 5-Fold CV ─────────────────────────────────────────
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_metrics = []

    for fold, (tr, vl) in enumerate(skf.split(df_train, y_train), 1):
        df_tr, df_vl = df_train.iloc[tr], df_train.iloc[vl]
        y_tr,  y_vl  = y_train[tr], y_train[vl]

        aec_sc  = StandardScaler()
        clin_sc = StandardScaler()
        X_aec_tr  = aec_sc.fit_transform(df_tr[AEC_COLS].values)
        X_aec_vl  = aec_sc.transform(df_vl[AEC_COLS].values)
        X_clin_tr = clin_sc.fit_transform(df_tr[CLIN_COLS].values)
        X_clin_vl = clin_sc.transform(df_vl[CLIN_COLS].values)

        model = _train_model(X_aec_tr, X_clin_tr, y_tr, device)
        prob_vl = _predict(model, X_aec_vl, X_clin_vl, device)

        m = _metrics(y_vl, prob_vl)
        cv_metrics.append(m)
        print(f'  Fold {fold}: AUC={m["AUC"]:.4f}  AUPRC={m["AUPRC"]:.4f}'
              f'  Brier={m["Brier"]:.4f}  Acc={m["Acc"]:.4f}')

    keys = ['AUC', 'AUPRC', 'Brier', 'Acc']
    means = {k: np.mean([m[k] for m in cv_metrics]) for k in keys}
    print(f'\n  CV Mean  AUC={means["AUC"]:.4f}  AUPRC={means["AUPRC"]:.4f}'
          f'  Brier={means["Brier"]:.4f}  Acc={means["Acc"]:.4f}')

    # ── 최종 모델: train 전체 학습 → test 평가 ──────────────
    print('\n  [Final model] training on full train set...')
    aec_sc_f  = StandardScaler()
    clin_sc_f = StandardScaler()
    X_aec_tr_f  = aec_sc_f.fit_transform(df_train[AEC_COLS].values)
    X_aec_te_f  = aec_sc_f.transform(df_test[AEC_COLS].values)
    X_clin_tr_f = clin_sc_f.fit_transform(df_train[CLIN_COLS].values)
    X_clin_te_f = clin_sc_f.transform(df_test[CLIN_COLS].values)

    model_f  = _train_model(X_aec_tr_f, X_clin_tr_f, y_train, device)
    prob_te  = _predict(model_f, X_aec_te_f, X_clin_te_f, device)

    test_m = _metrics(y_test, prob_te)
    print(f'  Test  AUC={test_m["AUC"]:.4f}  AUPRC={test_m["AUPRC"]:.4f}'
          f'  Brier={test_m["Brier"]:.4f}  Acc={test_m["Acc"]:.4f}')

    cr_text = classification_report(
        y_test, (prob_te >= 0.5).astype(int),
        target_names=['Normal', f'Low {target}'],
    )
    print(cr_text)

    subdir = os.path.join(RESULTS_DIR, target)
    _save_plots(y_test, prob_te, 'Secondary', subdir)
    _save_report(cv_metrics, test_m, cr_text, target, subdir)

    return test_m


if __name__ == '__main__':
    run_secondary_model('SMI')
