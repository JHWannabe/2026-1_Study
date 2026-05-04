import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, roc_curve, auc, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# ── 설정 ──────────────────────────────────────────────────────────────────────
DATA_PATH  = r'연구코드\data\강남_merged_features.xlsx'
BATCH_SIZE = 32
EPOCHS     = 5000
LR         = 1e-3
SEED       = 42
# ─────────────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)
np.random.seed(SEED)


# ── 데이터 ────────────────────────────────────────────────────────────────────

def load_data(data_path: str):
    df = pd.read_excel(data_path, sheet_name='metadata-bmi_add')

    gender = df['PatientSex'].copy()
    if gender.dtype == object:
        # 문자열 성별을 0/1 정수로 인코딩 (sorted → 알파벳 순 일관성 보장)
        gmap = {v: i for i, v in enumerate(sorted(gender.unique()))}
        print(f"성별 인코딩: {gmap}")
        gender = gender.map(gmap)

    X = np.column_stack([gender.values.astype(float),
                         df['PatientAge'].values.astype(float)])   # (N, 2)
    y = df['SMI'].values.astype(float)

    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    if (~valid).sum():
        print(f"NaN 샘플 {(~valid).sum()}개 제거")
    X, y = X[valid], y[valid]

    print(f"X: {X.shape}  SMI mean={y.mean():.3f} std={y.std():.3f}")
    return X, y


class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def _loader(X, y, shuffle): return DataLoader(TabularDataset(X, y), batch_size=BATCH_SIZE, shuffle=shuffle)


# ── 모델 ──────────────────────────────────────────────────────────────────────

class ClinicalMLP(nn.Module):
    """
    다층 퍼셉트론 (MLP) 회귀 모델
    - Linear → BatchNorm → ReLU → Dropout 패턴 반복
    - BatchNorm: 각 미니배치 내에서 정규화 → 학습률에 덜 민감, 빠른 수렴
    - Dropout: 학습 시 뉴런 일부를 랜덤 비활성화 → 과적합 억제
    - 은닉층 64→128→64: 점진적 확장 후 압축 (bottleneck 없는 hourglass)
    """
    def __init__(self, in_features=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),  nn.BatchNorm1d(64),  nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(64, 128),          nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, 64),          nn.BatchNorm1d(64),  nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x): return self.net(x)


# ── 학습 루프 ─────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y_b)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total, preds, trues = 0.0, [], []
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        p = model(X_b)
        total += criterion(p, y_b).item() * len(y_b)
        preds.append(p.cpu().numpy()); trues.append(y_b.cpu().numpy())
    return total / len(loader.dataset), np.concatenate(preds).ravel(), np.concatenate(trues).ravel()


def _fit(train_loader, val_loader, model, device, log_interval=500):
    """
    학습 루프 공통 헬퍼
    - HuberLoss(δ=1.0): |e|≤δ 이면 0.5·e²(MSE), 초과 시 δ(|e|−0.5δ)(MAE)
                         MSE의 수렴 안정성 + MAE의 이상치 강건성을 결합
    - AdamW: Adam + 분리된 weight decay → 정규화 효과가 Adam보다 정확
    - CosineAnnealingLR: LR(t) = η_min + 0.5(LR_max−η_min)(1+cos(πt/T_max))
                          초반 탐색, 후반 fine-tuning 자연스럽게 전환
    """
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_loss, best_state = float('inf'), None
    for epoch in range(1, EPOCHS + 1):
        tr = train_epoch(model, train_loader, optimizer, criterion, device)
        vl, _, _ = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        if vl < best_loss:
            best_loss  = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % log_interval == 0:
            print(f"  {epoch:5d} | train {tr:.4f} | val {vl:.4f}")
    return best_state, criterion


def _evaluate(model, loader, criterion, y_scaler, device):
    """역변환 후 (trues, preds) 반환"""
    _, ps, ts = eval_epoch(model, loader, criterion, device)
    preds = y_scaler.inverse_transform(ps.reshape(-1, 1)).ravel()
    trues = y_scaler.inverse_transform(ts.reshape(-1, 1)).ravel()
    return trues, preds


# ── 시각화 ────────────────────────────────────────────────────────────────────

def plot_results(trues, preds, save_dir, title='ClinicalMLP'):
    resid = preds - trues

    # 통계 검정
    r,  p_r  = stats.pearsonr(trues, preds)       # Pearson r: 선형 상관 유의성
    _,  p_sw = stats.shapiro(resid)               # Shapiro-Wilk: 잔차 정규성 (H0: 정규분포)
    _,  p_bt = stats.ttest_1samp(resid, 0)        # t-test: 잔차 평균 ≠ 0 (편향 유무)

    # 회귀 결과를 이진 분류로 변환 (SMI 하위 25% = 저근육량 위험군)
    thr        = np.percentile(trues, 25)
    y_bin      = (trues >= thr).astype(int)
    score_norm = (preds - preds.min()) / (np.ptp(preds) + 1e-8)
    fpr, tpr, _ = roc_curve(y_bin, score_norm)   # ROC-AUC: 이진 분류 성능 (0.5=랜덤, 1.0=완벽)
    roc_auc    = auc(fpr, tpr)
    cm         = confusion_matrix(y_bin, (preds >= thr).astype(int))

    fig = plt.figure(figsize=(18, 11))
    gs  = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.38)
    ax  = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    # ① Scatter
    lo, hi = min(trues.min(), preds.min()), max(trues.max(), preds.max())
    ax[0].scatter(trues, preds, alpha=0.6, edgecolors='k', linewidths=0.4)
    ax[0].plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='y = x')
    ax[0].set(xlabel='True SMI', ylabel='Predicted SMI',
              title=f'Predicted vs True\nr={r:.3f}  p={p_r:.2e}  R²={r2_score(trues, preds):.3f}')
    ax[0].legend()

    # ② 잔차 분포
    ax[1].hist(resid, bins=20, color='steelblue', edgecolor='white', alpha=0.85)
    ax[1].axvline(0, color='r', linestyle='--', lw=1.5)
    ax[1].set(xlabel='Residual (Pred−True)', ylabel='Count',
              title=f'Residual Distribution\nShapiro-Wilk p={p_sw:.3f}  |  Bias t-test p={p_bt:.3f}')

    # ③ Bland-Altman: 두 측정값의 일치도 분석 (Bland & Altman, 1986)
    mean_v = (trues + preds) / 2
    diff_v = preds - trues
    md, sd = diff_v.mean(), diff_v.std()
    ax[2].scatter(mean_v, diff_v, alpha=0.6, edgecolors='k', linewidths=0.4)
    ax[2].axhline(md,           color='r',    lw=1.5, label=f'Mean {md:+.3f}')
    ax[2].axhline(md + 1.96*sd, color='gray', lw=1.2, ls='--', label=f'+1.96SD {md+1.96*sd:+.3f}')
    ax[2].axhline(md - 1.96*sd, color='gray', lw=1.2, ls='--', label=f'−1.96SD {md-1.96*sd:+.3f}')
    ax[2].set(xlabel='Mean(True, Pred)', ylabel='Pred−True', title='Bland-Altman Plot')
    ax[2].legend(fontsize=8)

    # ④ ROC curve
    ax[3].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC={roc_auc:.3f}')
    ax[3].plot([0, 1], [0, 1], 'k--', lw=1)
    ax[3].set(xlabel='FPR', ylabel='TPR',
              title=f'ROC Curve (threshold=25th pct {thr:.2f})')
    ax[3].legend()

    # ⑤ Confusion Matrix
    im = ax[4].imshow(cm, cmap='Blues')
    fig.colorbar(im, ax=ax[4])
    cls = [f'<{thr:.1f}', f'≥{thr:.1f}']
    ax[4].set_xticks([0, 1]); ax[4].set_xticklabels(cls, fontsize=9)
    ax[4].set_yticks([0, 1]); ax[4].set_yticklabels(cls, fontsize=9)
    ax[4].set(xlabel='Predicted', ylabel='True', title='Confusion Matrix')
    for i in range(2):
        for j in range(2):
            ax[4].text(j, i, str(cm[i, j]), ha='center', va='center', color='black', fontsize=12)

    # ⑥ Q-Q plot: 잔차가 이론적 정규분포와 얼마나 일치하는지 시각적 확인
    (osm, osr), (slope, intercept, _) = stats.probplot(resid)
    ax[5].scatter(osm, osr, alpha=0.6, edgecolors='k', linewidths=0.4)
    qq = np.array([osm[0], osm[-1]])
    ax[5].plot(qq, slope*qq + intercept, 'r--', lw=1.5)
    ax[5].set(xlabel='Theoretical Quantiles', ylabel='Sample Quantiles',
              title=f'Q-Q Plot (Shapiro-Wilk p={p_sw:.3f})')

    fig.suptitle(f'{title} — Test Set Evaluation', fontsize=14, fontweight='bold', y=1.01)
    out = save_dir / f'{title.lower().replace(" ", "_")}_test_results.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"시각화 저장: {out}")
    print(f"\n[통계]  r={r:.4f} (p={p_r:.3e})  SW-p={p_sw:.4f}  bias-p={p_bt:.4f}  AUC={roc_auc:.4f}")


# ── 실행 모드 ─────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    X, y = load_data(DATA_PATH)

    # StandardScaler: (x - μ) / σ → 특징 간 스케일 통일, gradient 안정화
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_sc = x_scaler.fit_transform(X)
    y_sc = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

    X_tv, X_te, y_tv, y_te = train_test_split(X_sc, y_sc, test_size=0.15, random_state=SEED)
    X_tr, X_vl, y_tr, y_vl = train_test_split(X_tv, y_tv, test_size=0.15/0.85, random_state=SEED)
    print(f"Train {len(X_tr)} / Val {len(X_vl)} / Test {len(X_te)}")

    model = ClinicalMLP(in_features=X.shape[1]).to(device)
    print(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    print("\nEpoch | Train  | Val")
    best_state, criterion = _fit(_loader(X_tr, y_tr, True), _loader(X_vl, y_vl, False),
                                 model, device, log_interval=10)

    model.load_state_dict(best_state)
    trues, preds = _evaluate(model, _loader(X_te, y_te, False), criterion, y_scaler, device)
    print(f"\n[Test] MAE={mean_absolute_error(trues, preds):.4f}  R²={r2_score(trues, preds):.4f}")
    plot_results(trues, preds, Path(__file__).parent, title='ClinicalMLP')

    save_dir = Path(__file__).parent
    torch.save({'model_state': best_state, 'in_features': X.shape[1]}, save_dir / 'clinic_best.pt')
    with open(save_dir / 'clinic_x_scaler.pkl', 'wb') as f: pickle.dump(x_scaler, f)
    with open(save_dir / 'clinic_y_scaler.pkl', 'wb') as f: pickle.dump(y_scaler, f)
    print(f"저장: {save_dir / 'clinic_best.pt'}")


def run_cv(n_splits=5):
    """
    K-Fold Cross Validation (k=5)
    - 데이터를 k개로 분할, 각 fold를 한 번씩 validation으로 사용
    - 전체 데이터를 고르게 활용해 성능 추정의 분산을 줄임
    - hold-out test set은 CV와 완전히 분리해 최종 평가에만 사용
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    X, y = load_data(DATA_PATH)
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_sc = x_scaler.fit_transform(X)
    y_sc = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

    X_tv, X_te, y_tv, y_te = train_test_split(X_sc, y_sc, test_size=0.15, random_state=SEED)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    fold_maes, fold_r2s = [], []
    best_mae_cv, best_state_cv = float('inf'), None

    for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_tv), 1):
        print(f"\n── Fold {fold}/{n_splits} (train {len(tr_idx)}, val {len(vl_idx)}) ──")
        model = ClinicalMLP(in_features=X.shape[1]).to(device)
        best_state, criterion = _fit(
            _loader(X_tv[tr_idx], y_tv[tr_idx], True),
            _loader(X_tv[vl_idx], y_tv[vl_idx], False),
            model, device, log_interval=500)

        model.load_state_dict(best_state)
        trues, preds = _evaluate(model, _loader(X_tv[vl_idx], y_tv[vl_idx], False),
                                 criterion, y_scaler, device)
        mae, r2 = mean_absolute_error(trues, preds), r2_score(trues, preds)
        fold_maes.append(mae); fold_r2s.append(r2)
        print(f"  → Fold {fold}  MAE={mae:.4f}  R²={r2:.4f}")

        if mae < best_mae_cv:
            best_mae_cv, best_state_cv = mae, best_state

    print(f"\n{'='*45}\n {n_splits}-Fold CV (ClinicalMLP)")
    for i, (m, r) in enumerate(zip(fold_maes, fold_r2s), 1):
        print(f"  Fold {i}: MAE {m:.4f}  R² {r:.4f}")
    print(f"  Mean : MAE {np.mean(fold_maes):.4f}±{np.std(fold_maes):.4f}"
          f"  R² {np.mean(fold_r2s):.4f}±{np.std(fold_r2s):.4f}")

    # 최적 fold 모델로 hold-out test 평가
    model_final = ClinicalMLP(in_features=X.shape[1]).to(device)
    model_final.load_state_dict(best_state_cv)
    criterion = nn.HuberLoss(delta=1.0)
    trues, preds = _evaluate(model_final, _loader(X_te, y_te, False), criterion, y_scaler, device)
    print(f"\n[Best fold → Test] MAE={mean_absolute_error(trues, preds):.4f}"
          f"  R²={r2_score(trues, preds):.4f}")
    plot_results(trues, preds, Path(__file__).parent, title='ClinicalMLP')


def test_only(ckpt_path: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    ckpt_dir = Path(ckpt_path).parent
    with open(ckpt_dir / 'clinic_x_scaler.pkl', 'rb') as f: x_scaler = pickle.load(f)
    with open(ckpt_dir / 'clinic_y_scaler.pkl', 'rb') as f: y_scaler = pickle.load(f)

    X, y = load_data(DATA_PATH)
    X_sc = x_scaler.transform(X)
    y_sc = y_scaler.transform(y.reshape(-1, 1)).ravel()
    _, X_te, _, y_te = train_test_split(X_sc, y_sc, test_size=0.15, random_state=SEED)

    model = ClinicalMLP(in_features=ckpt['in_features']).to(device)
    model.load_state_dict(ckpt['model_state'])
    criterion = nn.HuberLoss(delta=1.0)
    trues, preds = _evaluate(model, _loader(X_te, y_te, False), criterion, y_scaler, device)
    print(f"[Test] MAE={mean_absolute_error(trues, preds):.4f}  R²={r2_score(trues, preds):.4f}")
    plot_results(trues, preds, Path(ckpt_path).parent, title='ClinicalMLP')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-only', metavar='CKPT', default=None)
    parser.add_argument('--cv', action='store_true')
    args = parser.parse_args()

    if args.test_only:   test_only(args.test_only)
    elif args.cv:        run_cv()
    else:                main()
