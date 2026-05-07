"""
Model 2: AEC-CNN only  (1D CNN on raw AEC curve, no clinical vars)
ref: 이홍선교수님_260506.docx §10 item 4
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_gangnam, train_test_idx, AEC_COLS
from utils import compute_metrics, save_eval_plots, save_report

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.normpath(
    os.path.join(_SCRIPT_DIR, '..', '..', 'results', '0508_2', 'model2_aec_cnn'))

BATCH  = 64
EPOCHS = 300
LR     = 1e-3
SEED   = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


class _AECDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


def _loader(X, y, shuffle):
    return DataLoader(_AECDataset(X, y), batch_size=BATCH, shuffle=shuffle)


class AECOnlyCNN(nn.Module):
    """AEC curve (1, L) -> single logit, L은 가변(AdaptiveAvgPool 사용)"""
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(1,   32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),  nn.ReLU(inplace=True),
            nn.Conv1d(32,  64, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64),  nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(inplace=True),
            nn.Dropout(0.3),    nn.Linear(64, 1),
        )
    def forward(self, x): return self.head(self.enc(x))


def _train(X_tr, y_tr, device) -> AECOnlyCNN:
    model  = AECOnlyCNN().to(device)
    crit   = nn.BCEWithLogitsLoss()
    opt    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    loader = _loader(X_tr, y_tr, shuffle=True)
    best_loss, best_state, no_imp = float('inf'), None, 0
    for _ in range(EPOCHS):
        model.train()
        ep_loss = sum(
            nn.functional.binary_cross_entropy_with_logits(
                model(Xb.to(device)), yb.to(device)).item() * len(yb)
            for Xb, yb in loader
        ) / len(loader.dataset)
        sched.step()
        if ep_loss < best_loss:
            best_loss, best_state, no_imp = ep_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            no_imp += 1
        if no_imp >= 30:
            break
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def _predict(model, X, device) -> np.ndarray:
    model.eval()
    ds = _AECDataset(X, np.zeros(len(X)))
    probs = []
    for Xb, _ in DataLoader(ds, batch_size=256, shuffle=False):
        probs.append(torch.sigmoid(model(Xb.to(device))).cpu().numpy())
    return np.concatenate(probs).ravel()


def run_model2(target: str = 'SMI') -> dict:
    print(f'\n{"="*60}')
    print(f'[Model 2] AEC-CNN only  |  target={target}')
    print(f'{"="*60}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'  device: {device}')

    df = load_gangnam(target)
    tr_idx, te_idx = train_test_idx(df, target)
    df_train, df_test = df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)
    y_train, y_test   = df_train[f'{target}_bin'].values, df_test[f'{target}_bin'].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_metrics = []

    for fold, (tr, vl) in enumerate(skf.split(df_train, y_train), 1):
        sc = StandardScaler()
        X_tr = sc.fit_transform(df_train.iloc[tr][AEC_COLS].values)
        X_vl = sc.transform(df_train.iloc[vl][AEC_COLS].values)
        y_tr, y_vl = y_train[tr], y_train[vl]
        model = _train(X_tr, y_tr, device)
        prob  = _predict(model, X_vl, device)
        m = compute_metrics(y_vl, prob)
        cv_metrics.append(m)
        print(f'  Fold {fold}: AUC={m["AUC"]:.4f}  AUPRC={m["AUPRC"]:.4f}'
              f'  Brier={m["Brier"]:.4f}  Acc={m["Acc"]:.4f}')

    means = {k: np.mean([m[k] for m in cv_metrics]) for k in ['AUC','AUPRC','Brier','Acc']}
    print(f'\n  CV Mean  AUC={means["AUC"]:.4f}  AUPRC={means["AUPRC"]:.4f}'
          f'  Brier={means["Brier"]:.4f}  Acc={means["Acc"]:.4f}')

    # 최종 모델
    sc_f = StandardScaler()
    X_tr_f = sc_f.fit_transform(df_train[AEC_COLS].values)
    X_te_f = sc_f.transform(df_test[AEC_COLS].values)
    model_f = _train(X_tr_f, y_train, device)
    prob_te = _predict(model_f, X_te_f, device)

    test_m  = compute_metrics(y_test, prob_te)
    print(f'  Test  AUC={test_m["AUC"]:.4f}  AUPRC={test_m["AUPRC"]:.4f}'
          f'  Brier={test_m["Brier"]:.4f}  Acc={test_m["Acc"]:.4f}')

    cr_text = classification_report(
        y_test, (prob_te >= 0.5).astype(int),
        target_names=['Normal', f'Low {target}'])
    print(cr_text)

    subdir = os.path.join(RESULTS_DIR, target)
    save_eval_plots(y_test, prob_te, 'Model2_AEConly', subdir)
    save_report(cv_metrics, test_m, cr_text,
                'Model2_AEConly', 'AEC-CNN only (no clinical)',
                target, subdir)
    return test_m


if __name__ == '__main__':
    run_model2('SMI')
