import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── 설정 ─────────────────────────────────────────────────────────────────────
DATA_PATH  = r'연구코드\data\강남_merged_features.xlsx'
SEQ_LEN    = 256    # 모든 시퀀스를 이 길이로 통일 (padding/truncate)
BATCH_SIZE = 32
EPOCHS     = 1000
LR         = 5e-4
SEED       = 42
# ─────────────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)
np.random.seed(SEED)


# ── 데이터 로드 & 전처리 ──────────────────────────────────────────────────────
def load_data(data_path: str, seq_len: int):
    df = pd.read_excel(data_path, sheet_name='merged')

    aec_cols = [f'aec_{i}' for i in range(seq_len)]
    X = np.array(df[aec_cols].values, dtype=float)   # (N, seq_len) — NaN 없음
    y = np.array(df['SMI'].values, dtype=float)       # (N,)

    # 변동성 없는 샘플 제외 (모든 AEC 값이 동일한 경우)
    valid = X.std(axis=1) > 0
    n_removed = (~valid).sum()
    if n_removed > 0:
        print(f"변동성 없는 샘플 {n_removed}개 제외 → 남은 샘플: {valid.sum()}")
    X, y = X[valid], y[valid]

    # 채널 축 추가: (N, 1, seq_len)
    X = X[:, np.newaxis, :]
    return X, y


class AECDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # (N, 1, seq_len)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)       # (N, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── 모델 ──────────────────────────────────────────────────────────────────────

class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),                # seq_len // 2

            # Block 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),                # seq_len // 4

            # Block 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),        # → (batch, 128, 8)
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.regressor(self.encoder(x))


# ── 학습 루프 ─────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, preds, trues = 0.0, [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        pred = model(X_batch)
        total_loss += criterion(pred, y_batch).item() * len(y_batch)
        preds.append(pred.cpu().numpy())
        trues.append(y_batch.cpu().numpy())
    preds = np.concatenate(preds).ravel()
    trues = np.concatenate(trues).ravel()
    return total_loss / len(loader.dataset), preds, trues


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 데이터
    X, y = load_data(DATA_PATH, SEQ_LEN)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"SMI 통계 — mean: {y.mean():.3f}, std: {y.std():.3f}, "
          f"min: {y.min():.3f}, max: {y.max():.3f}")

    # SMI 정규화 (입력 AEC 커브는 환자 내부에서 이미 상대적 스케일 유지)
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

    # Train / Val / Test split (7:1.5:1.5)
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y_scaled, test_size=0.15, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.15 / 0.85, random_state=SEED)

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    train_loader = DataLoader(AECDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(AECDataset(X_val,   y_val),
                              batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(AECDataset(X_test,  y_test),
                              batch_size=BATCH_SIZE, shuffle=False)

    # 모델
    model = CNN1D().to(device)
    print(f"\n모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5)

    best_val_loss = float('inf')
    best_state    = None

    print("\nEpoch | Train Loss | Val Loss")
    print("-" * 35)
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, _, _ = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            print(f"{epoch:5d} | {train_loss:.4f}     | {val_loss:.4f}")

    # 테스트 평가 (best checkpoint)
    assert best_state is not None
    model.load_state_dict(best_state)
    _, test_preds_scaled, test_trues_scaled = eval_epoch(
        model, test_loader, criterion, device)

    # 역변환
    test_preds = y_scaler.inverse_transform(test_preds_scaled.reshape(-1, 1)).ravel()
    test_trues = y_scaler.inverse_transform(test_trues_scaled.reshape(-1, 1)).ravel()

    mae = mean_absolute_error(test_trues, test_preds)
    r2  = r2_score(test_trues, test_preds)
    print(f"\n[Test] MAE: {mae:.4f}  R²: {r2:.4f}")

    # 모델 저장
    save_path = Path(__file__).parent / 'cnn1d_best.pt'
    torch.save({'model_state': best_state, 'y_scaler': y_scaler,
                'seq_len': SEQ_LEN}, save_path)
    print(f"모델 저장: {save_path}")


if __name__ == '__main__':
    main()
