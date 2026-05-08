"""ResNet1D 공통 훈련/평가 모듈 — 성별(PatientSex)별 임계값 적용 버전"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (
    mean_absolute_error, r2_score, roc_curve, auc, confusion_matrix,
    average_precision_score, precision_recall_curve, brier_score_loss,
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BATCH_SIZE = 32
EPOCHS     = 500
LR         = 1e-3
SEED       = 42

# (이름, scale_X, scale_y) 조합 — 모두 학습·비교
SCALE_CONFIGS = [
    ('noScale', False, False),
    ('scaleX',  True,  False),
    ('scaleY',  False, True),
    ('scaleXY', True,  True),
]

torch.manual_seed(SEED)
np.random.seed(SEED)


class _TabDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


def _loader(X, y, shuffle):
    return DataLoader(_TabDataset(X, y), batch_size=BATCH_SIZE, shuffle=shuffle)


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = (
            nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                          nn.BatchNorm1d(out_ch))
            if stride != 1 or in_ch != out_ch else None
        )

    def forward(self, x):
        identity = self.downsample(x) if self.downsample else x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet1D(nn.Module):
    def __init__(self, in_ch=1, base=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base), nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        self.layers = nn.Sequential(
            self._stage(base,   base,   2, stride=1),
            self._stage(base,   base*2, 2, stride=2),
            self._stage(base*2, base*4, 2, stride=2),
            self._stage(base*4, base*8, 2, stride=2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(base*8, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(256, 1),
        )

    @staticmethod
    def _stage(in_ch, out_ch, n, stride):
        return nn.Sequential(ResBlock1D(in_ch, out_ch, stride),
                             *[ResBlock1D(out_ch, out_ch) for _ in range(n - 1)])

    def forward(self, x):
        return self.head(self.layers(self.stem(x)))


def _train_epoch(model, loader, optimizer, criterion, device):
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
def _eval_epoch(model, loader, criterion, device):
    model.eval()
    total, preds, trues = 0.0, [], []
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        p = model(X_b)
        total += criterion(p, y_b).item() * len(y_b)
        preds.append(p.cpu().numpy())
        trues.append(y_b.cpu().numpy())
    return total / len(loader.dataset), np.concatenate(preds).ravel(), np.concatenate(trues).ravel()


def _fit(tr_loader, vl_loader, model, device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    best_loss, best_state = float('inf'), None
    for _ in range(EPOCHS):
        _train_epoch(model, tr_loader, optimizer, criterion, device)
        vl, _, _ = _eval_epoch(model, vl_loader, criterion, device)
        scheduler.step()
        if vl < best_loss:
            best_loss  = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    return best_state, criterion


def _get_gender_thresholds(trues, sex_arr):
    """성별별로 하위 25% 임계값을 개별 산출하여 샘플별 임계값 배열 반환"""
    # PatientSex: 0(남성), 1(여성) 가정
    m_mask = (sex_arr == 0)
    f_mask = (sex_arr == 1)
    
    # 기본값 (데이터 부족 시 대비)
    default_thr = np.percentile(trues, 25) if len(trues) > 0 else 0
    
    thr_m = np.percentile(trues[m_mask], 25) if np.any(m_mask) else default_thr
    thr_f = np.percentile(trues[f_mask], 25) if np.any(f_mask) else default_thr
    
    # 각 샘플의 성별에 맞는 임계값 배열 생성
    sample_thrs = np.where(sex_arr == 0, thr_m, thr_f)
    return thr_m, thr_f, sample_thrs


def _save_md(fold_maes, fold_r2s, test_trues, test_preds, target, results_dir, label,
             sex_arr, fold_accs=None, fold_aucs=None,
             fold_feat_lists=None, final_feat_list=None,
             report_name='ResNet1D_Report.md'):
    maes  = np.array(fold_maes)
    r2s   = np.array(fold_r2s)
    trues = np.array(test_trues)
    preds = np.array(test_preds)
    test_r2  = r2_score(trues, preds)
    test_mae = mean_absolute_error(trues, preds)

    # 성별별 임계값 적용
    thr_m, thr_f, sample_thrs = _get_gender_thresholds(trues, sex_arr)
    y_bin = (trues >= sample_thrs).astype(int)
    y_pred_bin = (preds >= sample_thrs).astype(int)
    test_acc = (y_pred_bin == y_bin).mean()

    # ROC/PR 분석용 스코어 정규화
    score_norm = (preds - preds.min()) / (np.ptp(preds) + 1e-8)
    fpr, tpr, _ = roc_curve(y_bin, score_norm)
    roc_auc  = auc(fpr, tpr)
    test_auprc = average_precision_score(y_bin, score_norm)
    test_brier = brier_score_loss(y_bin, score_norm)

    # 통계 분석
    r_val, p_r = stats.pearsonr(trues, preds)
    resid = preds - trues
    p_sw = stats.shapiro(resid)[1] if len(resid) <= 5000 else float('nan')
    p_sw_lbl = f'{p_sw:.4f}' if not np.isnan(p_sw) else 'N/A'
    _, p_bt = stats.ttest_1samp(resid, 0)

    has_feat = fold_feat_lists is not None
    has_clf  = fold_accs is not None and fold_aucs is not None
    fold_feat_counts = [len(f) for f in fold_feat_lists] if has_feat else None

    # 헤더/구분선 동적 생성
    hdr_cols = ['Fold', 'MAE', 'R²']
    if has_clf:
        hdr_cols += ['ACC', 'AUC']
    if has_feat:
        hdr_cols += ['피처 수']
    fold_hdr = '| ' + ' | '.join(hdr_cols) + ' |'
    fold_sep = '|' + '|'.join(['---'] * len(hdr_cols)) + '|'

    lines = [
        f'# ResNet1D Report — {target}',
        '',
        f'**Model:** {label}',
        '',
        '## 5-Fold CV 성능 (Train 80%)',
        '',
        fold_hdr,
        fold_sep,
    ]
    for i, (mae, r2) in enumerate(zip(fold_maes, fold_r2s), 1):
        row = f'| {i} | {mae:.4f} | {r2:.4f}'
        if has_clf:
            acc_v = fold_accs[i - 1]
            auc_v = fold_aucs[i - 1]
            acc_s = f'{acc_v:.4f}' if not np.isnan(acc_v) else 'N/A'
            auc_s = f'{auc_v:.4f}' if not np.isnan(auc_v) else 'N/A'
            row += f' | {acc_s} | {auc_s}'
        if has_feat:
            row += f' | {fold_feat_counts[i - 1]}'
        lines.append(row + ' |')

    accs = np.array(fold_accs) if has_clf else None
    aucs = np.array(fold_aucs) if has_clf else None
    mean_row = f'| **Mean** | **{maes.mean():.4f}** | **{r2s.mean():.4f}**'
    std_row  = f'| **Std** | **{maes.std():.4f}** | **{r2s.std():.4f}**'
    if has_clf:
        mean_row += f' | **{np.nanmean(accs):.4f}** | **{np.nanmean(aucs):.4f}**'
        std_row  += f' | **{np.nanstd(accs):.4f}** | **{np.nanstd(aucs):.4f}**'
    if has_feat:
        mean_row += f' | **{np.mean(fold_feat_counts):.1f}**'
        std_row  += ' |'
    lines += [mean_row + ' |', std_row + ' |', '']

    lines += [
        '## Test Set 성능 (Test 20%)',
        '',
        f'Test R² = **{test_r2:.4f}**',
        f'Test MAE = **{test_mae:.4f}**',
        '',
        '## 통계 분석 (성별별 임계값 적용)',
        '',
        '| 지표 | 값 |',
        '|---|---|',
        f'| 남성 임계값 (25th pct) | {thr_m:.2f} |',
        f'| 여성 임계값 (25th pct) | {thr_f:.2f} |',
        f'| Pearson r | {r_val:.4f} (p={p_r:.3e}) |',
        f'| Shapiro-Wilk p | {p_sw_lbl} |',
        f'| Bias t-test p | {p_bt:.4f} |',
        f'| 이진화 ACC (성별 기준) | {test_acc:.4f} |',
        f'| 이진화 AUC (성별 기준) | {roc_auc:.4f} |',
        f'| 이진화 AUPRC (성별 기준) | {test_auprc:.4f} |',
        f'| 이진화 Brier Score (성별 기준) | {test_brier:.4f} |',
    ]

    # 피처 선택 목록 저장
    if has_feat:
        lines += ['', '## 피처 선택 목록 (Fold별)', '']
        for i, feats in enumerate(fold_feat_lists, 1):
            lines += [
                f'### Fold {i} ({len(feats)}개)',
                '',
                ', '.join(feats),
                '',
            ]
        if final_feat_list is not None:
            lines += [
                f'### 최종 모델 (Train 전체, {len(final_feat_list)}개)',
                '',
                ', '.join(final_feat_list),
                '',
            ]

    os.makedirs(os.path.join(results_dir, target), exist_ok=True)
    path = os.path.join(results_dir, target, report_name)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  → ResNet1D MD 저장: {path}')


def _plot_oof(all_trues, all_preds, target, results_dir, label, sex_arr,
              plot_name='ResNet1D_Results.png'):
    trues = np.array(all_trues)
    preds = np.array(all_preds)
    resid = preds - trues

    # 성별별 임계값 계산 및 이진화
    thr_m, thr_f, sample_thrs = _get_gender_thresholds(trues, sex_arr)
    y_bin = (trues >= sample_thrs).astype(int)
    preds_bin = (preds >= sample_thrs).astype(int)

    r_val, p_r  = stats.pearsonr(trues, preds)
    p_sw = stats.shapiro(resid)[1] if len(resid) <= 5000 else float('nan')
    p_sw_lbl    = f'{p_sw:.3f}' if not np.isnan(p_sw) else 'N/A'
    _, p_bt     = stats.ttest_1samp(resid, 0)

    score_norm = (preds - preds.min()) / (np.ptp(preds) + 1e-8)
    fpr, tpr, _ = roc_curve(y_bin, score_norm)
    roc_auc    = auc(fpr, tpr)
    auprc_val  = average_precision_score(y_bin, score_norm)
    cm         = confusion_matrix(y_bin, preds_bin)

    fig = plt.figure(figsize=(24, 11))
    gs  = gridspec.GridSpec(2, 4, hspace=0.45, wspace=0.38)
    ax  = [fig.add_subplot(gs[ri, c]) for ri in range(2) for c in range(4)]

    # 1. Predicted vs True
    lo, hi = min(trues.min(), preds.min()), max(trues.max(), preds.max())
    ax[0].scatter(trues, preds, alpha=0.6, edgecolors='k', linewidths=0.4)
    ax[0].plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='y = x')
    ax[0].set(xlabel=f'True {target}', ylabel=f'Predicted {target}',
              title=f'Predicted vs True\nr={r_val:.3f}  p={p_r:.2e}  R²={r2_score(trues, preds):.3f}')
    ax[0].legend()

    # 2. Residuals
    ax[1].hist(resid, bins=20, color='steelblue', edgecolor='white', alpha=0.85)
    ax[1].axvline(0, color='r', linestyle='--', lw=1.5)
    ax[1].set(xlabel='Residual (Pred−True)', ylabel='Count',
              title=f'Residual Distribution\nShapiro-Wilk p={p_sw_lbl}  |  Bias t-test p={p_bt:.3f}')

    # 3. Bland-Altman
    mean_v = (trues + preds) / 2
    diff_v = preds - trues
    md, sd = diff_v.mean(), diff_v.std()
    ax[2].scatter(mean_v, diff_v, alpha=0.6, edgecolors='k', linewidths=0.4)
    ax[2].axhline(md,           color='r',    lw=1.5, label=f'Mean {md:+.3f}')
    ax[2].axhline(md + 1.96*sd, color='gray', lw=1.2, ls='--', label=f'+1.96SD {md+1.96*sd:+.3f}')
    ax[2].axhline(md - 1.96*sd, color='gray', lw=1.2, ls='--', label=f'−1.96SD {md-1.96*sd:+.3f}')
    ax[2].set(xlabel='Mean(True, Pred)', ylabel='Pred−True', title='Bland-Altman Plot')
    ax[2].legend(fontsize=8)

    # 4. ROC Curve (성별 기준 명시)
    ax[3].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC={roc_auc:.3f}')
    ax[3].plot([0, 1], [0, 1], 'k--', lw=1)
    ax[3].set(xlabel='FPR', ylabel='TPR',
              title=f'ROC Curve\n(Gender-specific 25th pct: M={thr_m:.1f}, F={thr_f:.1f})')
    ax[3].legend()

    # 5. Confusion Matrix
    im = ax[4].imshow(cm, cmap='Blues')
    fig.colorbar(im, ax=ax[4])
    cls = ['Low (Sarcopenia)', 'Normal']
    ax[4].set_xticks([0, 1]); ax[4].set_xticklabels(cls, fontsize=9)
    ax[4].set_yticks([0, 1]); ax[4].set_yticklabels(cls, fontsize=9)
    ax[4].set(xlabel='Predicted', ylabel='True', title='Confusion Matrix\n(Gender-specific Threshold)')
    for i in range(2):
        for j in range(2):
            ax[4].text(j, i, str(cm[i, j]), ha='center', va='center',
                       color='black', fontsize=12)

    # 6. Q-Q Plot
    (osm, osr), (slope, intercept, _) = stats.probplot(resid)
    ax[5].scatter(osm, osr, alpha=0.6, edgecolors='k', linewidths=0.4)
    qq = np.array([osm[0], osm[-1]])
    ax[5].plot(qq, slope * qq + intercept, 'r--', lw=1.5)
    ax[5].set(xlabel='Theoretical Quantiles', ylabel='Sample Quantiles',
              title=f'Q-Q Plot (Shapiro-Wilk p={p_sw_lbl})')

    # 7. PR Curve
    precision_arr, recall_arr, _ = precision_recall_curve(y_bin, score_norm)
    baseline_pr = float(y_bin.mean())
    ax[6].plot(recall_arr, precision_arr, color='darkorange', lw=2, label=f'AUPRC={auprc_val:.3f}')
    ax[6].axhline(baseline_pr, color='r', linestyle='--', lw=1, label=f'Baseline ({baseline_pr:.3f})')
    ax[6].set(xlabel='Recall', ylabel='Precision',
              title=f'PR Curve\n(Gender-specific 25th pct: M={thr_m:.1f}, F={thr_f:.1f})')
    ax[6].legend(fontsize=8)

    # 8. Calibration Plot
    prob_true_cal, prob_pred_cal = calibration_curve(y_bin, score_norm, n_bins=10)
    ax[7].plot(prob_pred_cal, prob_true_cal, 's-', label='Model')
    ax[7].plot([0, 1], [0, 1], 'r--', lw=1.5, label='Perfect')
    ax[7].set(xlabel='Mean Predicted Probability', ylabel='Fraction of Positives',
              title='Calibration Plot')
    ax[7].legend(fontsize=8)

    fig.suptitle(f'{label} — Test Set Evaluation ({target})',
                 fontsize=14, fontweight='bold', y=1.01)
    os.makedirs(os.path.join(results_dir, target), exist_ok=True)
    out = os.path.join(results_dir, target, plot_name)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → 시각화 저장: {out}')


def run_resnet1d(df, y_raw: np.ndarray,
                 target: str, results_dir: str,
                 label: str,
                 feature_selector) -> None:
    """
    SCALE_CONFIGS 4가지 조합(noScale/scaleX/scaleY/scaleXY)을 모두 학습·평가.
    피처 선택은 스케일링과 무관하므로 fold당 1회만 수행해 공유.
    noScale 결과는 기존 파일명(ResNet1D_Report.md) 유지.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tr_idx_all, te_idx_all = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=SEED
    )
    df_train_all = df.iloc[tr_idx_all].reset_index(drop=True)
    df_test_all  = df.iloc[te_idx_all].reset_index(drop=True)
    y_train_all  = y_raw[tr_idx_all]
    y_test_all   = y_raw[te_idx_all]
    sex_test     = df_test_all['PatientSex'].values

    # 피처 선택: 스케일링 무관 → fold당 1회 미리 계산
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_splits = list(kf.split(df_train_all))
    fold_feat_lists = []
    for fold_i, (tr_idx, vl_idx) in enumerate(fold_splits, 1):
        sel = feature_selector(df_train_all.iloc[tr_idx])
        fold_feat_lists.append(list(sel))
        print(f'  ── [피처선택] Fold {fold_i}/5 → {len(sel)}개 ──')
    sel_feats_final = feature_selector(df_train_all)

    for scale_name, scale_X, scale_y in SCALE_CONFIGS:
        print(f'\n  ══ [ResNet1D | {scale_name}] scale_X={scale_X}, scale_y={scale_y} ══')
        fold_maes, fold_r2s, fold_accs, fold_aucs = [], [], [], []

        for fold_i, (tr_idx, vl_idx) in enumerate(fold_splits, 1):
            df_fold_tr = df_train_all.iloc[tr_idx]
            df_fold_vl = df_train_all.iloc[vl_idx]
            sel_feats  = fold_feat_lists[fold_i - 1]
            y_fold_tr  = y_train_all[tr_idx]
            y_fold_vl  = y_train_all[vl_idx]
            print(f'    Fold {fold_i}/5 (train {len(tr_idx)}, val {len(vl_idx)}, feats {len(sel_feats)})')

            X_tr_raw = df_fold_tr[sel_feats].values.astype(float)
            X_vl_raw = df_fold_vl[sel_feats].values.astype(float)

            if scale_X:
                sc_X = StandardScaler().fit(X_tr_raw)
                X_tr_raw = sc_X.transform(X_tr_raw)
                X_vl_raw = sc_X.transform(X_vl_raw)

            y_tr_s = y_fold_tr.copy()
            y_vl_s = y_fold_vl.copy()
            sc_y = None
            if scale_y:
                sc_y = StandardScaler().fit(y_fold_tr.reshape(-1, 1))
                y_tr_s = sc_y.transform(y_fold_tr.reshape(-1, 1)).ravel()
                y_vl_s = sc_y.transform(y_fold_vl.reshape(-1, 1)).ravel()

            X_tr = X_tr_raw[:, np.newaxis, :]
            X_vl = X_vl_raw[:, np.newaxis, :]

            model = ResNet1D().to(device)
            best_state, criterion = _fit(
                _loader(X_tr, y_tr_s, True),
                _loader(X_vl, y_vl_s, False),
                model, device,
            )
            model.load_state_dict(best_state)
            _, ps, ts = _eval_epoch(model, _loader(X_vl, y_vl_s, False), criterion, device)

            if sc_y is not None:
                ps = sc_y.inverse_transform(ps.reshape(-1, 1)).ravel()
                ts = y_fold_vl

            fold_maes.append(mean_absolute_error(ts, ps))
            fold_r2s.append(r2_score(ts, ps))

            sex_vl_val = df_fold_vl['PatientSex'].values
            thr_m_f, thr_f_f, _ = _get_gender_thresholds(
                y_train_all[tr_idx], df_train_all['PatientSex'].values[tr_idx]
            )
            vl_sample_thrs = np.where(sex_vl_val == 0, thr_m_f, thr_f_f)
            vl_bin      = (ts >= vl_sample_thrs).astype(int)
            vl_score    = (ps - ps.min()) / (np.ptp(ps) + 1e-8)
            vl_pred_bin = (ps >= vl_sample_thrs).astype(int)
            fold_accs.append((vl_pred_bin == vl_bin).mean())
            try:
                from sklearn.metrics import roc_auc_score as _roc_auc
                fold_aucs.append(_roc_auc(vl_bin, vl_score))
            except Exception:
                fold_aucs.append(float('nan'))

        # 최종 모델: train 전체로 학습 → test 평가
        X_tr_raw = df_train_all[sel_feats_final].values.astype(float)
        X_te_raw = df_test_all[sel_feats_final].values.astype(float)

        if scale_X:
            sc_X_final = StandardScaler().fit(X_tr_raw)
            X_tr_raw = sc_X_final.transform(X_tr_raw)
            X_te_raw = sc_X_final.transform(X_te_raw)

        y_tr_s = y_train_all.copy()
        y_te_s = y_test_all.copy()
        sc_y_final = None
        if scale_y:
            sc_y_final = StandardScaler().fit(y_train_all.reshape(-1, 1))
            y_tr_s = sc_y_final.transform(y_train_all.reshape(-1, 1)).ravel()
            y_te_s = sc_y_final.transform(y_test_all.reshape(-1, 1)).ravel()

        X_tr_final = X_tr_raw[:, np.newaxis, :]
        X_te_final = X_te_raw[:, np.newaxis, :]

        model_final = ResNet1D().to(device)
        best_state_final, criterion_final = _fit(
            _loader(X_tr_final, y_tr_s, True),
            _loader(X_te_final, y_te_s, False),
            model_final, device,
        )
        model_final.load_state_dict(best_state_final)
        _, ps_te, ts_te = _eval_epoch(
            model_final, _loader(X_te_final, y_te_s, False), criterion_final, device
        )

        if sc_y_final is not None:
            ps_te = sc_y_final.inverse_transform(ps_te.reshape(-1, 1)).ravel()
            ts_te = y_test_all

        # noScale은 기존 파일명 유지 (5_generate_plots.py 호환)
        if scale_name == 'noScale':
            report_name = 'ResNet1D_Report.md'
            plot_name   = 'ResNet1D_Results.png'
        else:
            report_name = f'ResNet1D_{scale_name}_Report.md'
            plot_name   = f'ResNet1D_{scale_name}_Results.png'

        _save_md(fold_maes, fold_r2s, ts_te, ps_te, target, results_dir,
                 f'{label}_{scale_name}',
                 sex_arr=sex_test,
                 fold_accs=fold_accs, fold_aucs=fold_aucs,
                 fold_feat_lists=fold_feat_lists,
                 final_feat_list=list(sel_feats_final),
                 report_name=report_name)
        _plot_oof(ts_te.tolist(), ps_te.tolist(), target, results_dir,
                  f'{label}_{scale_name}',
                  sex_arr=sex_test, plot_name=plot_name)