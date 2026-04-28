"""
Model training pipeline.

Label: 1 (long) / -1 (short) / 0 (hold) based on future N-bar return.
Model: LightGBM gradient-boosted trees.

Train/test strategy
-------------------
- Chronological split with PREDICT_HORIZON-bar gap at each boundary to
  prevent label leakage (the label at row i uses future bars i..i+H,
  so the last H rows of a training window "see into" the next window).
- Walk-forward cross-validation (expanding window) is run before final
  training so that reported metrics are not from a single lucky split.
"""

import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import lightgbm as lgb

from features.indicators import add_all_indicators
from utils.logger import get_logger
import config

log = get_logger(__name__)

FEATURE_COLS: list[str] = []   # populated at build time
LABEL_COL = "label"

# Binary: short=0, long=1  (hold rows are dropped before training)
LABEL_MAP     = {-1: 0, 1: 1}
LABEL_MAP_INV = {0: -1, 1: 1}


# ─── Feature / Label construction ────────────────────────────────────────────

def build_features_and_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add indicators + forward-return label to the raw OHLCV DataFrame."""
    df = add_all_indicators(df)

    # Future return over PREDICT_HORIZON bars
    future_ret = df["close"].pct_change(config.PREDICT_HORIZON).shift(-config.PREDICT_HORIZON)

    thresh = config.PROFIT_THRESHOLD
    df[LABEL_COL] = 0
    df.loc[future_ret >  thresh, LABEL_COL] =  1    # long signal
    df.loc[future_ret < -thresh, LABEL_COL] = -1    # short signal

    df.dropna(inplace=True)

    # Drop hold rows — binary classifier trains on long/short only
    df = df[df[LABEL_COL] != 0].copy()
    return df


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    skip = {"open", "high", "low", "close", "volume", LABEL_COL}
    return [c for c in df.columns if c not in skip]


# ─── LightGBM params ─────────────────────────────────────────────────────────

def _get_lgbm_params() -> dict:
    return {
        "objective":         "binary",
        "metric":            "binary_logloss",
        "learning_rate":     0.05,
        "num_leaves":        63,
        "max_depth":         -1,
        "min_child_samples": 50,
        "feature_fraction":  0.8,
        "bagging_fraction":  0.8,
        "bagging_freq":      5,
        "reg_alpha":         0.1,
        "reg_lambda":        0.1,
        "verbosity":         -1,
    }


# ─── Train / Val / Test split ─────────────────────────────────────────────────

def split_data(df: pd.DataFrame):
    """
    Chronological train / validation / test split.

    A gap of PREDICT_HORIZON bars is inserted at each boundary so that
    the label of the last training row (which looks PREDICT_HORIZON bars
    ahead) does not overlap with the first row of the next window.
    """
    gap = config.PREDICT_HORIZON
    n     = len(df)
    i_tr  = int(n * config.TRAIN_RATIO)
    i_val = int(n * (config.TRAIN_RATIO + config.VALID_RATIO))

    feature_cols = _get_feature_cols(df)
    X = df[feature_cols].values
    y = df[LABEL_COL].values

    return (
        X[:i_tr],               y[:i_tr],           # train
        X[i_tr  + gap:i_val],   y[i_tr  + gap:i_val],  # val   (gap from train)
        X[i_val + gap:],        y[i_val + gap:],    # test  (gap from val)
        feature_cols,
    )


# ─── Walk-forward cross-validation ───────────────────────────────────────────

def walk_forward_cv(
    df: pd.DataFrame,
    n_splits: int = 5,
    min_train_ratio: float = 0.50,
) -> list[dict]:
    """
    Walk-forward cross-validation with an expanding training window.

    Timeline per fold (gap = PREDICT_HORIZON):

        |<────── train ──────>|  gap  |<── test ──>|
                              └─label leakage zone──┘

    Parameters
    ----------
    df              : feature+label DataFrame (output of build_features_and_labels)
    n_splits        : number of test folds
    min_train_ratio : fraction of data used as training in the first fold

    Returns
    -------
    List of per-fold result dicts with keys:
        fold, train_size, test_start, test_end, accuracy, f1_macro
    """
    gap          = config.PREDICT_HORIZON
    feature_cols = _get_feature_cols(df)
    X            = df[feature_cols].values
    y            = df[LABEL_COL].values
    n            = len(X)

    min_train = int(n * min_train_ratio)
    # split the remaining data into n_splits equal-sized test windows
    remaining  = n - min_train
    test_size  = remaining // (n_splits + 1)   # +1 so the last fold still has room

    if test_size < 50:
        log.warning("walk_forward_cv: test window too small (%d bars). "
                    "Reduce n_splits or collect more data.", test_size)

    results = []
    params  = _get_lgbm_params()

    for fold in range(n_splits):
        train_end  = min_train + fold * test_size
        test_start = train_end + gap
        test_end   = test_start + test_size

        if test_end > n:
            log.warning("Fold %d: not enough data, stopping CV early.", fold + 1)
            break

        X_tr, y_tr = X[:train_end],          y[:train_end]
        X_te, y_te = X[test_start:test_end], y[test_start:test_end]

        # Scaler fit only on the current training window
        scaler   = RobustScaler()
        X_tr_s   = scaler.fit_transform(X_tr)
        X_te_s   = scaler.transform(X_te)

        y_tr_ = np.vectorize(LABEL_MAP.get)(y_tr)
        y_te_ = np.vectorize(LABEL_MAP.get)(y_te)

        # Fixed rounds — no early stopping in CV (no separate val set per fold)
        ds_tr = lgb.Dataset(X_tr_s, label=y_tr_)
        model = lgb.train(params, ds_tr, num_boost_round=500)

        probs  = model.predict(X_te_s)          # shape (N,): prob of long (class 1)
        preds_ = (probs >= 0.5).astype(int)
        preds  = np.vectorize(LABEL_MAP_INV.get)(preds_)
        y_true = np.vectorize(LABEL_MAP_INV.get)(y_te_)

        acc = accuracy_score(y_true, preds)
        f1  = f1_score(y_true, preds, average="macro", zero_division=0)

        fold_result = {
            "fold":       fold + 1,
            "train_size": train_end,
            "test_start": test_start,
            "test_end":   test_end,
            "accuracy":   round(acc, 4),
            "f1_macro":   round(f1,  4),
        }
        results.append(fold_result)
        log.info(
            "CV Fold %d/%d  train=%d bars  test=[%d:%d]  acc=%.4f  f1=%.4f",
            fold + 1, n_splits, train_end, test_start, test_end, acc, f1,
        )

    if results:
        avg_acc = np.mean([r["accuracy"] for r in results])
        avg_f1  = np.mean([r["f1_macro"] for r in results])
        log.info("Walk-forward CV summary  avg_acc=%.4f  avg_f1=%.4f", avg_acc, avg_f1)
        _print_cv_summary(results)

    return results


def _print_cv_summary(results: list[dict]):
    print("\n" + "=" * 55)
    print("  WALK-FORWARD CV RESULTS")
    print("=" * 55)
    print(f"  {'Fold':>4}  {'Train bars':>10}  {'Test window':>16}  {'Acc':>7}  {'F1(macro)':>9}")
    print("  " + "-" * 51)
    for r in results:
        print(f"  {r['fold']:>4}  {r['train_size']:>10}  "
              f"[{r['test_start']:>6}:{r['test_end']:>6}]  "
              f"{r['accuracy']:>7.4f}  {r['f1_macro']:>9.4f}")
    accs = [r["accuracy"] for r in results]
    f1s  = [r["f1_macro"] for r in results]
    print("  " + "-" * 51)
    print(f"  {'AVG':>4}  {'':>10}  {'':>16}  {np.mean(accs):>7.4f}  {np.mean(f1s):>9.4f}")
    print(f"  {'STD':>4}  {'':>10}  {'':>16}  {np.std(accs):>7.4f}  {np.std(f1s):>9.4f}")
    print("=" * 55 + "\n")


# ─── Main training entry point ────────────────────────────────────────────────

def train(df: pd.DataFrame, run_cv: bool = True, cv_splits: int = 5) -> tuple:
    """
    Train LightGBM classifier.

    Parameters
    ----------
    df        : raw OHLCV DataFrame
    run_cv    : whether to run walk-forward CV before final training
    cv_splits : number of CV folds

    Returns
    -------
    (model, scaler, feature_cols, label_map, label_map_inv)
    """
    log.info("Building features & labels …")
    df = build_features_and_labels(df)
    log.info("Dataset size: %d rows, label dist: %s",
             len(df), dict(pd.Series(df[LABEL_COL]).value_counts()))

    # ── Walk-forward CV ───────────────────────────────────────────────────────
    if run_cv:
        log.info("Running walk-forward cross-validation (%d folds) …", cv_splits)
        walk_forward_cv(df, n_splits=cv_splits)

    # ── Final train on full train split ──────────────────────────────────────
    log.info("Training final model on train/val/test split …")
    X_tr, y_tr, X_val, y_val, X_te, y_te, feature_cols = split_data(df)

    log.info("Split sizes — train: %d  val: %d  test: %d  (gap=%d bars each)",
             len(X_tr), len(X_val), len(X_te), config.PREDICT_HORIZON)

    scaler = RobustScaler()
    X_tr  = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    X_te  = scaler.transform(X_te)

    y_tr_  = np.vectorize(LABEL_MAP.get)(y_tr)
    y_val_ = np.vectorize(LABEL_MAP.get)(y_val)
    y_te_  = np.vectorize(LABEL_MAP.get)(y_te)

    ds_tr  = lgb.Dataset(X_tr,  label=y_tr_)
    ds_val = lgb.Dataset(X_val, label=y_val_, reference=ds_tr)

    callbacks = [
        lgb.early_stopping(50, verbose=False),
        lgb.log_evaluation(100),
    ]

    log.info("Training LightGBM …")
    model = lgb.train(
        _get_lgbm_params(),
        ds_tr,
        num_boost_round=2000,
        valid_sets=[ds_val],
        callbacks=callbacks,
    )

    # ── Evaluate on held-out test set ─────────────────────────────────────────
    probs    = model.predict(X_te)              # shape (N,): prob of long
    preds_   = (probs >= 0.5).astype(int)
    preds    = np.vectorize(LABEL_MAP_INV.get)(preds_)
    y_te_raw = np.vectorize(LABEL_MAP_INV.get)(y_te_)

    log.info("\n%s", classification_report(y_te_raw, preds,
                                           target_names=["short", "long"]))
    cm = confusion_matrix(y_te_raw, preds)
    log.info("Confusion matrix:\n%s", cm)

    # Feature importance
    importance = pd.Series(model.feature_importance(importance_type="gain"),
                           index=feature_cols).sort_values(ascending=False)
    log.info("Top 15 features:\n%s", importance.head(15).to_string())

    _save(model, scaler, feature_cols)
    return model, scaler, feature_cols, LABEL_MAP, LABEL_MAP_INV


# ─── Persist / Load ───────────────────────────────────────────────────────────

def _save(model, scaler, feature_cols):
    out = Path(config.MODEL_DIR)
    out.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out / "lgbm.txt"))
    joblib.dump(scaler,       out / "scaler.pkl")
    joblib.dump(feature_cols, out / "feature_cols.pkl")
    joblib.dump(LABEL_MAP,     out / "label_map.pkl")
    joblib.dump(LABEL_MAP_INV, out / "label_map_inv.pkl")
    log.info("Model saved to %s", out)


def load_model():
    """Load saved model artifacts."""
    out = Path(config.MODEL_DIR)
    model        = lgb.Booster(model_file=str(out / "lgbm.txt"))
    scaler       = joblib.load(out / "scaler.pkl")
    feature_cols = joblib.load(out / "feature_cols.pkl")
    label_map    = joblib.load(out / "label_map.pkl")
    label_map_inv= joblib.load(out / "label_map_inv.pkl")
    return model, scaler, feature_cols, label_map, label_map_inv


# ─── Inference ────────────────────────────────────────────────────────────────

def predict_signal(df_latest: pd.DataFrame, model, scaler, feature_cols,
                   label_map_inv, min_confidence: float = 0.50) -> tuple[int, float, dict]:
    """
    Given a recent OHLCV window, predict the trading signal.
    Returns (signal, confidence, all_probs)
      signal     ∈ {-1, 0, 1}
      confidence : probability of the winning class
      all_probs  : {"short": p, "hold": p, "long": p}
    """
    from features.indicators import add_all_indicators
    df_feat = add_all_indicators(df_latest)
    if df_feat.empty:
        return 0, 0.0, {"short": 0.0, "hold": 0.0, "long": 0.0}

    X = scaler.transform(df_feat[feature_cols].iloc[[-1]].values)
    prob_long  = float(model.predict(X)[0])     # scalar: prob of long
    prob_short = 1.0 - prob_long

    signal     = 1 if prob_long >= 0.5 else -1
    confidence = max(prob_long, prob_short)

    if confidence < min_confidence:
        signal = 0

    all_probs = {
        "short": round(prob_short, 4),
        "long":  round(prob_long,  4),
    }
    return signal, confidence, all_probs
