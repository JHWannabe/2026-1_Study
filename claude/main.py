"""
OrangeX Auto-Trading — Main Entry Point

Modes:
  python main.py --mode fetch       Download historical OHLCV data
  python main.py --mode train       Train the ML model
  python main.py --mode backtest    Run backtest on historical data
  python main.py --mode paper       Run paper trading (real-time, virtual account)
"""

import argparse
import pandas as pd
from pathlib import Path

import config
from utils.logger import get_logger

log = get_logger("main")


def cmd_fetch(args):
    from api.data_fetcher import fetch_ohlcv
    from api.client import OrangeXClient

    client = OrangeXClient() if config.API_KEY else None

    for res in [args.resolution, args.sub_resolution]:
        df = fetch_ohlcv(
            instrument=args.instrument,
            resolution=res,
            days=args.days,
            client=client,
            use_cache=False,
        )
        if df.empty:
            log.error("No data fetched for resolution=%s. Check API credentials.", res)
        else:
            log.info("Fetched %d bars for %s (%smin)", len(df), args.instrument, res)


def cmd_train(args):
    from api.data_fetcher import fetch_ohlcv
    from api.client import OrangeXClient
    from model.trainer import train
    from features.multi_tf import build_multi_tf_features

    client = OrangeXClient() if config.API_KEY else None

    df_main = fetch_ohlcv(args.instrument, args.resolution,     args.days, client, use_cache=True)
    df_sub  = fetch_ohlcv(args.instrument, args.sub_resolution, args.days, client, use_cache=True)

    if df_main.empty:
        log.error("No main timeframe data available.")
        return

    if df_sub.empty:
        log.warning("No sub timeframe data — training on main TF only.")
        df_feat = df_main
    else:
        df_feat = build_multi_tf_features(df_main, df_sub)

    log.info("Training on %d bars with %d features.", len(df_feat), df_feat.shape[1])
    train(df_feat)


def cmd_backtest(args):
    from api.data_fetcher import fetch_ohlcv
    from api.client import OrangeXClient
    from model.trainer import build_features_and_labels, load_model, predict_signal
    from features.indicators import add_all_indicators
    from features.multi_tf import build_multi_tf_features
    from backtest.engine import run_backtest, plot_results

    client = OrangeXClient() if config.API_KEY else None

    df_main = fetch_ohlcv(args.instrument, args.resolution,     args.days, client, use_cache=True)
    df_sub  = fetch_ohlcv(args.instrument, args.sub_resolution, args.days, client, use_cache=True)

    if df_main.empty:
        log.error("No data for backtest.")
        return

    if df_sub.empty:
        df_feat = add_all_indicators(df_main.copy())
        df_feat.dropna(inplace=True)
    else:
        df_feat = build_multi_tf_features(df_main, df_sub)

    df = df_main  # price reference is always main TF

    model, scaler, feature_cols, label_map, label_map_inv = load_model()

    log.info("Generating signals for %d bars ...", len(df_feat))

    X = scaler.transform(df_feat[feature_cols].values)
    import numpy as np
    import lightgbm as lgb
    prob_long   = model.predict(X)                # shape (N,): prob of long
    prob_short  = 1.0 - prob_long
    confs       = np.maximum(prob_long, prob_short)
    signals_raw = np.where(prob_long >= 0.5, 1, -1).astype(int)
    signals_raw[confs < args.min_confidence] = 0  # below threshold → hold

    signals = pd.Series(signals_raw, index=df_feat.index, name="signal")

    log.info("Signal distribution: %s", dict(pd.Series(signals_raw).value_counts()))

    metrics = run_backtest(
        df=df.loc[df_feat.index],
        signals=signals,
        leverage=config.LEVERAGE,
        initial_capital=args.capital,
    )

    if args.plot and metrics:
        plot_results(metrics, save_path="backtest/results.png")


def cmd_paper(args):
    from trading.paper_trader import PaperTrader
    trader = PaperTrader(
        instrument    = args.instrument,
        resolution    = args.resolution,
        poll_seconds  = args.poll,
        min_confidence= args.min_confidence,
    )
    trader.run(duration_hours=args.hours)


def cmd_live(args):
    from trading.live_trader import LiveTrader
    from api.client import OrangeXClient

    # 잔고 조회
    balance_str = "조회 실패"
    try:
        client  = OrangeXClient()
        summary = client.get_account_summary(currency="USDT")
        if isinstance(summary, list):
            summary = summary[0] if summary else {}
        balance = float(
            summary.get("available_funds")
            or summary.get("available")
            or summary.get("wallet_balance")
            or summary.get("equity")
            or summary.get("total")
            or 0
        )
        balance_str = f"{balance:,.4f} USDT"
    except Exception as e:
        balance_str = f"조회 실패 ({e})"

    print("\n" + "=" * 55)
    print("  실계좌 자동매매를 시작합니다")
    print("=" * 55)
    print(f"  종목    : {args.instrument}")
    print(f"  레버리지: {config.LEVERAGE}x")
    print(f"  SL      : -{config.STOP_LOSS_PCT * config.LEVERAGE * 100:.0f}% (마진 기준)")
    print(f"  TP      : +{config.TAKE_PROFIT_PCT * config.LEVERAGE * 100:.0f}% (마진 기준)")
    print(f"  현재 시드: {balance_str}")
    print(f"  실행 시간: {args.hours}시간")
    if args.dry_run:
        print("  모드    : DRY-RUN (주문 미실행)")
    print("=" * 55)

    confirm = input("\n계속하려면 'yes' 입력: ").strip().lower()
    if confirm != "yes":
        print("취소됨.")
        return

    trader = LiveTrader(
        instrument     = args.instrument,
        resolution     = args.resolution,
        sub_resolution = args.sub_resolution,
        poll_seconds   = args.poll,
        min_confidence = args.min_confidence,
        dry_run        = args.dry_run,
    )
    trader.run(duration_hours=args.hours)


def main():
    parser = argparse.ArgumentParser(description="OrangeX Auto-Trader")
    parser.add_argument("--mode", choices=["fetch", "train", "backtest", "paper", "live"],
                        default="backtest")
    parser.add_argument("--instrument",       default=config.INSTRUMENT)
    parser.add_argument("--resolution",      default=config.RESOLUTION)
    parser.add_argument("--sub-resolution",  default=config.SUB_RESOLUTION,
                        dest="sub_resolution")
    parser.add_argument("--days",            type=int,   default=1000)
    parser.add_argument("--capital",         type=float, default=config.INITIAL_CAPITAL)
    parser.add_argument("--min-confidence",  type=float, default=0.7,
                        dest="min_confidence")
    parser.add_argument("--plot",            action="store_true", default=True)
    parser.add_argument("--hours",           type=float, default=24,
                        help="Duration for paper trading (hours)")
    parser.add_argument("--poll",            type=int,   default=1,
                        help="Poll interval for paper/live trading (seconds)")
    parser.add_argument("--dry-run",         action="store_true", default=False,
                        dest="dry_run",
                        help="Live 모드에서 신호만 확인하고 실제 주문은 내지 않음")
    args = parser.parse_args()

    log.info("Mode: %s | Instrument: %s | Resolution: %s",
             args.mode, args.instrument, args.resolution)

    dispatch = {
        "fetch":    cmd_fetch,
        "train":    cmd_train,
        "backtest": cmd_backtest,
        "paper":    cmd_paper,
        "live":     cmd_live,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
