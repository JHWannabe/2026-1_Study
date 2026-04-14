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

    client   = OrangeXClient() if config.API_KEY else None
    end_date = args.end_date or config.TRAIN_END_DATE

    df = fetch_ohlcv(
        instrument=args.instrument,
        resolution=args.resolution,
        days=args.days,
        client=client,
        use_cache=False,
        end_date=end_date,
    )
    if df.empty:
        log.error("No data fetched for resolution=%s. Check API credentials.", args.resolution)
    else:
        log.info("Fetched %d bars for %s (%smin)  end=%s",
                 len(df), args.instrument, args.resolution, end_date)


def cmd_train(args):
    from api.data_fetcher import fetch_ohlcv
    from api.client import OrangeXClient
    from model.trainer import train

    client   = OrangeXClient() if config.API_KEY else None
    end_date = args.end_date or config.TRAIN_END_DATE

    df_main = fetch_ohlcv(
        args.instrument, args.resolution, args.days,
        client, use_cache=True, end_date=end_date,
    )

    if df_main.empty:
        log.error("No main timeframe data available.")
        return

    log.info("Training on %d bars  end=%s", len(df_main), end_date)
    train(df_main)


def cmd_backtest(args):
    from api.data_fetcher import fetch_ohlcv
    from api.client import OrangeXClient
    from features.bb_signals import generate_bb_signals
    from backtest.engine import run_backtest, plot_results

    client   = OrangeXClient() if config.API_KEY else None
    end_date = args.end_date or config.TRAIN_END_DATE

    df_main = fetch_ohlcv(
        args.instrument, args.resolution, args.days,
        client, use_cache=True, end_date=end_date,
    )

    if df_main.empty:
        log.error("No data for backtest.")
        return

    log.info("BB 신호 생성 중 (%d 캔들, 기간=%d) ...", len(df_main), config.BB_PERIOD)
    signals = generate_bb_signals(df_main, period=config.BB_PERIOD, std_mult=config.BB_STD_MULT)

    dist = dict(signals.value_counts().sort_index())
    log.info("신호 분포: 숏(%d)  플랫(%d)  롱(%d)",
             dist.get(-1, 0), dist.get(0, 0), dist.get(1, 0))

    metrics = run_backtest(
        df=df_main,
        signals=signals,
        leverage=config.LEVERAGE,
        initial_capital=args.capital,
    )

    if args.plot and metrics:
        plot_results(metrics, save_path="backtest/results.png")


def cmd_paper(args):
    from trading.paper_trader import PaperTrader
    trader = PaperTrader(
        instrument   = args.instrument,
        resolution   = args.resolution,
        poll_seconds = args.poll,
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
        instrument   = args.instrument,
        resolution   = args.resolution,
        poll_seconds = args.poll,
        dry_run      = args.dry_run,
    )
    trader.run(duration_hours=args.hours)


def main():
    parser = argparse.ArgumentParser(description="OrangeX Auto-Trader (BB 전략)")
    parser.add_argument("--mode", choices=["fetch", "train", "backtest", "paper", "live"],
                        default="backtest")
    parser.add_argument("--instrument",  default=config.INSTRUMENT)
    parser.add_argument("--resolution",  default=config.RESOLUTION,
                        help="캔들 기간 (기본 30분봉)")
    parser.add_argument("--days",        type=int,   default=config.TRAIN_DAYS,
                        help=f"조회 기간(일). 기본={config.TRAIN_DAYS} (10년)")
    parser.add_argument("--end-date",    default=None,
                        dest="end_date",
                        help=f"데이터 종료 날짜 YYYY-MM-DD. 기본={config.TRAIN_END_DATE}")
    parser.add_argument("--capital",     type=float, default=config.INITIAL_CAPITAL)
    parser.add_argument("--plot",        action="store_true", default=True)
    parser.add_argument("--hours",       type=float, default=24,
                        help="모의/실매매 실행 시간 (hours)")
    parser.add_argument("--poll",        type=int,   default=5,
                        help="모의/실매매 폴링 간격 (seconds)")
    parser.add_argument("--dry-run",     action="store_true", default=True,
                        dest="dry_run",
                        help="Live 모드에서 신호만 확인하고 실제 주문은 내지 않음")
    # 하위 호환용
    parser.add_argument("--sub-resolution", default=config.SUB_RESOLUTION,
                        dest="sub_resolution")
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
