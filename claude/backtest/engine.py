"""
Backtesting engine — simulates trading on historical data with virtual assets.

Features:
  - 30x leverage on futures
  - Long / Short positions
  - Stop-loss and take-profit orders
  - Taker fees on entries and exits
  - Liquidation check (margin call if loss > initial margin)
  - Detailed trade log and performance metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dataclasses import dataclass, field
from typing import Literal
from pathlib import Path

from utils.logger import get_logger
import config

log = get_logger(__name__)

Side = Literal["long", "short", "flat"]


@dataclass
class Trade:
    entry_time:  pd.Timestamp
    exit_time:   pd.Timestamp
    side:        Side
    entry_price: float
    exit_price:  float
    size_usd:    float          # notional size in USD
    leverage:    int
    margin_used: float          # = size_usd / leverage
    pnl:         float          # net PnL in USD after fees
    pnl_pct:     float          # PnL / margin_used
    exit_reason: str            # "signal", "stop_loss", "take_profit", "liquidation", "end"


@dataclass
class BacktestState:
    capital:      float
    position:     Side  = "flat"
    entry_price:  float = 0.0
    entry_time:   pd.Timestamp = None
    size_usd:     float = 0.0   # notional
    margin_used:  float = 0.0
    stop_price:   float = 0.0
    tp_price:     float = 0.0
    trades:       list  = field(default_factory=list)
    equity_curve: list  = field(default_factory=list)


def run_backtest(
    df: pd.DataFrame,
    signals: pd.Series,         # index aligned with df, values ∈ {-1, 0, 1}
    leverage:       int   = config.LEVERAGE,
    initial_capital: float = config.INITIAL_CAPITAL,
    max_pos_pct:    float = config.MAX_POSITION_PCT,
    stop_loss_pct:  float = config.STOP_LOSS_PCT,
    take_profit_pct: float = config.TAKE_PROFIT_PCT,
    taker_fee:      float = config.TAKER_FEE,
) -> dict:
    """
    Run simulation bar-by-bar.

    Parameters
    ----------
    df      : OHLCV DataFrame (must include open/high/low/close)
    signals : Series of signals per bar (-1, 0, 1), same index as df
    """
    state = BacktestState(capital=initial_capital)
    state.equity_curve.append({"time": df.index[0], "equity": initial_capital})

    aligned = df.join(signals.rename("signal"), how="left").fillna(0)

    for i, (ts, row) in enumerate(aligned.iterrows()):
        price  = row["close"]
        signal = int(row["signal"])
        hi     = row["high"]
        lo     = row["low"]

        # ── Check exits first (SL / TP / liquidation hit within bar) ──────────
        if state.position != "flat":
            exit_reason, exit_price = _check_exit(state, hi, lo, price, leverage)
            if exit_reason:
                _close_position(state, ts, exit_price, exit_reason, taker_fee)

        # ── Open new position on signal ────────────────────────────────────────
        if state.position == "flat" and signal != 0:
            side        = "long" if signal == 1 else "short"
            margin_used = state.capital * max_pos_pct
            if margin_used < 1:                      # not enough capital
                continue

            notional    = margin_used * leverage
            fee         = notional * taker_fee
            margin_used -= fee                       # fee debited from margin

            if side == "long":
                sl_price = price * (1 - stop_loss_pct)
                tp_price = price * (1 + take_profit_pct)
            else:
                sl_price = price * (1 + stop_loss_pct)
                tp_price = price * (1 - take_profit_pct)

            state.position    = side
            state.entry_price = price
            state.entry_time  = ts
            state.size_usd    = notional
            state.margin_used = margin_used
            state.stop_price  = sl_price
            state.tp_price    = tp_price
            state.capital    -= margin_used          # lock margin

        # ── BB 청산 신호(0) 또는 방향 전환 처리 ──────────────────────────────────
        elif state.position != "flat":
            if signal == 0:
                # 중심선 돌파 청산 신호
                _close_position(state, ts, price, "bb_exit", taker_fee)
            else:
                current_side  = state.position
                expected_side = "long" if signal == 1 else "short"
                if current_side != expected_side:
                    _close_position(state, ts, price, "signal", taker_fee)

        state.equity_curve.append({"time": ts, "equity": _total_equity(state, price)})

    # Force-close any open position at end
    if state.position != "flat":
        last_row   = aligned.iloc[-1]
        last_price = last_row["close"]
        _close_position(state, aligned.index[-1], last_price, "end", taker_fee)

    return _compute_metrics(state, initial_capital)


# ─── Internals ────────────────────────────────────────────────────────────────

def _check_exit(state: BacktestState, hi: float, lo: float,
                close: float, leverage: int) -> tuple[str, float]:
    """Returns (reason, exit_price) or (None, None) if no exit triggered."""
    if state.position == "long":
        # Liquidation: price drops enough to wipe margin
        liq_price = state.entry_price * (1 - 1 / leverage * 0.9)
        if lo <= liq_price:
            return "liquidation", liq_price
        if lo <= state.stop_price:
            return "stop_loss", state.stop_price
        if hi >= state.tp_price:
            return "take_profit", state.tp_price

    elif state.position == "short":
        liq_price = state.entry_price * (1 + 1 / leverage * 0.9)
        if hi >= liq_price:
            return "liquidation", liq_price
        if hi >= state.stop_price:
            return "stop_loss", state.stop_price
        if lo <= state.tp_price:
            return "take_profit", state.tp_price

    return None, None


def _close_position(state: BacktestState, ts, exit_price: float,
                    reason: str, taker_fee: float):
    ep    = state.entry_price
    sz    = state.size_usd
    fee   = sz * taker_fee

    if state.position == "long":
        raw_pnl = sz * (exit_price - ep) / ep
    else:
        raw_pnl = sz * (ep - exit_price) / ep

    pnl      = raw_pnl - fee
    margin   = state.margin_used

    if reason == "liquidation":
        pnl = -margin                   # total margin loss

    state.capital += margin + pnl

    trade = Trade(
        entry_time  = state.entry_time,
        exit_time   = ts,
        side        = state.position,
        entry_price = ep,
        exit_price  = exit_price,
        size_usd    = sz,
        leverage    = config.LEVERAGE,
        margin_used = margin,
        pnl         = pnl,
        pnl_pct     = pnl / margin if margin > 0 else 0,
        exit_reason = reason,
    )
    state.trades.append(trade)
    state.position    = "flat"
    state.entry_price = 0.0
    state.size_usd    = 0.0
    state.margin_used = 0.0

    log.debug("CLOSE %s @ %.2f  reason=%s  pnl=%.2f  capital=%.2f",
              trade.side, exit_price, reason, pnl, state.capital)


def _total_equity(state: BacktestState, price: float) -> float:
    if state.position == "flat":
        return state.capital
    ep = state.entry_price
    sz = state.size_usd
    mg = state.margin_used
    if state.position == "long":
        unrealized = sz * (price - ep) / ep
    else:
        unrealized = sz * (ep - price) / ep
    return state.capital + mg + unrealized


def _compute_metrics(state: BacktestState, initial_capital: float) -> dict:
    trades = state.trades
    equity = pd.DataFrame(state.equity_curve).set_index("time")["equity"]

    total_trades  = len(trades)
    if total_trades == 0:
        log.warning("No trades executed.")
        return {}

    pnls        = np.array([t.pnl for t in trades])
    wins        = pnls[pnls > 0]
    losses      = pnls[pnls <= 0]
    win_rate    = len(wins) / total_trades

    final_cap   = state.capital
    total_return= (final_cap - initial_capital) / initial_capital

    # Sharpe (hourly returns → annualised)
    ret_series  = equity.pct_change().dropna()
    sharpe      = (ret_series.mean() / (ret_series.std() + 1e-9)) * np.sqrt(8760)

    # Max drawdown
    roll_max    = equity.cummax()
    drawdown    = (equity - roll_max) / roll_max
    max_dd      = drawdown.min()

    # Profit factor
    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss   = abs(losses.sum()) if len(losses) > 0 else 1e-9
    profit_factor= gross_profit / gross_loss

    by_reason = {}
    for t in trades:
        by_reason[t.exit_reason] = by_reason.get(t.exit_reason, 0) + 1

    metrics = {
        "initial_capital": initial_capital,
        "final_capital":   round(final_cap, 2),
        "total_return_pct": round(total_return * 100, 2),
        "total_trades":    total_trades,
        "win_rate":        round(win_rate * 100, 2),
        "avg_win_usd":     round(wins.mean(), 2) if len(wins) > 0 else 0,
        "avg_loss_usd":    round(losses.mean(), 2) if len(losses) > 0 else 0,
        "profit_factor":   round(profit_factor, 3),
        "sharpe_ratio":    round(sharpe, 3),
        "max_drawdown_pct":round(max_dd * 100, 2),
        "exit_reasons":    by_reason,
        "trades":          trades,
        "equity_curve":    equity,
    }

    _print_metrics(metrics)
    return metrics


def _print_trade_journal(trades: list):
    if not trades:
        return
    print("\n" + "=" * 100)
    print("  TRADE JOURNAL")
    print("=" * 100)
    header = f"  {'#':>4}  {'Side':>6}  {'Entry Time':>16}  {'Exit Time':>16}  " \
             f"{'Entry':>9}  {'Exit':>9}  {'Margin':>8}  {'PnL':>9}  {'PnL%':>7}  {'Reason'}"
    print(header)
    print("-" * 100)
    cumulative = 0.0
    for i, t in enumerate(trades, 1):
        cumulative += t.pnl
        sign = "+" if t.pnl >= 0 else ""
        entry_str = t.entry_time.strftime("%m-%d %H:%M") if t.entry_time else "-"
        exit_str  = t.exit_time.strftime("%m-%d %H:%M")  if t.exit_time  else "-"
        print(
            f"  {i:>4}  {t.side:>6}  {entry_str:>16}  {exit_str:>16}  "
            f"{t.entry_price:>9.2f}  {t.exit_price:>9.2f}  "
            f"{t.margin_used:>8.2f}  {sign}{t.pnl:>8.2f}  "
            f"{sign}{t.pnl_pct*100:>6.1f}%  {t.exit_reason}"
        )
    print("-" * 100)
    sign = "+" if cumulative >= 0 else ""
    print(f"  {'누적 손익':>50}  {sign}{cumulative:.2f} USD")
    print("=" * 100 + "\n")

    # CSV 저장
    rows = [
        {
            "trade_no":    i + 1,
            "side":        t.side,
            "entry_time":  t.entry_time.strftime("%Y-%m-%d %H:%M") if t.entry_time else "",
            "exit_time":   t.exit_time.strftime("%Y-%m-%d %H:%M")  if t.exit_time  else "",
            "entry_price": round(t.entry_price, 4),
            "exit_price":  round(t.exit_price, 4),
            "size_usd":    round(t.size_usd, 4),
            "margin_used": round(t.margin_used, 4),
            "pnl_usd":     round(t.pnl, 4),
            "pnl_pct":     round(t.pnl_pct * 100, 2),
            "exit_reason": t.exit_reason,
        }
        for i, t in enumerate(trades)
    ]
    save_path = Path("backtest/trade_journal.csv")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(save_path, index=False)
    log.info("매매일지 저장 → %s", save_path)


def _print_metrics(m: dict):
    _print_trade_journal(m.get("trades", []))
    print("\n" + "=" * 55)
    print("  BACKTEST RESULTS")
    print("=" * 55)
    print(f"  Capital:        ${m['initial_capital']:>10,.2f}  →  ${m['final_capital']:>10,.2f}")
    print(f"  Total Return:   {m['total_return_pct']:>+.2f}%")
    print(f"  Total Trades:   {m['total_trades']}")
    print(f"  Win Rate:       {m['win_rate']:.2f}%")
    print(f"  Avg Win:        ${m['avg_win_usd']:>8.2f}")
    print(f"  Avg Loss:       ${m['avg_loss_usd']:>8.2f}")
    print(f"  Profit Factor:  {m['profit_factor']:.3f}")
    print(f"  Sharpe Ratio:   {m['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown:   {m['max_drawdown_pct']:.2f}%")
    print(f"  Exit Reasons:   {m['exit_reasons']}")
    print("=" * 55 + "\n")


def plot_results(metrics: dict, save_path: str = "backtest/results.png"):
    equity = metrics.get("equity_curve")
    trades = metrics.get("trades", [])
    if equity is None or equity.empty:
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    fig.suptitle("Backtest Results", fontsize=14, fontweight="bold")

    # ── Equity curve ──────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(equity.index, equity.values, color="steelblue", linewidth=1.2)
    ax.set_title("Equity Curve")
    ax.set_ylabel("Capital (USD)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(True, alpha=0.3)

    # ── Drawdown ──────────────────────────────────────────────────────────────
    ax = axes[1]
    roll_max = equity.cummax()
    dd       = (equity - roll_max) / roll_max * 100
    ax.fill_between(dd.index, dd.values, 0, color="crimson", alpha=0.5)
    ax.set_title("Drawdown (%)")
    ax.set_ylabel("DD %")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(True, alpha=0.3)

    # ── Trade PnL distribution ────────────────────────────────────────────────
    ax = axes[2]
    pnls = [t.pnl for t in trades]
    colors = ["green" if p > 0 else "red" for p in pnls]
    ax.bar(range(len(pnls)), pnls, color=colors, alpha=0.7, width=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Trade PnL (USD)")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("PnL (USD)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    log.info("Plot saved → %s", save_path)
    plt.show()
