"""
Paper trader — runs in real-time using live OrangeX data
but executes orders against a virtual account only.
"""

import time
import pandas as pd
from datetime import datetime, timezone

from api.client import OrangeXClient
from api.data_fetcher import fetch_ohlcv
from model.trainer import load_model, predict_signal
from utils.logger import get_logger
import config

log = get_logger(__name__)


class VirtualAccount:
    def __init__(self, capital: float = config.INITIAL_CAPITAL):
        self.capital   = capital
        self.position  = None          # None | {"side", "entry", "size_usd", "margin"}
        self.trades    = []
        self.leverage  = config.LEVERAGE

    # ── Position management ───────────────────────────────────────────────────

    def open_position(self, side: str, price: float, ts):
        if self.position:
            log.warning("Already in a position — skip open.")
            return

        margin  = self.capital * config.MAX_POSITION_PCT
        notional = margin * self.leverage
        fee     = notional * config.TAKER_FEE
        margin  -= fee
        self.capital -= margin

        self.position = {
            "side":     side,
            "entry":    price,
            "size_usd": notional,
            "margin":   margin,
            "time":     ts,
            "sl":       price * (1 - config.STOP_LOSS_PCT)  if side == "long" else price * (1 + config.STOP_LOSS_PCT),
            "tp":       price * (1 + config.TAKE_PROFIT_PCT) if side == "long" else price * (1 - config.TAKE_PROFIT_PCT),
        }
        log.info("[PAPER] OPEN %s @ %.2f  notional=%.2f  margin=%.2f  SL=%.2f  TP=%.2f",
                 side.upper(), price, notional, margin,
                 self.position["sl"], self.position["tp"])

    def close_position(self, price: float, ts, reason: str = "signal"):
        if not self.position:
            return
        pos  = self.position
        sz   = pos["size_usd"]
        ep   = pos["entry"]
        mg   = pos["margin"]

        if pos["side"] == "long":
            raw_pnl = sz * (price - ep) / ep
        else:
            raw_pnl = sz * (ep - price) / ep

        fee     = sz * config.TAKER_FEE
        pnl     = raw_pnl - fee

        if reason == "liquidation":
            pnl = -mg

        self.capital += mg + pnl
        self.trades.append({
            "entry_time":  pos["time"],
            "exit_time":   ts,
            "side":        pos["side"],
            "entry_price": ep,
            "exit_price":  price,
            "pnl":         round(pnl, 4),
            "reason":      reason,
        })
        self.position = None
        log.info("[PAPER] CLOSE %s @ %.2f  pnl=%.2f  reason=%s  capital=%.2f",
                 pos["side"].upper(), price, pnl, reason, self.capital)

    def check_sl_tp(self, hi: float, lo: float, close: float, ts):
        if not self.position:
            return
        pos = self.position
        if pos["side"] == "long":
            liq = pos["entry"] * (1 - 1 / self.leverage * 0.9)
            if lo <= liq:
                return self.close_position(liq, ts, "liquidation")
            if lo <= pos["sl"]:
                return self.close_position(pos["sl"], ts, "stop_loss")
            if hi >= pos["tp"]:
                return self.close_position(pos["tp"], ts, "take_profit")
        else:
            liq = pos["entry"] * (1 + 1 / self.leverage * 0.9)
            if hi >= liq:
                return self.close_position(liq, ts, "liquidation")
            if hi >= pos["sl"]:
                return self.close_position(pos["sl"], ts, "stop_loss")
            if lo <= pos["tp"]:
                return self.close_position(pos["tp"], ts, "take_profit")

    @property
    def unrealized_pnl(self, price: float = 0):
        return 0.0   # computed externally

    def summary(self) -> dict:
        return {
            "capital":       round(self.capital, 2),
            "open_position": self.position,
            "trade_count":   len(self.trades),
            "realized_pnl":  round(sum(t["pnl"] for t in self.trades), 2),
        }


class PaperTrader:
    """
    Real-time paper trading loop.
    Polls OrangeX every `poll_seconds` seconds for new candles,
    runs the model, and updates the virtual account.
    """

    def __init__(
        self,
        instrument:   str   = config.INSTRUMENT,
        resolution:   str   = config.RESOLUTION,
        poll_seconds: int   = 60,
        min_confidence: float = 0.55,
    ):
        self.instrument    = instrument
        self.resolution    = resolution
        self.poll_seconds  = poll_seconds
        self.min_confidence= min_confidence
        self.client        = OrangeXClient()
        self.account       = VirtualAccount()
        self.model, self.scaler, self.feature_cols, self.label_map, self.label_map_inv = load_model()
        log.info("PaperTrader ready — instrument=%s  leverage=%dx", instrument, config.LEVERAGE)

    def run(self, duration_hours: float = 24):
        """Run the paper trading loop for `duration_hours` hours."""
        end_time = time.time() + duration_hours * 3600
        log.info("Starting paper trading for %.1f hours …", duration_hours)

        while time.time() < end_time:
            try:
                self._step()
            except Exception as e:
                log.error("Step error: %s", e)
            time.sleep(self.poll_seconds)

        if self.account.position:
            ticker = self.client.get_ticker(self.instrument)
            price  = ticker.get("last_price", 0)
            self.account.close_position(price, datetime.now(timezone.utc), "end")

        log.info("Paper trading finished. Summary: %s", self.account.summary())
        self._print_trades()

    def _step(self):
        # Fetch recent candles
        df = fetch_ohlcv(self.instrument, self.resolution, days=7,
                         client=self.client, use_cache=False)
        if df.empty or len(df) < 60:
            log.warning("Not enough data (%d bars)", len(df))
            return

        latest = df.iloc[-1]
        ts     = df.index[-1]
        hi, lo, close = latest["high"], latest["low"], latest["close"]

        # Check SL/TP on latest candle
        self.account.check_sl_tp(hi, lo, close, ts)

        # Get model prediction
        signal, conf, all_probs = predict_signal(
            df, self.model, self.scaler,
            self.feature_cols, self.label_map_inv,
            min_confidence=self.min_confidence,
        )
        log.info(
            "[%s] price=%.2f  signal=%+d  conf=%.4f  short=%.4f  long=%.4f  capital=%.2f",
            ts.strftime("%H:%M"), close, signal, conf,
            all_probs["short"], all_probs["long"],
            self.account.capital,
        )

        if self.account.position and conf <= config.EXIT_CONFIDENCE:
            self.account.close_position(close, ts, "low_confidence")
            return

        # Execute signal
        if signal != 0 and not self.account.position:
            side = "long" if signal == 1 else "short"
            self.account.open_position(side, close, ts)
        elif signal != 0 and self.account.position:
            current = self.account.position["side"]
            wanted  = "long" if signal == 1 else "short"
            if current != wanted:
                self.account.close_position(close, ts, "signal")
                self.account.open_position(wanted, close, ts)

    def _print_trades(self):
        if not self.account.trades:
            print("No trades executed.")
            return
        df = pd.DataFrame(self.account.trades)
        print("\n── Paper Trading History ──────────────────────────────")
        print(df.to_string(index=False))
        print(f"\nFinal capital: ${self.account.capital:,.2f}")
