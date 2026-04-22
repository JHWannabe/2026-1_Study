"""
Paper trader — 볼린저 밴드 전략 모의 자동매매
실제 OrangeX 실시간 데이터를 사용하되 주문은 가상 계좌에서 실행.
"""

import time
import pandas as pd
from datetime import datetime, timezone

from api.client import OrangeXClient
from api.data_fetcher import fetch_ohlcv
from features.bb_signals import generate_bb_signals, calc_bb_mid
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
    볼린저 밴드 전략 모의매매 루프.
    OrangeX 실시간 데이터를 poll_seconds 간격으로 조회하여 가상 계좌에서 매매.
    """

    def __init__(
        self,
        instrument:   str = config.INSTRUMENT,
        resolution:   str = config.RESOLUTION,
        poll_seconds: int = 60,
    ):
        self.instrument   = instrument
        self.resolution   = resolution
        self.poll_seconds = poll_seconds
        self.client       = OrangeXClient()
        self.account      = VirtualAccount()
        log.info("PaperTrader 준비 — instrument=%s  leverage=%dx  봉=%smin",
                 instrument, config.LEVERAGE, resolution)

    def run(self, duration_hours: float = 24):
        """duration_hours 동안 모의매매 루프 실행."""
        end_time = time.time() + duration_hours * 3600
        log.info("모의매매 시작 (%.1f시간)", duration_hours)

        while time.time() < end_time:
            try:
                self._step()
            except Exception as e:
                log.error("Step error: %s", e)
            time.sleep(self.poll_seconds)

        if self.account.position:
            ticker = self.client.get_ticker(self.instrument)
            price  = float(ticker.get("last_price", 0))
            self.account.close_position(price, datetime.now(timezone.utc), "end")

        log.info("모의매매 종료. 결과: %s", self.account.summary())
        self._print_trades()

    def _step(self):
        # ① 캔들 조회 (최근 7일)
        df = fetch_ohlcv(self.instrument, self.resolution, days=7,
                         client=self.client, use_cache=False)
        if df is None or df.empty or len(df) < config.BB_PERIOD + 5:
            log.warning("데이터 부족 (%d 캔들)", len(df) if df is not None else 0)
            return

        # ② 완성된 캔들 (마지막은 형성 중)
        df_complete = df.iloc[:-1]
        ts    = df_complete.index[-1]
        close = float(df_complete["close"].iloc[-1])
        hi    = float(df_complete["high"].iloc[-1])
        lo    = float(df_complete["low"].iloc[-1])

        # ③ SL/TP 점검
        self.account.check_sl_tp(hi, lo, close, ts)

        # ④ BB 중심선 계산
        bb_mid_series = calc_bb_mid(df_complete["close"], period=config.BB_PERIOD)
        curr_mid = float(bb_mid_series.iloc[-1])
        if curr_mid != curr_mid:   # NaN 체크
            return

        # ⑤ BB 중심선 기반 청산 점검 (진입 후 최소 1캔들 경과 후)
        if self.account.position:
            pos       = self.account.position
            candle_sec = int(self.resolution) * 60
            age = (ts - pos["time"]).total_seconds() if hasattr(pos["time"], "total_seconds") else \
                  (datetime.now(timezone.utc) - pos["time"]).total_seconds()

            if age >= candle_sec:
                if pos["side"] == "long" and close < curr_mid:
                    log.info("BB 청산 — 롱 중심선 하향 (종가=%.2f 중심선=%.2f)", close, curr_mid)
                    self.account.close_position(close, ts, "bb_exit")
                    return
                elif pos["side"] == "short" and close > curr_mid:
                    log.info("BB 청산 — 숏 중심선 상향 (종가=%.2f 중심선=%.2f)", close, curr_mid)
                    self.account.close_position(close, ts, "bb_exit")
                    return

        # ⑥ BB 신호 생성
        signals = generate_bb_signals(df_complete,
                                      period=config.BB_PERIOD,
                                      std_mult=config.BB_STD_MULT)
        signal = int(signals.iloc[-1])

        log.info("[%s] 종가=%.2f  중심선=%.2f  신호=%+d  자본=%.2f",
                 ts.strftime("%H:%M"), close, curr_mid, signal, self.account.capital)

        # ⑦ 진입
        if signal != 0 and not self.account.position:
            side = "long" if signal == 1 else "short"
            self.account.open_position(side, close, ts)

    def _print_trades(self):
        if not self.account.trades:
            print("No trades executed.")
            return
        df = pd.DataFrame(self.account.trades)
        print("\n── Paper Trading History ──────────────────────────────")
        print(df.to_string(index=False))
        print(f"\nFinal capital: ${self.account.capital:,.2f}")
