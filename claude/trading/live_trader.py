"""
Live Trader — OrangeX 실계좌 자동매매 (볼린저 밴드 전략)

전략:
  [롱]  이전 종가 < 중심선, 현재 종가 > 중심선 → 이후 저가가 중심선 도달 시 롱 진입
        종가 < 중심선 → 청산
  [숏]  이전 종가 > 중심선, 현재 종가 < 중심선 → 이후 고가가 중심선 도달 시 숏 진입
        종가 > 중심선 → 청산

실행 흐름:
  1. API 잔고 조회 → 주문 수량 계산
  2. 완성된 30분봉 기준 BB 신호 생성
  3. 현재가 vs BB 중심선으로 청산 여부 실시간 점검
  4. 시장가 주문 실행 (buy / sell)
  5. 포지션 모니터링 (SL 소프트웨어 감시)
  6. 청산 조건 충족 시 reduce_only 주문으로 청산
"""

import time
import math
import pandas as pd
from datetime import datetime, timezone
from typing import Optional

from api.client import OrangeXClient
from api.data_fetcher import fetch_ohlcv
from features.bb_signals import generate_bb_signals, calc_bb_mid
from utils.logger import get_logger
import config

log = get_logger(__name__)

# OrangeX ETH-USDT-PERPETUAL 최소 주문 수량 (계약 단위 = USDT 기준)
MIN_ORDER_AMOUNT = 0.01
POSITION_SYNC_GRACE_SECONDS = 10


class LiveTrader:
    """
    실계좌 자동매매 루프 (볼린저 밴드 전략).

    Parameters
    ----------
    instrument   : 종목 (기본 ETH-USDT-PERPETUAL)
    resolution   : 캔들 기간 (기본 30분)
    poll_seconds : 루프 간격 (기본 5초)
    dry_run      : True면 주문 직전까지만 진행하고 실제 API 호출 안 함
    """

    def __init__(
        self,
        instrument:   str  = config.INSTRUMENT,
        resolution:   str  = config.RESOLUTION,
        poll_seconds: int  = 5,
        dry_run:      bool = False,
    ):
        self.instrument   = instrument
        self.resolution   = resolution
        self.poll_seconds = poll_seconds
        self.dry_run      = dry_run

        self.client = OrangeXClient()
        self.hedge_mode: bool = False   # dual_side_position 여부 — _set_leverage에서 설정
        self.min_order_amount: float = MIN_ORDER_AMOUNT
        self.min_notional: float = 10.0
        self.quantity_precision: int = 2

        self._load_instrument_rules()

        # 레버리지 설정
        self._set_leverage()

        # 현재 포지션 상태 (API에서 동기화)
        self.position: dict | None = None   # {"side", "size", "entry_price", "sl", "tp"}

        # 거래 기록
        self.trades: list[dict] = []

        # 연속 손실 추적
        self._consecutive_losses: int = 0
        self._halt_until_date: Optional[str] = None  # "YYYY-MM-DD" 형식

    # ─── 공개 진입점 ──────────────────────────────────────────────────────────

    def run(self, duration_hours: float = 24):
        """duration_hours 동안 매매 루프 실행."""
        end_time = time.time() + duration_hours * 3600

        # 기존 미결 포지션 동기화
        self._sync_position()

        while time.time() < end_time:
            try:
                self._step()
            except KeyboardInterrupt:
                break
            except Exception as e:
                log.error("루프 오류: %s", e, exc_info=True)

            time.sleep(self.poll_seconds)

        # 종료 시 포지션 강제 청산
        if self.position:
            self._close_position(reason="time_end")

        self._print_summary()

    # ─── 핵심 루프 ────────────────────────────────────────────────────────────

    def _step(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._halt_until_date == today:
            return

        # ① 현재가 조회
        ticker = self.client.get_ticker(self.instrument)
        price  = float(ticker.get("last_price", 0))
        if price == 0:
            return

        # ② 포지션 API 동기화 (dry-run은 실제 포지션 없으므로 스킵)
        if not self.dry_run:
            self._sync_position()

        # ③ SL 점검
        if self.position:
            if self._check_sl_tp(price):
                return   # 청산 완료 → 이번 스텝 종료

        # ④ 30분봉 캔들 조회 (최근 7일 = ~336캔들)
        df = fetch_ohlcv(self.instrument, self.resolution, days=7,
                         client=self.client, use_cache=False)
        if df is None or df.empty or len(df) < config.BB_PERIOD + 5:
            return

        # ⑤ 완성된 캔들만 사용 (마지막 캔들은 현재 형성 중)
        df_complete = df.iloc[:-1]

        # ⑥ 볼린저 밴드 중심선 계산 (완성된 캔들 기준)
        bb_mid_series = calc_bb_mid(df_complete["close"], period=config.BB_PERIOD)
        curr_mid = bb_mid_series.iloc[-1]
        if curr_mid != curr_mid:   # NaN 체크
            return

        # ⑦ BB 중심선 기반 청산 점검 (현재가 vs 마지막 완성 캔들 중심선)
        #    진입 직후 즉시 청산 방지: 최소 1캔들(30분) 경과 후부터 체크
        if self.position:
            entry_time = self.position.get("entry_time")
            candle_seconds = int(self.resolution) * 60
            age = (datetime.now(timezone.utc) - entry_time).total_seconds() if entry_time else 0

            if age >= candle_seconds:
                pos_side = self.position["side"]
                if pos_side == "long" and price < curr_mid:
                    log.info("BB 청산 — 롱 중심선 하향 돌파 (가격=%.2f 중심선=%.2f)", price, curr_mid)
                    self._close_position(reason="bb_exit")
                    return
                elif pos_side == "short" and price > curr_mid:
                    log.info("BB 청산 — 숏 중심선 상향 돌파 (가격=%.2f 중심선=%.2f)", price, curr_mid)
                    self._close_position(reason="bb_exit")
                    return

        # ⑧ BB 신호 생성 (완성된 캔들 기준)
        signals = generate_bb_signals(df_complete,
                                      period=config.BB_PERIOD,
                                      std_mult=config.BB_STD_MULT)
        signal = int(signals.iloc[-1])

        log.info("[%s] 가격=%.2f  BB중심선=%.2f  신호=%+d  포지션=%s",
                 datetime.now(timezone.utc).strftime("%H:%M"),
                 price, curr_mid, signal,
                 self.position["side"] if self.position else "flat")

        # ⑨ 포지션 없을 때 진입
        if not self.position and signal != 0:
            side = "long" if signal == 1 else "short"
            self._open_position(side, price)

    # ─── 주문 실행 ────────────────────────────────────────────────────────────

    def _open_position(self, side: str, price: float):
        """잔고 조회 → 수량 계산 → 시장가 진입."""
        balance = self._get_available_balance()
        if balance <= 0:
            return

        amount = self._calc_amount(balance, price)

        api_side = "buy" if side == "long" else "sell"

        if not self.dry_run:
            self._place_order_with_mode_retry(
                side=api_side,
                amount=amount,
                order_type="market",
                logical_side=side,
            )

        sl_price = price * (1 - config.STOP_LOSS_PCT)  if side == "long" else price * (1 + config.STOP_LOSS_PCT)
        tp_price = price * (1 + config.TAKE_PROFIT_PCT) if side == "long" else price * (1 - config.TAKE_PROFIT_PCT)

        self.position = {
            "side":        side,
            "size":        amount,
            "entry_price": price,
            "entry_time":  datetime.now(timezone.utc),
            "sl":          sl_price,
            "tp":          tp_price,
            "notional":    amount,
        }

    def _close_position(self, reason: str = "signal"):
        """현재 포지션 전량 청산."""
        if not self.position:
            return

        pos      = self.position
        api_side = "sell" if pos["side"] == "long" else "buy"

        ticker = self.client.get_ticker(self.instrument)
        price  = float(ticker.get("last_price", pos["entry_price"]))

        if not self.dry_run:
            self._place_order_with_mode_retry(
                side=api_side,
                amount=pos["size"],
                order_type="market",
                reduce_only=True,
                logical_side=pos["side"],
            )

        # PnL 계산 (참고용)
        ep = pos["entry_price"]
        sz = pos["notional"]
        if pos["side"] == "long":
            pnl = sz * (price - ep) / ep - sz * config.TAKER_FEE
        else:
            pnl = sz * (ep - price) / ep - sz * config.TAKER_FEE

        self.trades.append({
            "entry_time":  pos["entry_time"].strftime("%Y-%m-%d %H:%M"),
            "exit_time":   datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "side":        pos["side"],
            "entry_price": round(ep, 4),
            "exit_price":  round(price, 4),
            "pnl_usdt":    round(pnl, 4),
            "reason":      reason,
        })
        self.position = None

        # 연속 손실 카운터 업데이트
        if pnl < 0:
            self._consecutive_losses += 1
            if self._consecutive_losses >= 3:
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                self._halt_until_date = today
        else:
            self._consecutive_losses = 0

    # ─── 유틸리티 ─────────────────────────────────────────────────────────────

    def _check_sl_tp(self, price: float) -> bool:
        """SL 또는 TP 터치 여부 확인. 청산 시 True 반환."""
        pos = self.position
        if not pos:
            return False

        hit_sl = hit_tp = False
        if pos["side"] == "long":
            hit_sl = price <= pos["sl"]
            hit_tp = price >= pos["tp"]
        else:
            hit_sl = price >= pos["sl"]
            hit_tp = price <= pos["tp"]

        if hit_sl:
            self._close_position(reason="stop_loss")
            return True
        if hit_tp:
            self._close_position(reason="take_profit")
            return True
        return False

    def _sync_position(self):
        """API에서 실제 포지션 조회 후 로컬 상태와 동기화."""
        try:
            positions = self.client.get_positions(currency="PERPETUAL", kind="perpetual")
            eth_pos   = [p for p in positions
                         if p.get("instrument_name") == self.instrument and self._safe_float(p.get("size")) != 0]

            if not eth_pos:
                if self.position:
                    entry_time = self.position.get("entry_time")
                    if isinstance(entry_time, datetime):
                        age_seconds = (datetime.now(timezone.utc) - entry_time).total_seconds()
                        if age_seconds < POSITION_SYNC_GRACE_SECONDS:
                            return
                    self.position = None
                return

            p = eth_pos[0]
            api_size = self._safe_float(p.get("size"))
            api_side = "long" if api_size > 0 else "short"

            if self.position is None:
                # 외부에서 진입된 포지션 감지
                entry_price = self._safe_float(p.get("average_price") or p.get("mark_price"))
                size        = abs(api_size)
                sl_price = entry_price * (1 - config.STOP_LOSS_PCT)  if api_side == "long" else entry_price * (1 + config.STOP_LOSS_PCT)
                tp_price = entry_price * (1 + config.TAKE_PROFIT_PCT) if api_side == "long" else entry_price * (1 - config.TAKE_PROFIT_PCT)
                self.position = {
                    "side":        api_side,
                    "size":        size,
                    "entry_price": entry_price,
                    "entry_time":  datetime.now(timezone.utc),
                    "sl":          sl_price,
                    "tp":          tp_price,
                    "notional":    size,
                }
        except Exception:
            pass

    def _get_available_balance(self) -> float:
        """USDT 가용 잔고 조회."""
        try:
            raw = self.client.get_account_summary(currency="USDT")

            # 응답이 list면 첫 번째 원소 사용
            summary = raw[0] if isinstance(raw, list) else raw

            # 중첩 dict인 경우 PERPETUAL → USDT 순으로 탐색
            if isinstance(summary, dict):
                for key in ("PERPETUAL", "perpetual"):
                    if key in summary:
                        inner = summary[key]
                        summary = inner.get("USDT", inner) if isinstance(inner, dict) else inner
                        break

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
            return balance
        except Exception:
            return 0.0

    def _calc_amount(self, notional_usdt: float, price: float) -> float:
        """
        주문 수량 계산.
        OrangeX ETH-USDT-PERPETUAL: 수량 단위 확인 필요.
        일단 USDT 명목 가치로 제출 (quantityPrec=0 이므로 정수 내림).
        """
        if price <= 0:
            return 0.0

        raw_amount = notional_usdt / price
        step = self.min_order_amount if self.min_order_amount > 0 else MIN_ORDER_AMOUNT
        amount = math.floor(raw_amount / step) * step
        return round(max(amount, 0.0), self.quantity_precision)

    def _normalize_dual_side(self, value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "y", "on"}
        return False

    def _safe_float(self, value, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _load_instrument_rules(self):
        try:
            instruments = self.client.get_instruments(currency="PERPETUAL", kind="perpetual")
            instrument = next((x for x in instruments if x.get("instrument_name") == self.instrument), None)
            if not instrument:
                return

            self.min_order_amount = self._safe_float(
                instrument.get("min_trade_amount") or instrument.get("min_qty"),
                MIN_ORDER_AMOUNT,
            )
            self.min_notional = self._safe_float(instrument.get("min_notional"), 10.0)
            try:
                self.quantity_precision = int(instrument.get("quantityPrec", 2) or 2)
            except (TypeError, ValueError):
                self.quantity_precision = 2
        except Exception:
            pass

    def _refresh_position_mode(self) -> bool:
        cfg = self.client.get_perpetual_config(self.instrument)
        self.hedge_mode = self._normalize_dual_side(cfg.get("dual_side_position", False))
        return self.hedge_mode

    def _place_order_with_mode_retry(
        self,
        side: str,
        amount: float,
        order_type: str = "market",
        reduce_only: bool = False,
        logical_side: str | None = None,
    ) -> dict:
        position_side = "BOTH"
        if self.hedge_mode:
            position_side = "LONG" if logical_side == "long" else "SHORT"
        try:
            return self.client.place_order(
                instrument_name=self.instrument,
                side=side,
                amount=amount,
                order_type=order_type,
                reduce_only=reduce_only,
                position_side=position_side,
            )
        except RuntimeError as e:
            if "position mode" not in str(e).lower():
                raise

            retry_hedge_mode = self._refresh_position_mode()
            retry_position_side = "BOTH"
            if retry_hedge_mode:
                retry_position_side = "LONG" if logical_side == "long" else "SHORT"
            return self.client.place_order(
                instrument_name=self.instrument,
                side=side,
                amount=amount,
                order_type=order_type,
                reduce_only=reduce_only,
                position_side=retry_position_side,
            )

    def _set_leverage(self):
        # 현재 설정 조회
        try:
            cfg            = self.client.get_perpetual_config(self.instrument)
            current_lev    = cfg.get("leverage", "?")
            dual_side      = self._normalize_dual_side(cfg.get("dual_side_position", False))
        except Exception:
            return

        # 헤지 모드이면 API 변경 불가
        if dual_side:
            self.hedge_mode = dual_side
            return

        # 단방향 모드: 이미 원하는 레버리지면 스킵
        try:
            if int(current_lev) == config.LEVERAGE:
                return
        except (TypeError, ValueError):
            pass

        # 레버리지 변경 시도
        try:
            self.client.set_leverage(self.instrument, config.LEVERAGE)
        except Exception:
            pass

    def _print_summary(self):
        print("\n" + "=" * 55)
        print("  실매매 결과 요약")
        print("=" * 55)
        if not self.trades:
            print("  체결된 거래 없음")
        else:
            df = pd.DataFrame(self.trades)
            total_pnl = df["pnl_usdt"].sum()
            wins      = (df["pnl_usdt"] > 0).sum()
            print(df.to_string(index=False))
            print(f"\n  총 거래: {len(df)}건")
            print(f"  승률:   {wins}/{len(df)} ({wins/len(df)*100:.1f}%)")
            print(f"  손익:   {total_pnl:+.4f} USDT")
        print("=" * 55)
