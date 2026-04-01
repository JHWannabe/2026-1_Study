"""
Live Trader — OrangeX 실계좌 자동매매

실행 흐름:
  1. API 잔고 조회 → 주문 수량 계산
  2. 모델 신호 생성 (메인 15분 + 서브 5분)
  3. 시장가 주문 실행 (buy / sell)
  4. 포지션 모니터링 (SL / TP 소프트웨어 감시)
  5. 청산 조건 충족 시 reduce_only 주문으로 청산
"""

import time
import math
import pandas as pd
from datetime import datetime, timezone
from typing import Optional

from api.client import OrangeXClient
from api.data_fetcher import fetch_ohlcv
from features.multi_tf import build_multi_tf_features
from features.indicators import add_all_indicators
from model.trainer import load_model, predict_signal
from utils.logger import get_logger
import config

log = get_logger(__name__)

# OrangeX ETH-USDT-PERPETUAL 최소 주문 수량 (계약 단위 = USDT 기준)
MIN_ORDER_AMOUNT = 0.01
POSITION_SYNC_GRACE_SECONDS = 10


class LiveTrader:
    """
    실계좌 자동매매 루프.

    Parameters
    ----------
    instrument    : 종목 (기본 ETH-USDT-PERPETUAL)
    resolution    : 메인 캔들 (기본 15분)
    sub_resolution: 서브 캔들 (기본 5분)
    poll_seconds  : 루프 간격 (기본 30초)
    min_confidence: 최소 신호 신뢰도 (기본 0.55)
    dry_run       : True면 주문 직전까지만 진행하고 실제 API 호출 안 함
    """

    def __init__(
        self,
        instrument:     str   = config.INSTRUMENT,
        resolution:     str   = config.RESOLUTION,
        sub_resolution: str   = config.SUB_RESOLUTION,
        poll_seconds:   int   = 30,
        min_confidence: float = 0.55,
        dry_run:        bool  = False,
    ):
        self.instrument     = instrument
        self.resolution     = resolution
        self.sub_resolution = sub_resolution
        self.poll_seconds   = poll_seconds
        self.min_confidence = min_confidence
        self.dry_run        = dry_run

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

        # 모델 로드
        self.model, self.scaler, self.feature_cols, self.label_map, self.label_map_inv = load_model()

        # 거래 기록
        self.trades: list[dict] = []

        # 캔들 캐시 (API 과호출 방지)
        self._df_main_cache: Optional[pd.DataFrame] = None
        self._df_sub_cache:  Optional[pd.DataFrame] = None
        self._cache_ts: float = 0.0                               # 마지막 캐시 갱신 시각
        self._cache_ttl: int  = 300  # 5분마다 캔들 캐시 갱신

        mode = "[DRY-RUN]" if dry_run else "[LIVE]"
        log.info("%s LiveTrader 준비 — %s  leverage=%dx  SL=%.2f%%  TP=%.2f%%",
                 mode, instrument, config.LEVERAGE,
                 config.STOP_LOSS_PCT * config.LEVERAGE * 100,
                 config.TAKE_PROFIT_PCT * config.LEVERAGE * 100)

    # ─── 공개 진입점 ──────────────────────────────────────────────────────────

    def run(self, duration_hours: float = 24):
        """duration_hours 동안 매매 루프 실행."""
        end_time = time.time() + duration_hours * 3600
        log.info("매매 시작 — %.1f시간 동안 실행", duration_hours)

        # 기존 미결 포지션 동기화
        self._sync_position()

        while time.time() < end_time:
            try:
                self._step()
            except KeyboardInterrupt:
                log.info("사용자 중단 요청")
                break
            except Exception as e:
                log.error("루프 오류: %s", e, exc_info=True)

            time.sleep(self.poll_seconds)

        # 종료 시 포지션 강제 청산
        if self.position:
            log.info("시간 종료 — 포지션 청산 중 ...")
            self._close_position(reason="time_end")

        self._print_summary()

    # ─── 핵심 루프 ────────────────────────────────────────────────────────────

    def _step(self):
        now = datetime.now(timezone.utc).strftime("%H:%M:%S")

        # ① 현재가 조회
        ticker = self.client.get_ticker(self.instrument)
        price  = float(ticker.get("last_price", 0))
        if price == 0:
            log.warning("현재가 조회 실패")
            return

        # ② 포지션 API 동기화 (dry-run은 실제 포지션 없으므로 스킵)
        if not self.dry_run:
            self._sync_position()

        # ③ SL / TP 점검
        if self.position:
            if self._check_sl_tp(price):
                return   # 청산 완료 → 이번 스텝 종료

        # ④ 캔들 데이터 조회 (캔들 주기마다만 API 재요청)
        now_ts = time.time()
        if self._df_main_cache is None or (now_ts - self._cache_ts) >= self._cache_ttl:
            df_main = fetch_ohlcv(self.instrument, self.resolution,     days=30,
                                  client=self.client, use_cache=False)
            df_sub  = fetch_ohlcv(self.instrument, self.sub_resolution, days=14,
                                  client=self.client, use_cache=False)
            if not df_main.empty:
                self._df_main_cache = df_main
                self._df_sub_cache  = df_sub
                self._cache_ts      = now_ts
                log.info("캔들 캐시 갱신 — main=%d bars  sub=%d bars",
                         len(df_main), len(df_sub) if not df_sub.empty else 0)
        else:
            df_main = self._df_main_cache
            df_sub  = self._df_sub_cache
            log.debug("캐시 사용 (%.0f초 후 갱신)", self._cache_ttl - (now_ts - self._cache_ts))

        if df_main is None or df_main.empty or len(df_main) < 60:
            log.warning("캔들 데이터 부족 (%d bars)", len(df_main) if df_main is not None else 0)
            return

        # ⑤ 피처 생성 & 신호 예측
        # 현재가로 마지막 캔들 갱신 → conf가 매 틱마다 변동
        df_main_live = df_main.copy()
        last_idx = df_main_live.index[-1]
        df_main_live.at[last_idx, 'close'] = price
        if price > df_main_live.at[last_idx, 'high']:
            df_main_live.at[last_idx, 'high'] = price
        if price < df_main_live.at[last_idx, 'low']:
            df_main_live.at[last_idx, 'low'] = price

        if not df_sub.empty:
            df_feat = build_multi_tf_features(df_main_live, df_sub)
        else:
            df_feat = add_all_indicators(df_main_live)
            df_feat.dropna(inplace=True)

        signal, conf, all_probs = predict_signal(
            df_feat, self.model, self.scaler,
            self.feature_cols, self.label_map_inv,
            min_confidence=self.min_confidence,
        )

        log.info(
            "[%s] price=%.2f  signal=%+d  short=%.4f  long=%.4f  pos=%s",
            now, price, signal,
            all_probs["short"], all_probs["long"],
            self.position["side"] if self.position else "flat",
        )

        # ⑥ 주문 실행
        if self.position and conf <= config.EXIT_CONFIDENCE:
            log.info("confidence %.4f <= %.2f, closing position", conf, config.EXIT_CONFIDENCE)
            self._close_position(reason="low_confidence")
            return

        if signal == 0:
            return

        wanted_side = "long" if signal == 1 else "short"

        if not self.position:
            self._open_position(wanted_side, price)

    # ─── 주문 실행 ────────────────────────────────────────────────────────────

    def _open_position(self, side: str, price: float):
        """잔고 조회 → 수량 계산 → 시장가 진입."""
        balance = self._get_available_balance()
        if balance <= 0:
            log.error("사용 가능한 잔고 없음")
            return

        margin_to_use = balance * config.MAX_POSITION_PCT
        notional      = margin_to_use * config.LEVERAGE
        amount        = self._calc_amount(notional, price)

        if amount < self.min_order_amount:
            log.warning("주문 수량 %.4f이 최소값 %s 미만 — 스킵", amount, self.min_order_amount)
            return

        order_notional = amount * price
        if order_notional < self.min_notional:
            log.warning("주문 명목가 %.4f USDT가 최소값 %s 미만 — 스킵", order_notional, self.min_notional)
            return

        api_side = "buy" if side == "long" else "sell"

        log.info("[ORDER] %s %s  amount=%.4f  price~%.2f  notional~%.2f USDT",
                 "OPEN", api_side.upper(), amount, price, order_notional)

        if not self.dry_run:
            result = self._place_order_with_mode_retry(
                side=api_side,
                amount=amount,
                order_type="market",
                logical_side=side,
            )
            order_id = result.get("order", {}).get("order_id", "")
            log.info("주문 완료 — order_id=%s", order_id)
        else:
            log.info("[DRY-RUN] 실제 주문 미실행")

        sl_price = price * (1 - config.STOP_LOSS_PCT)  if side == "long" else price * (1 + config.STOP_LOSS_PCT)
        tp_price = price * (1 + config.TAKE_PROFIT_PCT) if side == "long" else price * (1 - config.TAKE_PROFIT_PCT)

        self.position = {
            "side":        side,
            "size":        amount,
            "entry_price": price,
            "entry_time":  datetime.now(timezone.utc),
            "sl":          sl_price,
            "tp":          tp_price,
            "notional":    order_notional,
        }
        log.info("포지션 오픈 — %s  SL=%.2f  TP=%.2f", side.upper(), sl_price, tp_price)

    def _close_position(self, reason: str = "signal"):
        """현재 포지션 전량 청산."""
        if not self.position:
            return

        pos      = self.position
        api_side = "sell" if pos["side"] == "long" else "buy"

        ticker = self.client.get_ticker(self.instrument)
        price  = float(ticker.get("last_price", pos["entry_price"]))

        log.info("[ORDER] CLOSE %s  amount=%.4f  price~%.2f  reason=%s",
                 api_side.upper(), pos["size"], price, reason)

        if not self.dry_run:
            result = self._place_order_with_mode_retry(
                side=api_side,
                amount=pos["size"],
                order_type="market",
                reduce_only=True,
                logical_side=pos["side"],
            )
            order_id = result.get("order", {}).get("order_id", "")
            log.info("청산 완료 — order_id=%s", order_id)
        else:
            log.info("[DRY-RUN] 실제 청산 미실행")

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
            log.warning("SL 도달 — price=%.2f  sl=%.2f", price, pos["sl"])
            self._close_position(reason="stop_loss")
            return True
        if hit_tp:
            log.info("TP 도달 — price=%.2f  tp=%.2f", price, pos["tp"])
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
                    log.info("API에 포지션 없음 — 로컬 포지션 초기화")
                    entry_time = self.position.get("entry_time")
                    if isinstance(entry_time, datetime):
                        age_seconds = (datetime.now(timezone.utc) - entry_time).total_seconds()
                        if age_seconds < POSITION_SYNC_GRACE_SECONDS:
                            log.warning(
                                "API 포지션 조회가 아직 비었습니다. 주문 직후 %.1f초 경과라 로컬 포지션을 유지합니다.",
                                age_seconds,
                            )
                            return
                    log.info("API에 포지션 없음 — 로컬 포지션 초기화")
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
                log.info("외부 포지션 감지 — %s  entry=%.2f  size=%.4f",
                         api_side.upper(), entry_price, size)
        except Exception as e:
            log.error("포지션 동기화 오류: %s", e)

    def _get_available_balance(self) -> float:
        """USDT 가용 잔고 조회."""
        try:
            raw = self.client.get_account_summary(currency="USDT")
            log.info("잔고 원본 응답: %s", raw)

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
            log.info("가용 잔고: %.4f USDT", balance)
            return balance
        except Exception as e:
            log.error("잔고 조회 오류: %s", e)
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
                log.warning("instruments API에서 %s 제약값을 찾지 못해 기본값을 사용합니다.", self.instrument)
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

            log.info(
                "주문 제약값 로드 — min_amount=%s  min_notional=%s  quantity_prec=%s",
                self.min_order_amount,
                self.min_notional,
                self.quantity_precision,
            )
        except Exception as e:
            log.warning("instruments API 제약값 조회 실패, 기본값 사용: %s", e)

    def _refresh_position_mode(self) -> bool:
        cfg = self.client.get_perpetual_config(self.instrument)
        self.hedge_mode = self._normalize_dual_side(cfg.get("dual_side_position", False))
        log.info("포지션 모드 동기화 — hedge_mode=%s", self.hedge_mode)
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

            log.warning("포지션 모드 불일치 감지 — 모드를 다시 읽고 한 번 더 주문합니다.")
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
            long_lev       = cfg.get("long_leverage", "?")
            short_lev      = cfg.get("short_leverage", "?")
            margin_type    = cfg.get("margin_type", "?")
            dual_side      = self._normalize_dual_side(cfg.get("dual_side_position", False))
            log.info(
                "현재 포지션 설정 — margin=%s  leverage=%s  long=%s  short=%s  hedge_mode=%s",
                margin_type, current_lev, long_lev, short_lev, dual_side
            )
        except Exception as e:
            log.warning("현재 설정 조회 실패: %s", e)
            return

        # 헤지 모드이면 API 변경 불가 → 안내만 출력
        if dual_side:
            self.hedge_mode = dual_side
            log.warning(
                "헤지 모드(dual_side_position) 감지 — 주문 시 pos_side 자동 추가\n"
                "  현재: Long %sx / Short %sx\n"
                "  목표: %dx\n"
                "  ※ 레버리지는 OrangeX 웹사이트에서 수동 조정 필요",
                long_lev, short_lev, config.LEVERAGE
            )
            return

        # 단방향 모드: 이미 원하는 레버리지면 스킵
        try:
            if int(current_lev) == config.LEVERAGE:
                log.info("레버리지 이미 %dx — 변경 불필요", config.LEVERAGE)
                return
        except (TypeError, ValueError):
            pass

        # 레버리지 변경 시도
        try:
            self.client.set_leverage(self.instrument, config.LEVERAGE)
            log.info("레버리지 %dx 설정 완료", config.LEVERAGE)
        except Exception as e:
            log.warning("레버리지 자동 설정 실패: %s", e)

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
