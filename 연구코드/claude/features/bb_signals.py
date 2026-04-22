"""
볼린저 밴드 평균회귀 신호 생성기

전략 로직
---------
[롱 진입]
  1. 트리거 캔들: 이전 종가 < 중심선 AND 현재 종가 > 중심선
     (하단 → 상단 이동)
  2. 진입 조건: 이후 캔들의 저가(low) <= 중심선
     (중심선까지 되돌림 발생)
  3. 청산 조건: 종가 < 중심선 (중심선 아래로 돌파)

[숏 진입]
  1. 트리거 캔들: 이전 종가 > 중심선 AND 현재 종가 < 중심선
     (상단 → 하단 이동)
  2. 진입 조건: 이후 캔들의 고가(high) >= 중심선
     (중심선까지 되돌림 발생)
  3. 청산 조건: 종가 > 중심선 (중심선 위로 돌파)
"""

import numpy as np
import pandas as pd


def calc_bb_mid(close: pd.Series, period: int = 20) -> pd.Series:
    """볼린저 밴드 중심선(단순이동평균) 계산."""
    return close.rolling(period).mean()


def calc_bb_bands(
    close: pd.Series,
    period: int = 20,
    std_mult: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """볼린저 밴드 상/중/하단 반환 (upper, mid, lower)."""
    mid   = close.rolling(period).mean()
    std   = close.rolling(period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return upper, mid, lower


def generate_bb_signals(
    df: pd.DataFrame,
    period: int = 20,
    std_mult: float = 2.0,
) -> pd.Series:
    """
    볼린저 밴드 평균회귀 전략 신호 생성.

    Parameters
    ----------
    df        : OHLCV DataFrame (open, high, low, close, volume 컬럼 필요)
    period    : 볼린저 밴드 기간 (기본 20)
    std_mult  : 표준편차 배수 (기본 2.0)

    Returns
    -------
    pd.Series[int]  (index = df.index)
         1 : 롱 포지션 (진입 또는 유지)
        -1 : 숏 포지션 (진입 또는 유지)
         0 : 플랫 (포지션 없음 또는 청산)

    상태 머신
    ---------
    0 = flat          : 포지션 없음
    1 = watching_long : 롱 트리거 발생, 중심선 되돌림 대기
    2 = long          : 롱 포지션 보유 중
    3 = watching_short: 숏 트리거 발생, 중심선 되돌림 대기
    4 = short         : 숏 포지션 보유 중
    """
    _, mid, _ = calc_bb_bands(df["close"], period, std_mult)

    c  = df["close"].values
    lo = df["low"].values
    hi = df["high"].values
    m  = mid.values
    n  = len(df)

    signals = np.zeros(n, dtype=int)
    state   = 0  # flat

    for i in range(1, n):
        m_prev  = m[i - 1]
        m_curr  = m[i]
        c_prev  = c[i - 1]
        c_curr  = c[i]
        lo_curr = lo[i]
        hi_curr = hi[i]

        if np.isnan(m_curr) or np.isnan(m_prev):
            continue

        # ── flat ─────────────────────────────────────────────────────────────
        if state == 0:
            if c_prev < m_prev and c_curr > m_curr:   # 하단 → 상단 돌파
                state = 1  # watching_long
            elif c_prev > m_prev and c_curr < m_curr:  # 상단 → 하단 돌파
                state = 3  # watching_short

        # ── watching_long ─────────────────────────────────────────────────────
        elif state == 1:
            if lo_curr <= m_curr:                      # 저가가 중심선에 도달 → 롱 진입
                state = 2
                signals[i] = 1
            elif c_curr < m_curr:                      # 중심선 아래로 내려감 → 대기 취소
                state = 0
                if c_prev > m_prev:                    # 동시에 크로스다운 → 숏 대기로
                    state = 3

        # ── long ──────────────────────────────────────────────────────────────
        elif state == 2:
            if c_curr < m_curr:                        # 종가 < 중심선 → 청산
                signals[i] = 0
                state = 0
                if c_prev > m_prev:                    # 크로스다운 발생 → 숏 대기
                    state = 3
            else:
                signals[i] = 1                         # 롱 유지

        # ── watching_short ────────────────────────────────────────────────────
        elif state == 3:
            if hi_curr >= m_curr:                      # 고가가 중심선에 도달 → 숏 진입
                state = 4
                signals[i] = -1
            elif c_curr > m_curr:                      # 중심선 위로 올라감 → 대기 취소
                state = 0
                if c_prev < m_prev:                    # 크로스업 → 롱 대기로
                    state = 1

        # ── short ─────────────────────────────────────────────────────────────
        elif state == 4:
            if c_curr > m_curr:                        # 종가 > 중심선 → 청산
                signals[i] = 0
                state = 0
                if c_prev < m_prev:                    # 크로스업 발생 → 롱 대기
                    state = 1
            else:
                signals[i] = -1                        # 숏 유지

    return pd.Series(signals, index=df.index, name="bb_signal")
