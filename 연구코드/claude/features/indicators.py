"""
Technical indicator feature engineering.
All functions accept a DataFrame with columns: open, high, low, close, volume
and return the same DataFrame with added indicator columns.
"""

import pandas as pd
import numpy as np


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _add_trend(df)
    df = _add_momentum(df)
    df = _add_volatility(df)
    df = _add_volume(df)
    df = _add_candle_patterns(df)
    df = _add_supertrend(df)
    df = _add_macd_supertrend_signals(df)
    df.dropna(inplace=True)
    return df


# ─── Trend ────────────────────────────────────────────────────────────────────

def _add_trend(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]

    for p in [7, 14, 21, 50, 200]:
        df[f"ema_{p}"]  = c.ewm(span=p, adjust=False).mean()
        df[f"sma_{p}"]  = c.rolling(p).mean()

    # EMA crosses
    df["ema_7_21_cross"]  = (df["ema_7"]  > df["ema_21"]).astype(int)
    df["ema_21_50_cross"] = (df["ema_21"] > df["ema_50"]).astype(int)

    # Price relative to EMAs
    df["price_vs_ema21"] = c / df["ema_21"] - 1
    df["price_vs_ema50"] = c / df["ema_50"] - 1

    # Trend slope (normalised)
    df["ema21_slope"] = df["ema_21"].pct_change(3)

    return df


# ─── Momentum ─────────────────────────────────────────────────────────────────

def _add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]

    # RSI
    for p in [7, 14, 21]:
        df[f"rsi_{p}"] = _rsi(c, p)

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]
    df["macd_cross"]  = (df["macd"] > df["macd_signal"]).astype(int)

    # Rate of change
    for p in [1, 3, 6, 12, 24]:
        df[f"roc_{p}"] = c.pct_change(p)

    # Stochastic oscillator
    for p in [14]:
        lo  = df["low"].rolling(p).min()
        hi  = df["high"].rolling(p).max()
        df[f"stoch_k_{p}"] = 100 * (c - lo) / (hi - lo + 1e-9)
        df[f"stoch_d_{p}"] = df[f"stoch_k_{p}"].rolling(3).mean()

    # Williams %R
    p = 14
    lo = df["low"].rolling(p).min()
    hi = df["high"].rolling(p).max()
    df["williams_r"] = -100 * (hi - c) / (hi - lo + 1e-9)

    return df


# ─── Volatility ───────────────────────────────────────────────────────────────

def _add_volatility(df: pd.DataFrame) -> pd.DataFrame:
    c  = df["close"]
    hi = df["high"]
    lo = df["low"]

    # Bollinger Bands
    for p in [20]:
        sma          = c.rolling(p).mean()
        std          = c.rolling(p).std()
        df[f"bb_upper_{p}"] = sma + 2 * std
        df[f"bb_lower_{p}"] = sma - 2 * std
        df[f"bb_mid_{p}"]   = sma
        df[f"bb_pct_{p}"]   = (c - df[f"bb_lower_{p}"]) / (df[f"bb_upper_{p}"] - df[f"bb_lower_{p}"] + 1e-9)
        df[f"bb_width_{p}"] = (df[f"bb_upper_{p}"] - df[f"bb_lower_{p}"]) / sma

    # ATR
    for p in [14]:
        df[f"atr_{p}"] = _atr(df, p)
        df[f"atr_pct_{p}"] = df[f"atr_{p}"] / c

    # Historical volatility
    for p in [24, 72]:
        df[f"hvol_{p}"] = c.pct_change().rolling(p).std() * np.sqrt(p)

    # High-Low range
    df["hl_range"]     = (hi - lo) / c
    df["hl_range_ma"]  = df["hl_range"].rolling(14).mean()

    return df


# ─── Volume ───────────────────────────────────────────────────────────────────

def _add_volume(df: pd.DataFrame) -> pd.DataFrame:
    v = df["volume"]
    c = df["close"]

    df["volume_ma20"]  = v.rolling(20).mean()
    df["volume_ratio"] = v / df["volume_ma20"]          # volume surge indicator
    df["obv"]          = (np.sign(c.diff()) * v).cumsum()
    df["obv_slope"]    = df["obv"].pct_change(5)

    # VWAP proxy (rolling)
    df["vwap_20"] = (c * v).rolling(20).sum() / v.rolling(20).sum()
    df["price_vs_vwap"] = c / df["vwap_20"] - 1

    return df


# ─── Candle patterns ──────────────────────────────────────────────────────────

def _add_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    o = df["open"]
    h = df["high"]
    l = df["low"]
    c = df["close"]

    body  = (c - o).abs()
    range_ = h - l + 1e-9

    df["body_pct"]      = body / range_         # body relative to candle range
    df["upper_shadow"]  = (h - c.clip(lower=o)) / range_
    df["lower_shadow"]  = (c.clip(upper=o) - l) / range_
    df["is_bullish"]    = (c > o).astype(int)

    # Doji
    df["doji"] = (body / range_ < 0.1).astype(int)

    return df


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - 100 / (1 + rs)


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    hi, lo, pc = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([hi - lo, (hi - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


# ─── SuperTrend ───────────────────────────────────────────────────────────────

def _add_supertrend(df: pd.DataFrame,
                    periods: list[int] | None = None,
                    multipliers: list[float] | None = None) -> pd.DataFrame:
    """
    SuperTrend indicator.  For each (period, multiplier) pair adds:
      st_{p}_{m}_dir   : +1 (bullish) / -1 (bearish)
      st_{p}_{m}_val   : the SuperTrend line value
      st_{p}_{m}_dist  : (close - st_val) / close  (signed distance)
      st_{p}_{m}_flip  : 1 if direction changed this bar, else 0
    """
    if periods is None:
        periods = [10, 14]
    if multipliers is None:
        multipliers = [2.0, 3.0]

    hi = df["high"].values
    lo = df["low"].values
    c  = df["close"].values
    n  = len(c)

    for period in periods:
        atr = _atr(df, period).values

        for mult in multipliers:
            suffix = f"st_{period}_{str(mult).replace('.', '')}"

            basic_upper = (hi + lo) / 2 + mult * atr
            basic_lower = (hi + lo) / 2 - mult * atr

            final_upper = np.zeros(n)
            final_lower = np.zeros(n)
            st_dir      = np.zeros(n, dtype=int)
            st_val      = np.zeros(n)

            # initialise first valid bar (need at least `period` bars for ATR)
            first = period
            final_upper[first] = basic_upper[first]
            final_lower[first] = basic_lower[first]
            st_dir[first]      = 1
            st_val[first]      = final_lower[first]

            for i in range(first + 1, n):
                # upper band only tightens unless price broke above it
                if c[i - 1] <= final_upper[i - 1]:
                    final_upper[i] = min(basic_upper[i], final_upper[i - 1])
                else:
                    final_upper[i] = basic_upper[i]

                # lower band only rises unless price broke below it
                if c[i - 1] >= final_lower[i - 1]:
                    final_lower[i] = max(basic_lower[i], final_lower[i - 1])
                else:
                    final_lower[i] = basic_lower[i]

                # direction flip logic
                if st_dir[i - 1] == 1:
                    st_dir[i] = -1 if c[i] < final_lower[i] else 1
                else:
                    st_dir[i] =  1 if c[i] > final_upper[i] else -1

                st_val[i] = final_lower[i] if st_dir[i] == 1 else final_upper[i]

            st_dir_s  = pd.Series(st_dir,  index=df.index, dtype=int)
            st_val_s  = pd.Series(st_val,  index=df.index)

            df[f"{suffix}_dir"]  = st_dir_s
            df[f"{suffix}_val"]  = st_val_s
            df[f"{suffix}_dist"] = (df["close"] - st_val_s) / df["close"]
            df[f"{suffix}_flip"] = (st_dir_s.diff().abs() > 0).astype(int)

    return df


# ─── MACD × SuperTrend composite signals ─────────────────────────────────────

def _add_macd_supertrend_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derived features that combine MACD and SuperTrend for the model.

    Assumes _add_momentum() and _add_supertrend() have already been called
    (so macd, macd_signal, macd_hist, macd_cross, st_10_30_dir, st_14_30_dir
    already exist).
    """
    # Primary SuperTrend direction column (10-period, 3.0×)
    st_dir = df["st_10_30_dir"]

    # ── MACD-based ───────────────────────────────────────────────────────────
    # Histogram momentum: rising / falling over last 2 bars
    df["macd_hist_slope"]    = df["macd_hist"].diff(2)
    df["macd_hist_accel"]    = df["macd_hist_slope"].diff()

    # MACD cross confirmed by SuperTrend (strong signal)
    df["macd_bull_confirmed"] = ((df["macd_cross"] == 1) & (st_dir == 1)).astype(int)
    df["macd_bear_confirmed"] = ((df["macd_cross"] == 0) & (st_dir == -1)).astype(int)

    # ── SuperTrend-based ─────────────────────────────────────────────────────
    # Agreement across both ST parameter sets
    st_dir2 = df["st_14_30_dir"]
    df["st_agree"]   = (st_dir == st_dir2).astype(int)   # 1 = both agree on direction
    df["st_net_dir"] = (st_dir + st_dir2) / 2            # -1 / 0 / +1 blended

    # Any ST flip this bar
    df["st_any_flip"] = ((df["st_10_30_flip"] == 1) | (df["st_14_30_flip"] == 1)).astype(int)

    # ── Combined alignment ───────────────────────────────────────────────────
    # +1 : MACD bullish + both STs bullish   -1 : all bearish   0 : mixed
    macd_dir = df["macd_cross"].map({1: 1, 0: -1})
    df["macd_st_alignment"] = ((macd_dir == 1) & (st_dir == 1) & (st_dir2 == 1)).astype(int) \
                             - ((macd_dir == -1) & (st_dir == -1) & (st_dir2 == -1)).astype(int)

    return df
