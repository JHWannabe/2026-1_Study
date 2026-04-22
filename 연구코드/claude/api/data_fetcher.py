"""
Fetch and cache OHLCV data from OrangeX.
Saves/loads from CSV so you don't re-fetch the same data repeatedly.
"""

import os
import time
import pandas as pd
from pathlib import Path
from api.client import OrangeXClient
from utils.logger import get_logger
import config

log = get_logger(__name__)

RESOLUTION_MINUTES = {
    "1": 1, "3": 3, "5": 5, "10": 10, "15": 15, "30": 30,
    "60": 60, "120": 120, "180": 180, "240": 240, "360": 360,
    "720": 720, "D": 1440,
}


def fetch_ohlcv(
    instrument: str = config.INSTRUMENT,
    resolution: str = config.RESOLUTION,
    days: int = 365,
    client: OrangeXClient = None,
    use_cache: bool = True,
    end_date: str | None = None,   # "YYYY-MM-DD" 형식. None이면 현재 시각 사용
) -> pd.DataFrame:
    """
    Download OHLCV data and return a clean DataFrame.

    Columns: open, high, low, close, volume  (index = datetime UTC)

    Parameters
    ----------
    end_date : 종료 날짜 ("YYYY-MM-DD"). None이면 현재 시각까지 조회.
               예: "2025-12-31" → 2025-12-31 23:59:59 UTC 까지만 수집.
    """
    # 캐시 파일명에 종료 날짜 포함 (날짜가 다를 때 재사용 방지)
    date_tag   = f"_{end_date}" if end_date else ""
    cache_path = Path(config.DATA_DIR) / f"{instrument}_{resolution}_{days}d{date_tag}.csv"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if use_cache and cache_path.exists():
        log.info("Loading cached data from %s", cache_path)
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df

    if client is None:
        client = OrangeXClient()

    if end_date:
        end_ts = int(pd.Timestamp(end_date, tz="UTC").replace(
            hour=23, minute=59, second=59).timestamp())
    else:
        end_ts = int(time.time())

    start_ts = end_ts - days * 86400

    # OrangeX limits the number of candles per request; chunk if needed
    res_min  = RESOLUTION_MINUTES.get(str(resolution), 60)
    max_bars = 500
    chunk_s  = max_bars * res_min * 60

    all_rows = []
    cur_start = start_ts

    while cur_start < end_ts:
        cur_end = min(cur_start + chunk_s, end_ts)
        log.info("Fetching %s %s  %s → %s", instrument, resolution,
                 _ts_fmt(cur_start), _ts_fmt(cur_end))
        try:
            raw = client.get_ohlcv(instrument, str(resolution), cur_start, cur_end)
        except Exception as e:
            log.error("Fetch error: %s", e)
            break

        log.debug("Raw OHLCV type: %s  keys: %s", type(raw).__name__,
                  list(raw.keys()) if isinstance(raw, dict) else f"len={len(raw)}")

        # OrangeX may return dict-of-arrays or list-of-dicts
        if isinstance(raw, dict):
            ticks = raw.get("ticks") or raw.get("tick", [])
            if not ticks:
                break
            for i, ts in enumerate(ticks):
                all_rows.append({
                    "timestamp": ts,
                    "open":   raw["open"][i],
                    "high":   raw["high"][i],
                    "low":    raw["low"][i],
                    "close":  raw["close"][i],
                    "volume": raw.get("volume", [0] * len(ticks))[i],
                })
        elif isinstance(raw, list):
            if not raw:
                break
            for candle in raw:
                all_rows.append({
                    "timestamp": candle.get("tick") or candle.get("time") or candle.get("timestamp"),
                    "open":   candle.get("open"),
                    "high":   candle.get("high"),
                    "low":    candle.get("low"),
                    "close":  candle.get("close"),
                    "volume": candle.get("volume", 0),
                })
        else:
            log.error("Unexpected OHLCV response type: %s", type(raw))
            break

        if all_rows:
            cur_start = all_rows[-1]["timestamp"] + res_min * 60
        else:
            break
        time.sleep(0.3)

    if not all_rows:
        log.warning("No data fetched.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df.drop_duplicates("timestamp", inplace=True)
    df.sort_values("timestamp", inplace=True)

    # API가 숫자를 문자열로 반환하는 경우 float 변환
    for col in ["open", "high", "low", "close", "volume", "timestamp"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df.set_index("datetime", inplace=True)
    df.drop(columns=["timestamp"], inplace=True)

    df.to_csv(cache_path)
    log.info("Saved %d bars to %s", len(df), cache_path)
    return df


def _ts_fmt(ts: int) -> str:
    return pd.Timestamp(ts, unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M")
