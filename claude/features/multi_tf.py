"""
Multi-timeframe feature builder.

메인(15분) + 서브(5분) 캔들 데이터를 합쳐 하나의 피처 행렬을 만든다.
서브 타임프레임 지표는 메인 캔들 시간 기준으로 정렬(forward-fill 후 마지막 값 사용).
"""

import pandas as pd
from features.indicators import add_all_indicators
from utils.logger import get_logger

log = get_logger(__name__)


def build_multi_tf_features(df_main: pd.DataFrame, df_sub: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    df_main : 15분 OHLCV DataFrame  (index = datetime UTC)
    df_sub  :  5분 OHLCV DataFrame  (index = datetime UTC)

    Returns
    -------
    DataFrame: 메인 타임프레임 기준, 서브 지표가 'sub_' prefix로 추가됨
    """

    # 메인 지표
    df_main_feat = add_all_indicators(df_main.copy())

    # 서브 지표
    df_sub_feat  = add_all_indicators(df_sub.copy())

    # 서브 컬럼에 prefix 추가 (ohlcv 제외)
    ohlcv = {"open", "high", "low", "close", "volume"}
    sub_cols = {c: f"sub_{c}" for c in df_sub_feat.columns if c not in ohlcv}
    df_sub_feat = df_sub_feat.rename(columns=sub_cols)
    df_sub_feat = df_sub_feat.drop(columns=[c for c in ohlcv if c in df_sub_feat.columns])

    # 서브를 메인 시간에 맞춰 정렬 (각 메인 캔들 시각 직전의 서브 값 사용)
    df_sub_feat = df_sub_feat.sort_index()
    df_main_feat = df_main_feat.sort_index()

    df_merged = pd.merge_asof(
        df_main_feat.reset_index(),
        df_sub_feat.reset_index().rename(columns={"datetime": "datetime_sub"}),
        left_on="datetime",
        right_on="datetime_sub",
        direction="backward",
    )
    df_merged = df_merged.set_index("datetime").drop(columns=["datetime_sub"], errors="ignore")
    df_merged.dropna(inplace=True)

    return df_merged
