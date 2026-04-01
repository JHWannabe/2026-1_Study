"""
OrangeX Auto-Trading Configuration
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── API Credentials ─────────────────────────────────────────────────────────
API_KEY    = os.getenv("ORANGEX_API_KEY", "")
API_SECRET = os.getenv("ORANGEX_API_SECRET", "")

# ─── API Endpoints ───────────────────────────────────────────────────────────
BASE_URL   = "https://api.orangex.com/api/v1"
WS_URL     = "wss://api.orangex.com/ws/api/v1"

# ─── Trading Parameters ──────────────────────────────────────────────────────
INSTRUMENT      = "ETH-USDT-PERPETUAL"   # default instrument
LEVERAGE        = 35                     # 15x leverage
RESOLUTION      = "5"                  # main candle: 5-minute
SUB_RESOLUTION  = "15"                   # sub candle: 15-minute

# ─── Model Parameters ────────────────────────────────────────────────────────
LOOK_BACK          = 30          # number of candles for feature window
PREDICT_HORIZON    = 3           # predict N candles ahead
PROFIT_THRESHOLD   = 0.003       # 0.3% move = signal (before leverage)
TRAIN_RATIO        = 0.7
VALID_RATIO        = 0.15

# ─── Backtest Parameters ─────────────────────────────────────────────────────
INITIAL_CAPITAL    = 20          # USD (virtual)
MAX_POSITION_PCT   = 0.95        # use 95% of capital per trade
STOP_LOSS_PCT      = round(0.20 / 35, 6)   # -20% on margin at 35x ≈ 0.005714 on price
TAKE_PROFIT_PCT    = round(0.20 / 35, 6)   # +20% on margin at 35x ≈ 0.005714 on price
TAKER_FEE          = 0.0006      # 0.06% taker fee (OrangeX 실제값)
MAKER_FEE          = 0.0002      # 0.02% maker fee
EXIT_CONFIDENCE    = 0.35        # close open positions when confidence is 0.35 or below

# ─── Data ────────────────────────────────────────────────────────────────────
DATA_DIR   = "data"
MODEL_DIR  = "model/saved"
