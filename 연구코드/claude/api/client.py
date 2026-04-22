"""
OrangeX REST API client with OAuth 2.0 authentication.

Auth flow:
  POST /public/auth  →  { access_token, refresh_token, expires_in }
  Use Bearer token on all /private/* endpoints.
"""

import time
import requests
from utils.logger import get_logger
import config

log = get_logger(__name__)

_TOKEN_CACHE: dict = {}   # { "token": str, "expires_at": float }


class OrangeXClient:
    def __init__(self, api_key: str = config.API_KEY, api_secret: str = config.API_SECRET):
        self.api_key    = api_key
        self.api_secret = api_secret
        self.base_url   = config.BASE_URL
        self.session    = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    # ─── Auth ─────────────────────────────────────────────────────────────────

    def _get_token(self) -> str:
        """Return a valid access token, refreshing if necessary."""
        now = time.time()
        if _TOKEN_CACHE.get("token") and _TOKEN_CACHE.get("expires_at", 0) > now + 30:
            return _TOKEN_CACHE["token"]

        if not self.api_key or not self.api_secret:
            raise ValueError("API credentials not set. Add ORANGEX_API_KEY / ORANGEX_API_SECRET to .env")

        resp = self._request("GET", "/public/auth", params={
            "grant_type":    "client_credentials",
            "client_id":     self.api_key,
            "client_secret": self.api_secret,
        }, auth=False)

        token      = resp["result"]["access_token"]
        expires_in = resp["result"].get("expires_in", 900)
        _TOKEN_CACHE["token"]      = token
        _TOKEN_CACHE["expires_at"] = now + expires_in
        log.info("OrangeX token refreshed, expires in %ds", expires_in)
        return token

    # ─── Core request ─────────────────────────────────────────────────────────

    def _request(self, method: str, path: str, params: dict = None,
                 json: dict = None, auth: bool = True) -> dict:
        url     = self.base_url + path
        headers = {}
        if auth:
            headers["Authorization"] = f"Bearer {self._get_token()}"

        resp = self.session.request(method, url, params=params, json=json, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"OrangeX API error: {data['error']}")
        return data

    # ─── Public market data ───────────────────────────────────────────────────

    def get_instruments(self, currency: str = None, kind: str = None) -> list:
        params = {}
        if currency:
            params["currency"] = currency
        if kind:
            params["kind"] = kind
        data = self._request("GET", "/public/get_instruments", params=params or None, auth=False)
        return data.get("result", [])

    def get_ticker(self, instrument_name: str) -> dict:
        data = self._request("GET", "/public/tickers",
                             params={"instrument_name": instrument_name}, auth=False)
        result = data.get("result", {})
        # OrangeX returns result as a list with one element
        if isinstance(result, list):
            return result[0] if result else {}
        return result

    def get_order_book(self, instrument_name: str, depth: int = 5) -> dict:
        data = self._request("GET", "/public/get_order_book",
                             params={"instrument_name": instrument_name, "depth": depth}, auth=False)
        result = data.get("result", {})
        if isinstance(result, list):
            return result[0] if result else {}
        return result

    # ─── OHLCV (requires auth) ────────────────────────────────────────────────

    def get_ohlcv(self, instrument_name: str, resolution: str,
                  start_ts: int, end_ts: int) -> dict:
        """
        Returns raw API response dict with keys: tick, open, high, low, close, volume, cost
        All arrays of equal length. tick values are Unix seconds.
        """
        data = self._request("GET", "/private/get_tradingview_chart_data", params={
            "instrument_name": instrument_name,
            "resolution":      resolution,
            "start_timestamp": start_ts,
            "end_timestamp":   end_ts,
        })
        return data.get("result", {})

    # ─── Private account/trading ──────────────────────────────────────────────

    def get_positions(self, currency: str = "PERPETUAL", kind: str = "perpetual") -> list:
        params = {"currency": currency}
        if kind:
            params["kind"] = kind
        data = self._request("GET", "/private/get_positions", params=params)
        return data.get("result", [])

    def get_assets_info(self, asset_type: str = "PERPETUAL") -> dict:
        data = self._request("GET", "/private/get_assets_info",
                             params={"asset_type": asset_type})
        return data.get("result", {})

    def get_account_summary(self, currency: str = "USDT") -> dict:
        """계좌 잔고 조회. 우선 PERPETUAL 자산 정보를 사용하고, 실패 시 기존 방식으로 fallback."""
        try:
            result = self.get_assets_info("PERPETUAL")
            if isinstance(result, dict):
                for key in ("PERPETUAL", "perpetual"):
                    if key in result and isinstance(result[key], dict):
                        return result[key]
                if "available_funds" in result or "wallet_balance" in result:
                    return result
        except Exception as e:
            log.warning("get_assets_info 잔고 조회 실패, fallback 사용: %s", e)

        data = self._request("GET", "/private/get_subaccounts")
        result = data.get("result", [])
        if isinstance(result, list) and result:
            account = result[0]
            total = float(account.get("assetTotal", 0))
            return {"available_funds": total, "wallet_balance": total}
        return {}

    def place_order(self, instrument_name: str, side: str, amount: float,
                    order_type: str = "market", price: float = None,
                    reduce_only: bool = False, position_side: str = None) -> dict:
        endpoint = "/private/buy" if side == "buy" else "/private/sell"
        params   = {
            "instrument_name": instrument_name,
            "amount":          amount,
            "type":            order_type,
            "reduce_only":     reduce_only,
        }
        if price and order_type == "limit":
            params["price"] = price
        if position_side:
            params["position_side"] = position_side
        data = self._request("GET", endpoint, params=params)
        return data.get("result", {})

    def cancel_order(self, order_id: str) -> dict:
        data = self._request("GET", "/private/cancel",
                             params={"order_id": order_id})
        return data.get("result", {})

    def get_perpetual_config(self, instrument_name: str) -> dict:
        data = self._request("GET", "/private/get_perpetual_user_config",
                             params={"instrument_name": instrument_name})
        result = data.get("result", {})
        if isinstance(result, list):
            return result[0] if result else {}
        return result

    def set_leverage(self, instrument_name: str, leverage: int) -> dict:
        data = self._request("GET", "/private/adjust_perpetual_leverage", params={
            "instrument_name": instrument_name,
            "leverage":        leverage,
        })
        return data.get("result", {})
