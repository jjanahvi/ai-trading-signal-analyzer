import requests

from src.schemas import MarketData


COINGECKO_BASE = "https://api.coingecko.com/api/v3"
FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1"

SYMBOL_TO_ID: dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "BNB": "binancecoin",
    "XRP": "ripple",
    "ADA": "cardano",
    "DOGE": "dogecoin",
    "AVAX": "avalanche-2",
    "MATIC": "matic-network",
    "LINK": "chainlink",
}


def resolve_coin_id(symbol: str) -> str:
    """
    Resolve a ticker symbol to its CoinGecko coin ID.

    Args:
        symbol: Uppercase ticker symbol, e.g. 'BTC'.

    Returns:
        The CoinGecko coin ID string.

    Raises:
        ValueError: If the symbol is not found in the lookup table.
    """
    coin_id = SYMBOL_TO_ID.get(symbol.upper())
    if not coin_id:
        supported = ", ".join(SYMBOL_TO_ID.keys())
        raise ValueError(
            f"Symbol '{symbol}' is not supported. "
            f"Supported symbols: {supported}."
        )
    return coin_id


def fetch_coin_data(coin_id: str) -> dict:
    """
    Fetch current price, volume, market cap, and percentage changes
    for a coin from the CoinGecko /coins/{id} endpoint.

    Args:
        coin_id: CoinGecko coin identifier, e.g. 'bitcoin'.

    Returns:
        A dictionary containing the raw API response for the coin.

    Raises:
        requests.RequestException: On network or HTTP errors.
        KeyError: If the expected fields are absent from the response.
    """
    url = f"{COINGECKO_BASE}/coins/{coin_id}"
    params = {
        "localization": "false",
        "tickers": "false",
        "community_data": "false",
        "developer_data": "false",
    }
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def fetch_fear_greed() -> tuple[int, str]:
    """
    Fetch the current Crypto Fear & Greed Index from alternative.me.

    The index ranges from 0 (Extreme Fear) to 100 (Extreme Greed)
    and provides a sentiment signal complementary to price data.

    Returns:
        A tuple of (index_value, index_label), e.g. (72, 'Greed').

    Raises:
        requests.RequestException: On network or HTTP errors.
    """
    response = requests.get(FEAR_GREED_URL, timeout=10)
    response.raise_for_status()
    data = response.json()["data"][0]
    return int(data["value"]), data["value_classification"]


def build_market_data(symbol: str) -> MarketData:
    """
    Fetch and assemble a MarketData instance for the given symbol.

    Orchestrates calls to CoinGecko and the Fear & Greed API,
    extracting and normalising the fields required for technical
    analysis and signal generation.

    Args:
        symbol: Uppercase ticker symbol, e.g. 'BTC'.

    Returns:
        A populated MarketData instance.

    Raises:
        ValueError: If the symbol is not supported.
        requests.RequestException: On any API network failure.
    """
    coin_id = resolve_coin_id(symbol)
    raw = fetch_coin_data(coin_id)
    fg_value, fg_label = fetch_fear_greed()

    market = raw["market_data"]

    return MarketData(
        symbol=symbol.upper(),
        name=raw["name"],
        price_usd=market["current_price"]["usd"],
        change_24h_pct=market["price_change_percentage_24h"] or 0.0,
        change_7d_pct=market["price_change_percentage_7d"] or 0.0,
        volume_24h_usd=market["total_volume"]["usd"],
        market_cap_usd=market["market_cap"]["usd"],
        fear_greed_value=fg_value,
        fear_greed_label=fg_label,
    )