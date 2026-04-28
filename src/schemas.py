from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SignalType(str, Enum):
    """Enumeration of possible trade signal directions."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class RiskLevel(str, Enum):
    """Enumeration of risk levels assigned to a signal."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Timeframe(str, Enum):
    """Enumeration of suggested trading timeframes."""

    SCALP = "scalp"
    INTRADAY = "intraday"
    SWING = "swing"


class MarketData(BaseModel):
    """
    Raw market data fetched from external APIs for a single asset.

    Attributes:
        symbol: Ticker symbol, e.g. 'BTC' or 'ETH'.
        name: Full asset name, e.g. 'Bitcoin'.
        price_usd: Current price in US dollars.
        change_24h_pct: Percentage price change over the last 24 hours.
        change_7d_pct: Percentage price change over the last 7 days.
        volume_24h_usd: Total trading volume in USD over the last 24 hours.
        market_cap_usd: Current market capitalisation in USD.
        fear_greed_value: Crypto Fear & Greed Index value (0–100).
        fear_greed_label: Human-readable label for the Fear & Greed value.
    """

    symbol: str
    name: str
    price_usd: float
    change_24h_pct: float
    change_7d_pct: float
    volume_24h_usd: float
    market_cap_usd: float
    fear_greed_value: int
    fear_greed_label: str


class TechnicalIndicators(BaseModel):
    """
    Technical indicators computed from raw market data.

    All indicators are derived from the data available in MarketData
    without requiring historical OHLCV series, using approximations
    suitable for a signal-generation pipeline.

    Attributes:
        rsi_14_approx: Approximated RSI-14 based on 24h and 7d price changes.
        trend: Directional trend label derived from moving average comparison.
        volume_signal: Volume trend signal relative to recent average.
        momentum_score: Composite momentum score in range [-100, 100].
        volatility_label: Qualitative volatility classification.
    """

    rsi_14_approx: float = Field(ge=0.0, le=100.0)
    trend: str
    volume_signal: str
    momentum_score: float = Field(ge=-100.0, le=100.0)
    volatility_label: str


class KeyLevels(BaseModel):
    """
    Estimated key price levels for a trade signal.

    Attributes:
        support: Estimated support price level as a string (may be 'N/A').
        resistance: Estimated resistance price level as a string (may be 'N/A').
    """

    support: str
    resistance: str


class TradeSignal(BaseModel):
    """
    Structured output produced by the AI signal analyzer.

    This model is used both as the LLM structured output schema and
    as the final result returned to the caller.

    Attributes:
        signal: The recommended trade direction (BUY / SELL / HOLD).
        confidence: Analyst confidence in the signal, 0–100.
        risk_level: Assessed risk level for the signal.
        suggested_timeframe: Recommended trading timeframe.
        reasoning: Two-to-three sentence explanation of the signal.
        key_factors: List of the most influential factors driving the signal.
        key_levels: Estimated support and resistance price levels.
        market_context: One-sentence summary of broader market conditions.
    """

    signal: SignalType
    confidence: int = Field(ge=0, le=100)
    risk_level: RiskLevel
    suggested_timeframe: Timeframe
    reasoning: str
    key_factors: list[str] = Field(min_length=2, max_length=5)
    key_levels: KeyLevels
    market_context: str