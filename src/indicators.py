from src.schemas import MarketData, TechnicalIndicators


def compute_rsi_approximation(change_24h: float, change_7d: float) -> float:
    """
    Approximate RSI-14 from 24h and 7d percentage changes.

    Without a full OHLCV history, RSI is approximated by mapping
    recent price momentum onto the RSI scale. The 7-day change is
    weighted at 70% and the 24-hour change at 30%, then scaled and
    clamped to the [0, 100] RSI range.

    This is a deliberate approximation — full RSI requires 14
    consecutive candles. The approximation is clearly documented
    in the schema field name (rsi_14_approx).

    Args:
        change_24h: Percentage price change over the last 24 hours.
        change_7d: Percentage price change over the last 7 days.

    Returns:
        An approximated RSI value clamped to [0.0, 100.0].
    """
    weighted_change = (change_7d * 0.7) + (change_24h * 0.3)
    # Map [-20%, +20%] range to [0, 100] RSI scale
    rsi = 50.0 + (weighted_change * 2.5)
    return round(max(0.0, min(100.0, rsi)), 2)


def compute_trend(change_24h: float, change_7d: float) -> str:
    """
    Classify the price trend from short- and medium-term changes.

    Compares the 24h and 7d price changes to identify whether price
    action is accelerating upward, decelerating, or falling.

    Args:
        change_24h: Percentage price change over the last 24 hours.
        change_7d: Percentage price change over the last 7 days.

    Returns:
        A trend label string: one of 'Strong Uptrend', 'Uptrend',
        'Neutral', 'Downtrend', or 'Strong Downtrend'.
    """
    avg = (change_24h + change_7d) / 2

    if avg > 5:
        return "Strong Uptrend"
    elif avg > 1:
        return "Uptrend"
    elif avg < -5:
        return "Strong Downtrend"
    elif avg < -1:
        return "Downtrend"
    else:
        return "Neutral"


def compute_volume_signal(volume_24h: float, market_cap: float) -> str:
    """
    Assess the volume level relative to market capitalisation.

    Volume-to-market-cap ratio is used as a proxy for volume
    relative to the asset's size. A high ratio suggests elevated
    activity compared to the asset's value; a low ratio suggests
    low liquidity or interest.

    Args:
        volume_24h: 24-hour trading volume in USD.
        market_cap: Total market capitalisation in USD.

    Returns:
        A volume signal string: 'High Volume', 'Normal Volume',
        or 'Low Volume'.
    """
    if market_cap == 0:
        return "Unknown"

    ratio = volume_24h / market_cap

    if ratio > 0.15:
        return "High Volume"
    elif ratio > 0.04:
        return "Normal Volume"
    else:
        return "Low Volume"


def compute_momentum_score(
    change_24h: float,
    change_7d: float,
    fear_greed: int,
) -> float:
    """
    Compute a composite momentum score in the range [-100, 100].

    Combines price momentum (24h and 7d changes) with market
    sentiment (Fear & Greed index). Price momentum is weighted
    at 70% and sentiment at 30%.

    Args:
        change_24h: Percentage price change over the last 24 hours.
        change_7d: Percentage price change over the last 7 days.
        fear_greed: Fear & Greed index value from 0 to 100.

    Returns:
        A composite momentum score clamped to [-100.0, 100.0].
    """
    price_momentum = (change_24h * 0.4) + (change_7d * 0.3)
    # Normalise fear/greed from [0, 100] to [-50, 50]
    sentiment_component = (fear_greed - 50) * 0.3
    score = price_momentum + sentiment_component
    return round(max(-100.0, min(100.0, score)), 2)


def compute_volatility_label(change_24h: float) -> str:
    """
    Classify volatility based on the absolute 24-hour price change.

    Uses absolute percentage change as a proxy for daily volatility.
    Thresholds are calibrated for the crypto market, which operates
    with higher baseline volatility than traditional assets.

    Args:
        change_24h: Percentage price change over the last 24 hours.

    Returns:
        A volatility label: 'High', 'Medium', or 'Low'.
    """
    abs_change = abs(change_24h)

    if abs_change > 8:
        return "High"
    elif abs_change > 3:
        return "Medium"
    else:
        return "Low"


def compute_indicators(market_data: MarketData) -> TechnicalIndicators:
    """
    Compute all technical indicators from a MarketData instance.

    Orchestrates all individual indicator computations and returns
    a fully populated TechnicalIndicators instance.

    Args:
        market_data: A populated MarketData instance.

    Returns:
        A TechnicalIndicators instance with all fields computed.
    """
    return TechnicalIndicators(
        rsi_14_approx=compute_rsi_approximation(
            market_data.change_24h_pct,
            market_data.change_7d_pct,
        ),
        trend=compute_trend(
            market_data.change_24h_pct,
            market_data.change_7d_pct,
        ),
        volume_signal=compute_volume_signal(
            market_data.volume_24h_usd,
            market_data.market_cap_usd,
        ),
        momentum_score=compute_momentum_score(
            market_data.change_24h_pct,
            market_data.change_7d_pct,
            market_data.fear_greed_value,
        ),
        volatility_label=compute_volatility_label(market_data.change_24h_pct),
    )