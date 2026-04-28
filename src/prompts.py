from langchain_core.prompts import ChatPromptTemplate


SYSTEM_PROMPT = """You are a professional cryptocurrency market analyst with deep expertise \
in technical analysis, market microstructure, and risk management.

Your role is to synthesize market data and technical indicators into a structured \
trade signal. You must be objective, data-driven, and concise.

IMPORTANT CONSTRAINTS:
- Base your signal ONLY on the data provided. Do not use prior knowledge of specific \
price levels unless derived from the data.
- Acknowledge uncertainty clearly in your confidence score.
- Never recommend high confidence on a HOLD signal.
- Your reasoning must directly reference the indicators provided.
- This analysis is for educational purposes. Always factor in risk."""


USER_PROMPT = """Analyse the following market data and technical indicators, then generate \
a structured trade signal.

=== MARKET DATA ===
Asset        : {symbol} ({name})
Price (USD)  : ${price_usd:,.4f}
24h Change   : {change_24h_pct:.2f}%
7d Change    : {change_7d_pct:.2f}%
Volume 24h   : ${volume_24h_usd:,.0f}
Market Cap   : ${market_cap_usd:,.0f}

=== SENTIMENT ===
Fear & Greed Index : {fear_greed_value} / 100 ({fear_greed_label})

=== TECHNICAL INDICATORS ===
RSI-14 (approx) : {rsi_14_approx}
Trend           : {trend}
Volume Signal   : {volume_signal}
Momentum Score  : {momentum_score} (range: -100 to +100)
Volatility      : {volatility_label}

Generate your trade signal now."""


def build_signal_prompt() -> ChatPromptTemplate:
    """
    Build the ChatPromptTemplate for trade signal generation.

    The template combines a system message defining the analyst
    persona and constraints, with a user message containing the
    fully formatted market data and indicators.

    The input variables must be populated with values from a
    MarketData instance and a TechnicalIndicators instance before
    invoking the chain.

    Returns:
        A ChatPromptTemplate ready for use in the analyzer chain.
    """
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT),
    ])