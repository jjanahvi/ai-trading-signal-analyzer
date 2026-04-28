from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable

from src.schemas import MarketData, TechnicalIndicators, TradeSignal
from src.prompts import build_signal_prompt


def build_analyzer_chain(llm: BaseChatModel) -> Runnable:
    """
    Build the LangChain runnable chain for trade signal generation.

    The chain connects the signal prompt template to the LLM with
    structured output constrained to the TradeSignal Pydantic schema.
    Using with_structured_output ensures the model returns valid,
    parsed TradeSignal instances rather than raw text.

    In langchain-google-genai >= 4.0, with_structured_output uses
    Gemini's native JSON schema mode (response_json_schema) by default,
    which is more reliable than function-calling-based extraction.

    Args:
        llm: A configured BaseChatModel instance.

    Returns:
        A LangChain Runnable that accepts a prompt input dict and
        returns a TradeSignal instance.
    """
    prompt = build_signal_prompt()
    structured_llm = llm.with_structured_output(TradeSignal)
    return prompt | structured_llm


def build_prompt_input(
    market_data: MarketData,
    indicators: TechnicalIndicators,
) -> dict:
    """
    Assemble the input dictionary required by the signal prompt template.

    Flattens MarketData and TechnicalIndicators into a single dict
    whose keys match the prompt template's input variables exactly.

    Args:
        market_data: A populated MarketData instance.
        indicators: A populated TechnicalIndicators instance.

    Returns:
        A dictionary mapping prompt variable names to their values.
    """
    return {
        "symbol": market_data.symbol,
        "name": market_data.name,
        "price_usd": market_data.price_usd,
        "change_24h_pct": market_data.change_24h_pct,
        "change_7d_pct": market_data.change_7d_pct,
        "volume_24h_usd": market_data.volume_24h_usd,
        "market_cap_usd": market_data.market_cap_usd,
        "fear_greed_value": market_data.fear_greed_value,
        "fear_greed_label": market_data.fear_greed_label,
        "rsi_14_approx": indicators.rsi_14_approx,
        "trend": indicators.trend,
        "volume_signal": indicators.volume_signal,
        "momentum_score": indicators.momentum_score,
        "volatility_label": indicators.volatility_label,
    }


def generate_signal(
    chain: Runnable,
    market_data: MarketData,
    indicators: TechnicalIndicators,
) -> TradeSignal:
    """
    Run the analyzer chain and return a validated TradeSignal.

    Builds the prompt input from the provided data, invokes the
    chain, and returns the parsed and validated TradeSignal.

    Args:
        chain: The analyzer chain built by build_analyzer_chain.
        market_data: A populated MarketData instance.
        indicators: A populated TechnicalIndicators instance.

    Returns:
        A validated TradeSignal instance.

    Raises:
        Exception: If the LLM call fails or the output does not
                   conform to the TradeSignal schema.
    """
    prompt_input = build_prompt_input(market_data, indicators)
    result = chain.invoke(prompt_input)
    return result