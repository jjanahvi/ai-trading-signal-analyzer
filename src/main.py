"""
AI Trading Signal Analyzer — Command Line Interface

Fetches live market data for a selected cryptocurrency, computes
technical indicators, and generates a structured AI trade signal
using Google Gemini via LangChain.

Supported symbols: BTC, ETH, SOL, BNB, XRP, ADA, DOGE, AVAX, MATIC, LINK

Usage:
    python -m src.main
    python -m src.main --symbol ETH

Environment:
    GEMINI_API_KEY must be set in .env or exported before running.

Disclaimer:
    This tool is for educational purposes only and does not constitute
    financial advice. Always conduct your own research before trading.
"""
import argparse
import sys

from dotenv import load_dotenv

from src.llm import load_api_key, build_llm
from src.data import build_market_data, SYMBOL_TO_ID
from src.indicators import compute_indicators
from src.analyzer import build_analyzer_chain, generate_signal
from src.schemas import MarketData, TechnicalIndicators, TradeSignal


SUPPORTED_SYMBOLS = list(SYMBOL_TO_ID.keys())


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        A Namespace containing the parsed --symbol argument.
    """
    parser = argparse.ArgumentParser(
        description="AI Trading Signal Analyzer — powered by Gemini + LangChain"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help=f"Ticker symbol to analyse. Supported: {', '.join(SUPPORTED_SYMBOLS)}",
    )
    return parser.parse_args()


def prompt_symbol_selection() -> str:
    """
    Interactively prompt the user to select a coin symbol.

    Displays a numbered list of supported symbols and reads the
    user's selection. Returns the selected symbol string.

    Returns:
        An uppercase ticker symbol string selected by the user.
    """
    print("\nSupported assets:")
    for i, symbol in enumerate(SUPPORTED_SYMBOLS, start=1):
        print(f"  {i:2}. {symbol}")

    while True:
        choice = input("\nEnter symbol or number: ").strip().upper()

        if choice in SUPPORTED_SYMBOLS:
            return choice

        if choice.isdigit():
            index = int(choice) - 1
            if 0 <= index < len(SUPPORTED_SYMBOLS):
                return SUPPORTED_SYMBOLS[index]

        print(f"Invalid selection. Choose from: {', '.join(SUPPORTED_SYMBOLS)}")


def print_market_data(data: MarketData) -> None:
    """
    Print formatted market data to stdout.

    Args:
        data: A populated MarketData instance.
    """
    print(f"\n{'=' * 55}")
    print(f"  {data.name} ({data.symbol}) — Live Market Data")
    print(f"{'=' * 55}")
    print(f"  Price         : ${data.price_usd:>15,.4f}")
    print(f"  24h Change    : {data.change_24h_pct:>+14.2f}%")
    print(f"  7d Change     : {data.change_7d_pct:>+14.2f}%")
    print(f"  Volume (24h)  : ${data.volume_24h_usd:>15,.0f}")
    print(f"  Market Cap    : ${data.market_cap_usd:>15,.0f}")
    print(f"  Fear & Greed  : {data.fear_greed_value:>15} ({data.fear_greed_label})")


def print_indicators(ind: TechnicalIndicators) -> None:
    """
    Print formatted technical indicators to stdout.

    Args:
        ind: A populated TechnicalIndicators instance.
    """
    print(f"\n{'─' * 55}")
    print("  Technical Indicators")
    print(f"{'─' * 55}")
    print(f"  RSI-14 (approx) : {ind.rsi_14_approx:>12.2f}")
    print(f"  Trend           : {ind.trend:>12}")
    print(f"  Volume Signal   : {ind.volume_signal:>12}")
    print(f"  Momentum Score  : {ind.momentum_score:>+12.2f}")
    print(f"  Volatility      : {ind.volatility_label:>12}")


def print_signal(signal: TradeSignal) -> None:
    """
    Print the formatted AI trade signal to stdout.

    Args:
        signal: A validated TradeSignal instance.
    """
    signal_icons = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
    icon = signal_icons.get(signal.signal.value, "⚪")

    print(f"\n{'=' * 55}")
    print(f"  AI SIGNAL: {icon} {signal.signal.value}")
    print(f"{'=' * 55}")
    print(f"  Confidence      : {signal.confidence}%")
    print(f"  Risk Level      : {signal.risk_level.value}")
    print(f"  Timeframe       : {signal.suggested_timeframe.value}")
    print(f"\n  Reasoning:")
    print(f"  {signal.reasoning}")
    print(f"\n  Key Factors:")
    for factor in signal.key_factors:
        print(f"    • {factor}")
    print(f"\n  Key Levels:")
    print(f"    Support    : {signal.key_levels.support}")
    print(f"    Resistance : {signal.key_levels.resistance}")
    print(f"\n  Market Context:")
    print(f"  {signal.market_context}")
    print(f"\n{'─' * 55}")
    print("  ⚠  Disclaimer: Educational use only. Not financial advice.")
    print(f"{'─' * 55}\n")


def run() -> None:
    """
    Execute the full signal analysis pipeline.

    Loads environment, resolves the target symbol, fetches market
    data, computes indicators, generates the AI signal, and prints
    the formatted results.

    Returns:
        None
    """
    load_dotenv()

    print("=== AI Trading Signal Analyzer ===")
    print("Powered by Gemini + LangChain\n")

    args = parse_args()
    symbol = args.symbol.upper() if args.symbol else prompt_symbol_selection()

    print(f"\nFetching live data for {symbol}...")
    try:
        market_data = build_market_data(symbol)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Failed to fetch market data: {e}")
        sys.exit(1)

    print_market_data(market_data)

    print("\nComputing technical indicators...")
    indicators = compute_indicators(market_data)
    print_indicators(indicators)

    print("\nGenerating AI signal...")
    try:
        load_api_key()
        llm = build_llm()
        chain = build_analyzer_chain(llm)
        signal = generate_signal(chain, market_data, indicators)
    except EnvironmentError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Signal generation failed: {e}")
        sys.exit(1)

    print_signal(signal)


if __name__ == "__main__":
    run()