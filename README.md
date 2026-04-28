# 📈 AI Trading Signal Analyzer

A modular CLI tool that fetches **live cryptocurrency market data**, computes **technical indicators**, and generates a structured **AI trade signal** using Google Gemini via LangChain.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![LangChain](https://img.shields.io/badge/LangChain-1.2-green)
![Gemini](https://img.shields.io/badge/Gemini-2.0--Flash-orange)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-purple)

> ⚠️ **Disclaimer:** For educational purposes only. Not financial advice.

---

## 🏗️ Architecture

```
CoinGecko API ──► MarketData (schemas.py)
Fear & Greed API ─┘      │
                          ▼
                  indicators.py
                  (RSI, trend, momentum,
                   volume, volatility)
                          │
                          ▼
                  analyzer.py
                  (prompt | Gemini LLM
                   with_structured_output)
                          │
                          ▼
                  TradeSignal (schemas.py)
                  BUY / SELL / HOLD
                  + confidence + reasoning
                  + key levels + risk
```

## 📁 Project Structure

```
ai_trading_signal_analyzer/
├── src/
│   ├── __init__.py
│   ├── schemas.py      # Pydantic v2 models: MarketData, TechnicalIndicators, TradeSignal
│   ├── data.py         # CoinGecko + Fear & Greed API calls
│   ├── indicators.py   # RSI approx, trend, momentum, volume, volatility
│   ├── prompts.py      # ChatPromptTemplate for signal generation
│   ├── llm.py          # Gemini LLM initialisation (langchain-google-genai 4.x)
│   ├── analyzer.py     # LangChain chain: prompt | LLM.with_structured_output
│   └── main.py         # CLI entry point with formatted output
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

## ✨ Features

- **Live data** — CoinGecko price, volume, market cap, 7d/24h changes (no API key)
- **Sentiment** — Crypto Fear & Greed Index (no API key)
- **Technical indicators** — RSI approximation, trend classification, momentum score, volume signal, volatility
- **Structured AI output** — Pydantic v2 `TradeSignal` schema enforced via `with_structured_output`
- **CLI** — interactive coin selection or `--symbol BTC` argument
- **10 supported assets** — BTC, ETH, SOL, BNB, XRP, ADA, DOGE, AVAX, MATIC, LINK

## 🚀 Quick Start

```bash
git clone https://github.com/janahvijanardhan/ai-trading-signal-analyzer.git
cd ai-trading-signal-analyzer

pip install -r requirements.txt

# Add your key to .env
echo "GEMINI_API_KEY=your_key_here" > .env

# Run interactively
python -m src.main

# Or specify a symbol directly
python -m src.main --symbol ETH
```

## 📊 Example Output

```
=== AI Trading Signal Analyzer ===

  Bitcoin (BTC) — Live Market Data
  Price         :      $67,432.1200
  24h Change    :           +2.34%
  7d Change     :           +8.91%
  Fear & Greed  :          72 (Greed)

  Technical Indicators
  RSI-14 (approx) :        72.28
  Trend           :  Strong Uptrend
  Volume Signal   :  Normal Volume
  Momentum Score  :       +12.45
  Volatility      :       Medium

  AI SIGNAL: 🟢 BUY
  Confidence      : 74%
  Risk Level      : MEDIUM
  Timeframe       : swing

  Reasoning:
  Strong 7-day momentum combined with a greed reading of 72 suggests
  sustained buying pressure...
```

## 🔑 Key Design Decisions

| Decision | Reasoning |
|----------|-----------|
| `with_structured_output(TradeSignal)` | Enforces schema at model level, not post-hoc parsing |
| Pydantic v2 enums for signal/risk/timeframe | Prevents invalid values from reaching the caller |
| RSI approximation documented in field name | Transparent about limitations — no OHLCV history available |
| `build_prompt_input()` in `analyzer.py` | Keeps prompt template decoupled from data models |
| `--symbol` CLI arg + interactive fallback | Usable both in scripts and interactively |

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Google Gemini 2.0 Flash |
| LangChain | 1.2.x (latest) |
| Structured Output | Pydantic v2 + `with_structured_output` |
| Market Data | CoinGecko REST API (free, no key) |
| Sentiment | Alternative.me Fear & Greed API (free) |
| Data Models | Pydantic v2 BaseModel |

---