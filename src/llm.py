import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel


def load_api_key() -> str:
    """
    Load the Gemini API key from the environment.

    In langchain-google-genai >= 4.0, the SDK checks GOOGLE_API_KEY
    first, then GEMINI_API_KEY as a fallback. This function validates
    presence explicitly to produce a clear error rather than a cryptic
    SDK error.

    Returns:
        The API key string.

    Raises:
        EnvironmentError: If neither key is found in the environment.
    """
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise EnvironmentError(
            "Neither GOOGLE_API_KEY nor GEMINI_API_KEY is set. "
            "Add one to your .env file."
        )
    # Ensure SDK picks it up under the name it checks first
    os.environ.setdefault("GOOGLE_API_KEY", key)
    return key


def build_llm(
    model: str = "gemini-2.0-flash-lite",
    temperature: float = 0.1,
) -> BaseChatModel:
    """
    Instantiate a Gemini chat model for structured signal generation.

    In langchain-google-genai >= 4.0, the API key is read automatically
    from the GEMINI_API_KEY or GOOGLE_API_KEY environment variable.
    The convert_system_message_to_human parameter is no longer needed
    and has been removed.

    A low temperature is used because signal generation requires
    consistent, deterministic structured output. Higher temperatures
    increase variance in JSON field values.

    Args:
        model: Gemini model identifier. Defaults to gemini-2.0-flash.
        temperature: Sampling temperature. Must be in [0.0, 1.0].

    Returns:
        A configured BaseChatModel instance.
    """
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_retries=3,
    )