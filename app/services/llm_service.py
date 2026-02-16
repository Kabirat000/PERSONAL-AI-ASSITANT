"""
LLM Service using Groq's chat completion API (Llama 3.3 70B).
Groq uses an OpenAI-compatible API, so we use the openai SDK with a custom base_url.
"""
import logging

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.exceptions import LLMError

logger = logging.getLogger(__name__)

# Initialize Groq client (OpenAI-compatible API)
settings = get_settings()
client = OpenAI(
    api_key=settings.groq_api_key,
    base_url="https://api.groq.com/openai/v1",
)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def run_llm(system_prompt: str, user_input: str, temperature: float | None = None) -> str:
    """
    Run LLM inference using Groq's Llama 3.3 70B.

    Args:
        system_prompt: The system prompt defining AI behavior
        user_input: The user's input text
        temperature: Optional temperature override (0.0-1.0)

    Returns:
        The LLM's response text

    Raises:
        LLMError: If the API call fails after retries
    """
    try:
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            temperature=temperature or settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"LLM inference failed: {e}")
        raise LLMError(f"Failed to generate response: {str(e)}")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def run_llm_with_context(
    system_prompt: str,
    user_input: str,
    context: list[str] | None = None,
    temperature: float | None = None,
) -> str:
    """
    Run LLM inference with optional RAG context.

    Args:
        system_prompt: The system prompt defining AI behavior
        user_input: The user's input text
        context: Optional list of relevant context strings from RAG
        temperature: Optional temperature override (0.0-1.0)

    Returns:
        The LLM's response text
    """
    # Build context-enhanced prompt
    enhanced_prompt = system_prompt
    if context:
        context_block = "\n\n---\n**Relevant Context:**\n" + "\n".join(f"- {c}" for c in context)
        enhanced_prompt = system_prompt + context_block

    return run_llm(enhanced_prompt, user_input, temperature)


def run_llm_structured(
    system_prompt: str,
    user_input: str,
    response_format: dict | None = None,
) -> str:
    """
    Run LLM with structured output (JSON mode).

    Args:
        system_prompt: The system prompt defining AI behavior
        user_input: The user's input text
        response_format: Optional response format specification

    Returns:
        The LLM's response text (expected to be valid JSON)
    """
    try:
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            temperature=0.1,  # Lower temperature for structured output
            max_tokens=settings.llm_max_tokens,
            response_format={"type": "json_object"} if response_format else None,
        )
        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"Structured LLM inference failed: {e}")
        raise LLMError(f"Failed to generate structured response: {str(e)}")
