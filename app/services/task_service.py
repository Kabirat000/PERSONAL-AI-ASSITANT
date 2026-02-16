"""
Task extraction service.
Extracts actionable tasks from messy thoughts using LLM.
"""
import json
import logging
from pathlib import Path
from typing import Any

from app.services.llm_service import run_llm_structured

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "task_extract_prompt.txt"


def extract_tasks(thought: str) -> list[dict[str, Any]]:
    """
    Extract actionable tasks from a messy thought.

    Args:
        thought: Raw user thought/input text

    Returns:
        List of task dicts with 'task' and 'priority' keys
    """
    try:
        system_prompt = PROMPT_PATH.read_text()
    except FileNotFoundError:
        logger.error(f"Prompt file not found: {PROMPT_PATH}")
        return []

    # Inject the thought into the prompt template
    formatted_prompt = system_prompt.replace("{thought}", thought)

    try:
        response = run_llm_structured(
            system_prompt=formatted_prompt,
            user_input=thought,
        )
        data = json.loads(response)
        return data.get("tasks", [])
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse task response: {e}")
        return []
    except Exception as e:
        logger.error(f"Task extraction failed: {e}")
        return []


def extract_tasks_with_context(
    thought: str,
    context: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Extract tasks with optional context from related ideas.

    Args:
        thought: Raw user thought/input text
        context: Optional list of related context strings

    Returns:
        List of task dicts
    """
    try:
        system_prompt = PROMPT_PATH.read_text()
    except FileNotFoundError:
        return []

    formatted_prompt = system_prompt.replace("{thought}", thought)

    if context:
        context_block = "\n\nContext from related notes:\n" + "\n".join(f"- {c}" for c in context)
        formatted_prompt += context_block

    try:
        response = run_llm_structured(
            system_prompt=formatted_prompt,
            user_input=thought,
        )
        data = json.loads(response)
        return data.get("tasks", [])
    except Exception as e:
        logger.error(f"Task extraction with context failed: {e}")
        return []
