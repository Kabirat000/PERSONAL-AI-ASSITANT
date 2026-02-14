"""
Idea processing service.
Core RAG flow for transforming messy thoughts into structured output.
"""
import json
import logging
from pathlib import Path
from typing import Any

from app.services.llm_service import run_llm, run_llm_with_context
from app.rag.retriever import store_idea, retrieve_similar_ideas

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "refine_prompt.txt"


def process_idea(raw_text: str, store_in_memory: bool = True) -> dict[str, Any]:
    """
    Core RAG flow for idea processing:
    1. Retrieve similar past ideas for context
    2. Generate structured output using LLM with context
    3. Store the new idea in vector memory
    4. Return structured JSON response

    Args:
        raw_text: The raw, unstructured thought from user
        store_in_memory: Whether to store this idea for future recall

    Returns:
        Structured dict with clean_note, themes, and suggested_tasks
    """
    # Load system prompt
    try:
        system_prompt = PROMPT_PATH.read_text()
    except FileNotFoundError:
        logger.error(f"Prompt file not found: {PROMPT_PATH}")
        return {"error": "System configuration error", "raw_output": None}

    # 1️⃣ MEMORY READ — retrieve similar ideas for context
    related_ideas = retrieve_similar_ideas(raw_text)
    logger.info(f"Retrieved {len(related_ideas)} related ideas for context")

    # 2️⃣ GENERATION WITH CONTEXT
    llm_output = run_llm_with_context(
        system_prompt=system_prompt,
        user_input=raw_text,
        context=related_ideas if related_ideas else None,
    )

    # 3️⃣ MEMORY WRITE — store current idea for future recall
    if store_in_memory:
        doc_id = store_idea(raw_text, metadata={"source": "user_input"})
        logger.info(f"Stored idea with ID: {doc_id}")

    # 4️⃣ PARSE AND RETURN STRUCTURED OUTPUT
    try:
        result = json.loads(llm_output)
        result["context_used"] = len(related_ideas) > 0
        result["related_ideas_count"] = len(related_ideas)
        return result
    except json.JSONDecodeError:
        logger.warning(f"LLM returned invalid JSON: {llm_output[:200]}")
        return {
            "error": "LLM returned invalid JSON",
            "raw_output": llm_output,
            "context_used": len(related_ideas) > 0,
        }


def process_idea_without_memory(raw_text: str) -> dict[str, Any]:
    """
    Process idea without storing in memory or using context.
    Useful for one-off queries.

    Args:
        raw_text: The raw thought from user

    Returns:
        Structured dict with clean_note, themes, and suggested_tasks
    """
    try:
        system_prompt = PROMPT_PATH.read_text()
    except FileNotFoundError:
        return {"error": "System configuration error"}

    llm_output = run_llm(system_prompt, raw_text)

    try:
        return json.loads(llm_output)
    except json.JSONDecodeError:
        return {"error": "LLM returned invalid JSON", "raw_output": llm_output}
