import json
from pathlib import Path

from app.services.llm_service import run_llm
from app.rag.retriever import store_idea, retrieve_similar_ideas

PROMPT_PATH = Path("app/prompts/refine_prompt.txt")


def process_idea(raw_text: str):
    """
    Core RAG flow:
    1. Retrieve similar past ideas
    2. Inject them as context
    3. Generate refined output
    4. Store the new idea in memory
    """

    system_prompt = PROMPT_PATH.read_text()

    # 1️⃣ MEMORY READ — retrieve similar ideas
    related_ideas = retrieve_similar_ideas(raw_text)

    if related_ideas:
        memory_block = (
            "\n\nRelated past ideas:\n"
            + "\n".join(f"- {idea}" for idea in related_ideas)
        )
    else:
        memory_block = ""

    # 2️⃣ GENERATION WITH CONTEXT
    llm_output = run_llm(
        system_prompt + memory_block,
        raw_text
    )

    # 3️⃣ MEMORY WRITE — store current idea
    store_idea(raw_text)

    # 4️⃣ RETURN STRUCTURED OUTPUT
    try:
        return json.loads(llm_output)
    except json.JSONDecodeError:
        return {
            "error": "LLM returned invalid JSON",
            "raw_output": llm_output
        }
import json
from pathlib import Path

from app.services.llm_service import run_llm
from app.rag.retriever import store_idea, retrieve_similar_ideas

PROMPT_PATH = Path("app/prompts/refine_prompt.txt")


def process_idea(raw_text: str):
    """
    Core RAG flow:
    1. Retrieve similar past ideas
    2. Inject them as context
    3. Generate refined output
    4. Store the new idea in memory
    """

    system_prompt = PROMPT_PATH.read_text()

    # 1️⃣ MEMORY READ — retrieve similar ideas
    related_ideas = retrieve_similar_ideas(raw_text)

    if related_ideas:
        memory_block = (
            "\n\nRelated past ideas:\n"
            + "\n".join(f"- {idea}" for idea in related_ideas)
        )
    else:
        memory_block = ""

    # 2️⃣ GENERATION WITH CONTEXT
    llm_output = run_llm(
        system_prompt + memory_block,
        raw_text
    )

    # 3️⃣ MEMORY WRITE — store current idea
    store_idea(raw_text)

    # 4️⃣ RETURN STRUCTURED OUTPUT
    try:
        return json.loads(llm_output)
    except json.JSONDecodeError:
        return {
            "error": "LLM returned invalid JSON",
            "raw_output": llm_output
        }


