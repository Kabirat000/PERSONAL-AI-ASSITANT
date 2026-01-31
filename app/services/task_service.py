import json
from app.services.llm_service import run_llm  # Correct import
from app.prompts.prompt import load_prompt  # Correct import path

def extract_tasks(thought: str) -> list[dict]:
    # Load the prompt with the thought (user input)
    prompt = load_prompt(
        "task_extract_prompt.txt",  # Ensure the prompt file exists
        thought=thought  # Pass thought as a keyword argument
    )

    # Call the LLM to extract tasks
    response = run_llm(prompt, thought)

    try:
        # Try to parse the response from the LLM
        data = json.loads(response)
        return data.get("tasks", [])
    except json.JSONDecodeError:
        # If response is not valid JSON, return an empty list
        return []
import json
from app.services.llm_service import run_llm  # Correct import
from app.prompts.prompt import load_prompt  # Correct import path

def extract_tasks(thought: str) -> list[dict]:
    # Load the prompt with the thought (user input)
    prompt = load_prompt(
        "task_extract_prompt.txt",  # Ensure the prompt file exists
        thought=thought  # Pass thought as a keyword argument
    )

    # Call the LLM to extract tasks
    response = run_llm(prompt, thought)

    try:
        # Try to parse the response from the LLM
        data = json.loads(response)
        return data.get("tasks", [])
    except json.JSONDecodeError:
        # If response is not valid JSON, return an empty list
        return []
