# app/prompts/prompts.py
def load_prompt(prompt_file: str, **kwargs) -> str:
    try:
        with open(prompt_file, 'r') as file:
            prompt = file.read()

        # Replace placeholders in the prompt with provided values (kwargs)
        for key, value in kwargs.items():
            prompt = prompt.replace(f"{{{{ {key} }}}}", value)

        return prompt
    except FileNotFoundError:
        return "Prompt file not found."
