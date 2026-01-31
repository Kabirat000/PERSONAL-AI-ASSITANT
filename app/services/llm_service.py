import os
import openai
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")

def run_llm(system_prompt: str, user_input: str) -> str:
    # Make the API call to OpenAI for task extraction
    response = openai.ChatCompletion.create(
        model="gpt-4.0-turbo",  # Use the appropriate model
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        temperature=0.3,  # Adjust temperature for creativity
    )

    # Return the content of the LLM response
    return response['choices'][0]['message']['content']
