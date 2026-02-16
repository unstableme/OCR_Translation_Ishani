import os
import openai
from dotenv import load_dotenv

load_dotenv()

client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_KEY")
)

MODEL = "google/gemini-2.0-flash-001"
#MODEL = "google/gemma-3-27b-it:free"
#MODEL = "meta-llama/llama-3.3-70b-instruct:free"


def translate_tamang_to_nepali(extracted_text: str) -> tuple[str, str]:
    """
    Translates Tamang text into Nepali using OpenRouter LLM.

    Returns
    -------
    tuple[str, str]
        (translated_text, model_used)
    """

    system_prompt = """
You are a professional Tamang → Nepali translation engine.

Your task is to translate the input text from Tamang to Nepali, while preserving any existing English or Nepali text exactly as it is.

INSTRUCTIONS:
1. **Identify Languages**: Scan the input text.
   - If a sentence or phrase is in **English**, keep it EXACTLY as is.
   - If a sentence or phrase is in **Nepali**, keep it EXACTLY as is.
   - If a sentence or phrase is in **Tamang**, translate it to **fluent Nepali**.

2. **Translation Rules (for Tamang text)**:
   - Follow Tamang grammar and meaning strictly.
   - Produce natural-sounding Nepali.
   - Do NOT add explanations, notes, or extra text.
   - Do NOT correct, guess, or expand unclear content.

3. **Output**:
   - Return only the final text with the mixed Nepali/English content.
   - Maintain the original formatting (newlines, spacing) as much as possible.
"""

    messages = []
    if any(m in MODEL.lower() for m in ["gemma", "llama"]):
        # Gemma models on some providers do not support 'system' role
        combined_content = f"{system_prompt.strip()}\n\n{extracted_text}"
        messages = [
            {"role": "user", "content": combined_content}
        ]
    else:
        messages = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": extracted_text}
        ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.1
    )

    translated_text = response.choices[0].message.content.strip()
    return translated_text, MODEL
