import os
import openai
from dotenv import load_dotenv

load_dotenv()

# Optimization: Limit Tesseract's internal multi-threading when we use parallel page processing.
# This prevents CPU over-saturation and context switching overhead.
os.environ["OMP_THREAD_LIMIT"] = "1"

client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_KEY")
)

MODEL = "google/gemini-2.0-flash-001"
#MODEL = "google/gemma-3-27b-it:free"
#MODEL = "meta-llama/llama-3.3-70b-instruct:free"


def translate_to_nepali(text_input: str | list[str]) -> tuple[str, str]:
    """
    Translates Himalayan languages to Nepali using OpenRouter LLM.
    Supports both single strings and lists of strings (for parallel processing).
    """
    if isinstance(text_input, list):
        return translate_pages_parallel(text_input)

    return _call_llm(text_input)


def _call_llm(text: str) -> tuple[str, str]:
    """Helper for a single LLM call."""
    if not text.strip():
        return "", MODEL

    system_prompt = """
You are a professional Himalayan language expert. Translate Tamang or Newari (Devanagari) into fluent Nepali.
- Keep English and Nepali text EXACTLY as is.
- Preserve original formatting and newlines.
- Return ONLY the translated text.
"""

    messages = []
    if any(m in MODEL.lower() for m in ["gemma", "llama"]):
        combined_content = f"{system_prompt.strip()}\n\n{text}"
        messages = [{"role": "user", "content": combined_content}]
    else:
        messages = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": text}
        ]

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.1
        )
        return response.choices[0].message.content.strip(), MODEL
    except Exception as e:
        print(f"LLM Error: {e}")
        return text, MODEL  # Fallback to original text on error


def translate_pages_parallel(pages: list[str]) -> tuple[str, str]:
    """
    Translates multiple pages/blocks in parallel to reduce total latency.
    """
    from concurrent.futures import ThreadPoolExecutor

    # Limit max workers to avoid hitting rate limits or crashing
    max_workers = min(len(pages), 5)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_call_llm, pages))

    # results is a list of (text, model) tuples
    translated_texts = [res[0] for res in results]
    combined_text = "\n\n".join(translated_texts)
    
    return combined_text, MODEL
