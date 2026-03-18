import os
import openai
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

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



def translate_text(text_input: str | list[str], source_lang: str = "Tamang/Newari", target_lang: str = "Nepali") -> tuple[str | list[str], str]:
    """
    Translates between Himalayan languages using OpenRouter LLM.
    Supports both single strings and lists of strings.
    If a list is provided, a list is returned.
    """
    if isinstance(text_input, list):
        return translate_pages_parallel(text_input, source_lang, target_lang, return_list=True)

    return _call_llm(text_input, source_lang, target_lang)


def _call_llm(text: str, source_lang: str, target_lang: str) -> tuple[str, str]:
    """Helper for a single LLM call."""
    if not text.strip():
        return "", MODEL

    # Define supported Himalayan languages that need specialized Devanagari handling
    HIMALAYAN_LANGS = {"tamang", "newari"}
    
    # Check if the source language (or any part of it if separated by / or ,) is a Himalayan language
    source_parts = [p.strip().lower() for p in source_lang.replace("/", " ").replace(",", " ").split()]
    is_himalayan = any(part in HIMALAYAN_LANGS for part in source_parts)

    if is_himalayan:
        source_description = "Tamang or Newari written in Devanagari"
    else:
        source_description = source_lang

    system_prompt = f"""
You are a careful, single-output translation engine for Himalayan languages.

Objective:
- Translate everything into fluent {target_lang}.
- Output ONLY the final translated text. No comparisons, no labels, no talk.

Rules:
1) Translate ALL non-English text that is not already in fluent {target_lang} into {target_lang}.
2) Preserve structure EXACTLY: same line breaks, spacing, punctuation, and symbols.
3) Keep English segments character-for-character.
4) Do NOT provide multiple translations (e.g., do not show Tamang AND Newari AND Nepali).
5) Do NOT include labels like "Tamang:", "Newari:", "Nepali:", or "English:".
6) If a segment is truly unclear, keep it as-is.
7) Output ONLY the result. No explanations. No extra lines.
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


def translate_pages_parallel(pages: list[str], source_lang: str, target_lang: str, return_list: bool = False) -> tuple[str | list[str], str]:
    """
    Translates multiple pages/blocks in parallel to reduce total latency.
    """
    from concurrent.futures import ThreadPoolExecutor

    # Limit max workers to avoid hitting rate limits or crashing
    max_workers = min(len(pages), 5)
    
    def translate_single(page):
        return _call_llm(page, source_lang, target_lang)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(translate_single, pages))

    # results is a list of (text, model) tuples
    translated_texts = [res[0] for res in results]
    
    if return_list:
        return translated_texts, MODEL
        
    combined_text = "\n\n".join(translated_texts)
    return combined_text, MODEL
