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
    Translates text between Himalayan languages.
    OPTIMIZATION: If a single long string is provided, it is chunked into paragraphs 
    and translated in parallel to hit the < 5s latency target.
    """
    if (isinstance(text_input, list)):
        # Already split by pages, run page-parallel
        return translate_parallel_chunks(text_input, source_lang, target_lang, return_list=False)

    # For a single page/string, check if it's long enough to benefit from chunking
    # Average 300 words is the threshold where parallelization starts saving time
    word_count = len(text_input.split())
    if word_count > 300:
        chunks = _split_into_chunks(text_input)
        # Pass the full original text as context to each chunk to preserve accuracy
        return translate_parallel_chunks(chunks, source_lang, target_lang, full_context=text_input)

    return _call_llm(text_input, source_lang, target_lang)


def _split_into_chunks(text: str, target_word_count: int = 250) -> list[str]:
    """Splits text into logical chunks by paragraph breaks."""
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_count = 0

    for p in paragraphs:
        p_words = len(p.split())
        if current_count + p_words > target_word_count and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [p]
            current_count = p_words
        else:
            current_chunk.append(p)
            current_count += p_words

    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks


def _call_llm(text: str, source_lang: str, target_lang: str, full_context: str = None) -> tuple[str, str]:
    """Helper for a single LLM call with context-awareness for chunked processing."""
    if not text.strip():
        return "", MODEL

    # Define supported Himalayan languages that need specialized Devanagari handling
    HIMALAYAN_LANGS = {"tamang", "newari"}
    source_parts = [p.strip().lower() for p in source_lang.replace("/", " ").replace(",", " ").split()]
    is_himalayan = any(part in HIMALAYAN_LANGS for part in source_parts)

    if full_context:
        # Prompt for chunked translation where accuracy depends on seeing the whole doc
        system_prompt = f"""
You are a high-precision Translation & OCR Restoration Engine.
PRIMARY TASK: Translate the provided snippet from {source_lang} into {target_lang}.

ACCURACY CONTEXT:
I will provide the FULL ORIGINAL DOCUMENT below. Use it ONLY as a reference to:
1. Fix OCR errors in the snippet (e.g. cross-check digits vs written words across the whole doc).
2. Understand the technical context/subject matter.

RULE: Output ONLY the translation for the specific SNIPPET provided. DO NOT re-translate the full document.

-- FULL DOCUMENT FOR REFERENCE --
{full_context}
-- END FULL DOCUMENT --

Step 1: RESTORE SNIPPET (Background)
- Fix broken Devanagari in the snippet using the Full Document context.
Step 2: TRANSLATE SNIPPET (Primary)
- Translate into fluent, formal {target_lang}.
- Maintain labels like "विषय :-" and English contact info exactly as-is.

Output Requirements:
- Output ONLY the final, polished {target_lang} translation of the snippet. 
- NO explanations, NO labels.
"""
    else:
        # Standard prompt for full-page or short-text translation
        system_prompt = f"""
You are a high-precision Translation & OCR Restoration Engine specializing in Himalayan languages (Nepali, Nepal Bhasa/Newari, Tamang).
Your primary goal is to produce a flawless, professional translation into {target_lang}.

CORE OBJECTIVE:
Translate the provided text from {source_lang} into {target_lang}.

Step 1: Background OCR Restoration (Mental Step)
- Silently repair character-level OCR errors in the source Devanagari (e.g., missing matras, disconnected Shirorekas, or fragmented words like "काठमाडौा" → "काठमाडौँ").
- Remove scanner noise, vertical bars ($|$), or digital artifacts.
- DO NOT invent or insert English words like 'cache' or 'nCache'.

Step 1.5: Numeric & Data Integrity
- Cross-verify Devanagari digits with written-out words (e.g., "रु. ४०,००,०००/-" + "दश लाख मात्र" → correct to "रु. १०,००,०००/-").
- Trust the written-out words over potentially misread digits (१↔४, ६↔९, ३↔८, ०↔६).
- Apply this to dates (मिति), reference numbers (चलानी नं.), and currency.

Step 2: Professional {target_lang} Translation (PRIMARY FOCUS)
- Transform the restored text into fluent, formal, and grammatically perfect {target_lang}.
- Cultural & Contextual Nuance: Use high-level administrative, legal, or social vocabulary appropriate for the document type.
- Language Separation: Ensure NO {source_lang} vocabulary leaks into the {target_lang} output unless it's a proper noun.
- Format Preservation: Maintain the original document's structure, labels, and hierarchy.
- English Preservation: Keep technical English terms and all contact details (Tel/Email/Website) exactly as-is.

Output Requirements:
- Output ONLY the final, polished {target_lang} text.
- NO preamble, NO explanations, NO labels like "Translated Text:".
- If content is absolutely illegible, keep the best guess or use [...].
"""

    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": f"SNIPPET TO TRANSLATE:\n{text}"}
    ]

    try:
        # OpenRouter-specific optimizations
        extra_headers = {
            "HTTP-Referer": "https://neptext-ocr.ai", 
            "X-Title": "NepText OCR Pipeline",
        }
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.1,
            extra_headers=extra_headers
        )
        return response.choices[0].message.content.strip(), MODEL
    except Exception as e:
        print(f"LLM Error: {e}")
        return text, MODEL


def translate_parallel_chunks(chunks: list[str], source_lang: str, target_lang: str, return_list: bool = False, full_context: str = None) -> tuple[str | list[str], str]:
    """Translates multiple chunks/pages in parallel to hit the < 5s target."""
    from concurrent.futures import ThreadPoolExecutor

    # Higher thread count for small chunks since I/O bound
    max_workers = min(len(chunks), 8)
    
    def translate_single(chunk):
        return _call_llm(chunk, source_lang, target_lang, full_context=full_context)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(translate_single, chunks))

    translated_texts = [res[0] for res in results]
    
    if return_list:
        return translated_texts, MODEL
        
    combined_text = "\n\n".join(translated_texts)
    return combined_text, MODEL


def detect_language(text: str) -> dict:
    """
    Detects the language of a given text snippet using LLM.
    """
    # Truncate to first 500 characters for efficiency to save tokens
    snippet = text[:500].strip()
    if not snippet:
        return {"language": "Unknown", "code": "unknown", "confidence": 0.0}

    system_prompt = """
You are a highly capable AI language identification expert. Your task is to accurately detect the language of the provided text.
You can detect any global language (e.g., English, Spanish, Chinese, French, Hindi, etc.).
Crucially, you have specialized expertise in distinguishing Himalayan languages. If the text is in Devanagari script, pay extremely close attention to the nuances to accurately distinguish whether it is Tamang, Newari, or Nepali.

Rules:
1. Output ONLY a valid JSON object in this format: {"language": "Language Name", "code": "lang_code", "confidence": 0.95}
2. Language codes for global languages: Use standard 3-letter ISO codes (e.g., 'eng' for English, 'spa' for Spanish, 'zho' for Chinese).
3. Language codes for Himalayan languages: strictly use 'tam' for Tamang, 'new' for Newari, and 'nep' for Nepali.
4. If the language is completely unrecognizable, use code 'ot' and Language Name 'Other'.
5. Use a confidence score between 0.0 and 1.0.
6. Output ONLY the JSON. No preamble, no markdown blocks, no talk.
"""

    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": snippet}
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.1
        )
        content = response.choices[0].message.content.strip()
        
        # Strip markdown code blocks if the model included them
        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()
            
        import json
        return json.loads(content)
    except Exception as e:
        print(f"Language detection failed: {e}")
        return {"language": "Detection Error", "code": "error", "confidence": 0.0}
