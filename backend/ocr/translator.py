import os
import re
import time
import openai
from google import genai
from google.genai import types
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Optimization: Limit Tesseract's internal multi-threading when we use parallel page processing.
# This prevents CPU over-saturation and context switching overhead.
os.environ["OMP_THREAD_LIMIT"] = "1"

# Commented out Groq OpenAI client setup in favor of Google Gemini 3.5 Flash
# client = openai.OpenAI(
#     base_url="https://api.groq.com/openai/v1",
#     api_key=os.getenv("GROQ_API_KEY")
# )
# MODEL = "llama-3.3-70b-versatile"

# Initialize Google GenAI client
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    api_key = os.getenv("GOOGLE_API_KEY")

# Ensure empty strings are treated as None so client uses default env lookup or handles properly
if api_key == "":
    api_key = None

try:
    if api_key:
        gemini_client = genai.Client(api_key=api_key)
    else:
        print("WARNING: Google GenAI Client initialization skipped: GEMINI_API_KEY is not set.")
        gemini_client = None
except Exception as e:
    print(f"WARNING: Google GenAI Client initialization failed: {e}")
    gemini_client = None

MODEL = "gemini-3.5-flash"
PIVOT_LANGUAGE = "Nepali"
PIVOTABLE_NON_NEPALI_LANGS = {"tamang", "newari"}
LANGUAGE_ALIASES = {
    "nepali": "nepali",
    "ne": "nepali",
    "tamang": "tamang",
    "tam": "tamang",
    "newari": "newari",
    "newar": "newari",
    "new": "newari",
    "nepal bhasa": "newari",
    "nepal bhasa newari": "newari",
    "newari nepal bhasa": "newari",
    "nepalbhasa": "newari",
}
LANGUAGE_DISPLAY_NAMES = {
    "nepali": "Nepali",
    "tamang": "Tamang",
    "newari": "Nepal Bhasa (Newari)",
}

class LanguageDetectionResult(BaseModel):
    language: str
    code: str
    confidence: float


def _canonical_lang_key(language: str) -> str:
    """Map UI/API language labels to stable keys for routing."""
    normalized = " ".join(
        (language or "")
        .lower()
        .replace("(", " ")
        .replace(")", " ")
        .replace("/", " ")
        .replace(",", " ")
        .replace("+", " ")
        .replace("_", " ")
        .replace("-", " ")
        .split()
    )

    has_tamang = "tamang" in normalized
    has_newari = "newari" in normalized or "nepal bhasa" in normalized
    if has_tamang and has_newari:
        return "auto"
    if has_tamang:
        return "tamang"
    if has_newari:
        return "newari"

    compact = normalized.replace(" ", "")
    return LANGUAGE_ALIASES.get(normalized) or LANGUAGE_ALIASES.get(compact) or normalized


def _display_lang(language: str) -> str:
    key = _canonical_lang_key(language)
    return LANGUAGE_DISPLAY_NAMES.get(key, (language or "").strip() or "Unknown")


def _should_translate_via_nepali(source_lang: str, target_lang: str) -> bool:
    source_key = _canonical_lang_key(source_lang)
    target_key = _canonical_lang_key(target_lang)
    return (
        source_key in PIVOTABLE_NON_NEPALI_LANGS
        and target_key in PIVOTABLE_NON_NEPALI_LANGS
        and source_key != target_key
    )


def _combine_model_names(*model_names: str) -> str:
    unique = [name for name in dict.fromkeys(model_names) if name]
    return " -> ".join(unique) if unique else MODEL


def _target_language_guard(target_lang: str) -> str:
    target_key = _canonical_lang_key(target_lang)
    if target_key == "tamang":
        return """
CRITICAL TARGET LANGUAGE CONTROL:
- The final output MUST be Tamang, written in Devanagari script.
- Do NOT output Nepali unless the source contains an official Nepali proper noun, address, or quoted phrase.
- Nepali is allowed only as meaning/context support, never as the final translation language.
- If the input is Nepali from an intermediate pivot step, translate it fully into Tamang; do not polish, summarize, or return the Nepali input.
- Prefer Tamang vocabulary and sentence structure. Avoid standard Nepali phrasing except for proper nouns, official names, or quoted source text.
""".strip()
    if target_key == "newari":
        return """
CRITICAL TARGET LANGUAGE CONTROL:
- The final output MUST be Nepal Bhasa/Newari, written in Devanagari script.
- Do NOT output Nepali unless the source contains an official Nepali proper noun, address, or quoted phrase.
- Nepali is allowed only as meaning/context support, never as the final translation language.
- If the input is Nepali from an intermediate pivot step, translate it fully into Nepal Bhasa/Newari; do not polish, summarize, or return the Nepali input.
- Prefer Nepal Bhasa/Newari vocabulary and sentence structure. Avoid standard Nepali phrasing except for proper nouns, official names, or quoted source text.
""".strip()
    if target_key == "nepali":
        return """
CRITICAL TARGET LANGUAGE CONTROL:
- The final output MUST be Nepali, written in Devanagari script.
- Keep Tamang or Nepal Bhasa/Newari words only when they are proper nouns or quoted terms.
""".strip()
    return ""


def _translate_via_nepali(
    text_input: str | list[str],
    source_lang: str,
    target_lang: str,
) -> tuple[str | list[str], str]:
    source_display = _display_lang(source_lang)
    target_display = _display_lang(target_lang)
    print(
        f"Using Nepali pivot translation: {source_display} -> "
        f"{PIVOT_LANGUAGE} -> {target_display}"
    )

    nepali_text, first_model = _translate_direct(text_input, source_display, PIVOT_LANGUAGE)
    final_text, second_model = _translate_direct(nepali_text, PIVOT_LANGUAGE, target_display)
    return final_text, f"nepali_pivot:{_combine_model_names(first_model, second_model)}"


def _translate_direct(
    text_input: str | list[str],
    source_lang: str,
    target_lang: str,
) -> tuple[str | list[str], str]:
    source_display = _display_lang(source_lang)
    target_display = _display_lang(target_lang)

    if isinstance(text_input, list):
        # Already split by pages, run page-parallel
        return translate_parallel_chunks(
            text_input,
            source_display,
            target_display,
            return_list=False,
        )

    # For a single page/string, check if it's long enough to benefit from chunking.
    # Pasted text is often one large block, unlike OCR uploads which arrive as a
    # list of pages and are already translated in parallel.
    word_count = len(text_input.split())
    if word_count > 300:
        chunks = _split_into_chunks(text_input)
        print(
            "Translation chunking: "
            f"single text with {word_count} words split into {len(chunks)} chunks"
        )
        return translate_parallel_chunks(
            chunks,
            source_display,
            target_display,
            full_context=_build_compact_context(text_input, chunks),
        )

    return _call_llm(text_input, source_display, target_display)


def translate_text(text_input: str | list[str], source_lang: str = "Tamang/Newari", target_lang: str = "Nepali") -> tuple[str | list[str], str]:
    """
    Translates text between Himalayan languages.
    OPTIMIZATION: If a single long string is provided, it is chunked into paragraphs 
    and translated in parallel to hit the < 5s latency target.
    """
    if _should_translate_via_nepali(source_lang, target_lang):
        return _translate_via_nepali(text_input, source_lang, target_lang)

    return _translate_direct(text_input, source_lang, target_lang)


def _split_into_chunks(text: str, target_word_count: int = 220) -> list[str]:
    """Split text into bounded chunks even when pasted as one large block."""
    units = _split_text_units(text)
    chunks = []
    current_chunk = []
    current_count = 0

    for unit in units:
        unit = unit.strip()
        if not unit:
            continue

        unit_words = len(unit.split())
        if unit_words > target_word_count:
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_count = 0
            chunks.extend(_split_large_unit(unit, target_word_count))
        elif current_count + unit_words > target_word_count and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [unit]
            current_count = unit_words
        else:
            current_chunk.append(unit)
            current_count += unit_words

    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    return chunks


def _split_text_units(text: str) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if len(paragraphs) > 1:
        return paragraphs

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines

    sentences = [
        part.strip()
        for part in re.split(r"(?<=[।.!?])\s+", text)
        if part.strip()
    ]
    return sentences or [text]


def _split_large_unit(text: str, target_word_count: int) -> list[str]:
    words = text.split()
    if len(words) <= target_word_count:
        return [text]

    chunks = []
    for start in range(0, len(words), target_word_count):
        chunks.append(" ".join(words[start:start + target_word_count]))
    return chunks


def _build_compact_context(text: str, chunks: list[str], max_chars: int = 1800) -> str | None:
    if len(chunks) <= 1:
        return None

    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned

    half = max_chars // 2
    return f"{cleaned[:half]} ... {cleaned[-half:]}"


def _call_llm(text: str, source_lang: str, target_lang: str, full_context: str | None = None) -> tuple[str, str]:
    """Helper for a single LLM call with context-awareness for chunked processing."""
    if not text.strip():
        return "", MODEL

    request_started = time.time()
    input_words = len(text.split())
    context_chars = len(full_context or "")
    print(
        "LLM request start: "
        f"source={source_lang}, target={target_lang}, "
        f"chars={len(text)}, words={input_words}, context_chars={context_chars}"
    )

    target_guard = _target_language_guard(target_lang)

    if full_context:
        # Prompt for chunked translation where accuracy depends on seeing the whole doc
        system_prompt = f"""
You are a high-precision Translation & OCR Restoration Engine.
PRIMARY TASK: Translate the provided snippet from {source_lang} into {target_lang}.

{target_guard}

IMPORTANT: If the target language ({target_lang}) is English or any non-Devanagari language, SKIP all OCR restoration steps below and ONLY produce a direct, fluent translation into {target_lang}. The restoration steps below apply ONLY when the output is in a Devanagari-script language.

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

{target_guard}

IMPORTANT: If the target language ({target_lang}) is English or any non-Devanagari language, SKIP all OCR restoration steps below and ONLY produce a direct, fluent translation into {target_lang}. The restoration steps below apply ONLY when the output is in a Devanagari-script language.

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

    if not gemini_client:
        print("LLM Error: Google GenAI Client is not initialized (missing or invalid API key)")
        return text, MODEL

    models_to_try = [
        "gemini-3.5-flash",
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-pro",
    ]
    last_error = None
    for model_name in models_to_try:
        model_started = time.time()
        try:
            response = gemini_client.models.generate_content(
                model=model_name,
                contents=f"SNIPPET TO TRANSLATE:\n{text}",
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt.strip(),
                    temperature=0.0,
                )
            )
            text_result = response.text
            if not text_result:
                raise ValueError("Empty response text (possible safety block)")
            model_duration = time.time() - model_started
            total_duration = time.time() - request_started
            print(
                "LLM request complete: "
                f"model={model_name}, model_seconds={model_duration:.2f}, "
                f"total_seconds={total_duration:.2f}, output_chars={len(text_result)}"
            )
            return text_result.strip(), model_name
        except Exception as e:
            last_error = e
            model_duration = time.time() - model_started
            print(
                f"LLM Error with {model_name} after {model_duration:.2f}s: "
                f"{e}. Trying next model..."
            )
            continue

    print(f"All Gemini models failed. Last error: {last_error}")
    return text, MODEL


def translate_parallel_chunks(chunks: list[str], source_lang: str, target_lang: str, return_list: bool = False, full_context: str | None = None) -> tuple[str | list[str], str]:
    """Translates multiple chunks/pages in parallel to hit the < 5s target."""
    from concurrent.futures import ThreadPoolExecutor

    # Higher thread count for small chunks since I/O bound
    max_workers = min(len(chunks), 8)
    start_time = time.time()
    
    def translate_single(chunk):
        return _call_llm(chunk, source_lang, target_lang, full_context=full_context)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(translate_single, chunks))

    translated_texts = [res[0] for res in results]
    model_used = _combine_model_names(*(res[1] for res in results))
    
    if return_list:
        return translated_texts, model_used
        
    combined_text = "\n\n".join(translated_texts)
    print(
        "Parallel translation complete: "
        f"{len(chunks)} chunks, {max_workers} workers, {time.time() - start_time:.2f}s"
    )
    return combined_text, model_used


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

    if not gemini_client:
        print("Language detection failed: Google GenAI Client is not initialized (missing or invalid API key)")
        return {"language": "Detection Error", "code": "error", "confidence": 0.0}

    # Ordered by recency & availability. Excludes TTS/Live/Image-only models and quota-exhausted gemini-2.0-flash.
    models_to_try = [
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash",
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-pro",
    ]
    last_error = None
    for model_name in models_to_try:
        try:
            response = gemini_client.models.generate_content(
                model=model_name,
                contents=snippet,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt.strip(),
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=LanguageDetectionResult,
                )
            )
            if response.parsed:
                if isinstance(response.parsed, BaseModel):
                    return response.parsed.model_dump()
                elif isinstance(response.parsed, dict):
                    return response.parsed
            else:
                import json
                text_result = response.text
                if not text_result:
                    raise ValueError("Empty response text (possible safety block)")
                return json.loads(text_result.strip())
        except Exception as e:
            last_error = e
            print(f"Language detection failed with {model_name}: {e}. Trying next model...")
            continue

    print(f"All Gemini models failed for language detection. Last error: {last_error}")
    return {"language": "Detection Error", "code": "error", "confidence": 0.0}
