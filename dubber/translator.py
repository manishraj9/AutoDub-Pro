import json
from typing import List
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dubber.config import client_manager, DEFAULT_TRANSLATION_MODEL

class TranslatedSegment(BaseModel):
    start_time: float = Field(description="Start time of the segment (must match input segment exactly)")
    end_time: float = Field(description="End time of the segment (must match input segment exactly)")
    text: str = Field(description="The translated text in the target language")

class TranslationResponse(BaseModel):
    segments: List[TranslatedSegment]

# Map of supported target languages to their locale prefixes for Edge-TTS voice filtering
SUPPORTED_LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Japanese": "ja",
    "Korean": "ko",
    "Portuguese": "pt",
    "Arabic": "ar",
    "Russian": "ru",
    "Chinese": "zh",
    "Italian": "it",
    "Turkish": "tr",
    "Indonesian": "id",
    "Bengali": "bn"
}

def translate_batch(client: genai.Client, segments: List[dict], model: str, target_language: str = "English") -> List[dict]:
    """
    Translates a single batch of segments using Gemini API.
    """
    prompt = (
        f"You are an elite, award-winning film localization director and translator.\n"
        f"Translate the spoken dialogue segments from their original language into high-level, natural, "
        f"and authentic spoken {target_language} designed for professional video dubbing and voice acting.\n\n"
        f"CRITICAL TRANSLATION GUIDELINES:\n"
        f"1. Produce 100% natural, fluent, professional dialogue. Avoid robotic or literal word-for-word translation.\n"
        f"2. Adapt idioms, emotional tone, and phrasing so it sounds like an authentic native speaker.\n"
        f"3. Ensure the sentence length and word count match the rhythm and speaking duration of dialogue.\n"
        f"4. Maintain exact context and character intent across all turns.\n"
        f"5. Retain the exact same start_time and end_time for each segment as provided in the input.\n\n"
        f"Input segments to translate:\n{json.dumps(segments, indent=2)}"
    )
    
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=TranslationResponse,
            temperature=0.2,
            max_output_tokens=35000,
        ),
    )
    
    raw_text = response.text
    data = json.loads(raw_text)
    
    translated_raw = data.get("segments", [])
    
    # Ensure they map perfectly to the inputs. If the model omitted or messed up timestamps, we'll
    # map them back strictly using index mapping as a fallback to maintain reliability.
    result = []
    for idx, input_seg in enumerate(segments):
        translated_text = input_seg["text"] # Fallback to original text if missing
        if idx < len(translated_raw):
            translated_text = translated_raw[idx]["text"].strip()
            
        result.append({
            "start_time": input_seg["start_time"],
            "end_time": input_seg["end_time"],
            "text": translated_text
        })
        
    return result

def translate_segments(segments: List[dict], model: str = DEFAULT_TRANSLATION_MODEL, target_language: str = "English") -> List[dict]:
    """
    Translates all segments into the target language, processing in batches.
    """
    if not segments:
        return []
        
    batch_size = 40  # Keep batch size conservative to ensure model has room to generate good translations
    translated_segments = []
    
    total_batches = (len(segments) + batch_size - 1) // batch_size
    print(f"[Info] Translating {len(segments)} segments to {target_language} in {total_batches} batch(es)...")
    
    for i in range(0, len(segments), batch_size):
        batch = segments[i:i + batch_size]
        batch_idx = (i // batch_size) + 1
        print(f"[Info] Translating batch {batch_idx}/{total_batches}...")
        
        # Use execute_with_retry to rotate Gemini keys
        def op(client):
            return translate_batch(client, batch, model, target_language)
            
        translated_batch = client_manager.execute_with_retry(op)
        translated_segments.extend(translated_batch)
        
    print(f"[Info] Translation completed.")
    return translated_segments

def condense_translation_if_fast(segment: dict, max_wpm: float = 175.0, model: str = DEFAULT_TRANSLATION_MODEL, target_language: str = "English") -> dict:
    """
    If the speaking rate of the translated text is too high for the segment duration,
    queries Gemini to shorten the text in target_language to fit comfortably while retaining meaning.
    """
    from dubber.utils import calculate_speaking_rate
    
    duration = segment["end_time"] - segment["start_time"]
    if duration <= 0.2:
        return segment
        
    wpm = calculate_speaking_rate(segment["text"], duration)
    if wpm <= max_wpm:
        return segment
        
    print(f"[Info] Segment rate of {wpm} WPM exceeds threshold of {max_wpm}. Condensing text in {target_language}...")
    
    prompt = (
        f"You are an expert copyeditor translating and dubbing audio into {target_language}. "
        f"The following {target_language} text is going to be dubbed over video, "
        f"but it is too long to fit in its available time slot of {duration:.2f} seconds. "
        f"The current speaking rate is {wpm:.1f} WPM, which is too fast.\n\n"
        f"Your task is to shorten/condense the text in fluent, natural {target_language} so it can be spoken in the timeframe. "
        "Keep the exact same meaning, tone, and critical information identical, but make it significantly more concise. "
        f"Target a length of around {max(1, int(duration * (max_wpm / 60.0)))} words or less.\n\n"
        f"Current {target_language} Text: \"{segment['text']}\"\n\n"
        f"Return ONLY the condensed text in {target_language}. Do not add quotes, explanations, English translations, or introductory text."
    )
    
    def op(client):
        res = client.models.generate_content(
            model=model,
            contents=prompt
        )
        return res.text.strip().strip('"').strip("'")
        
    try:
        condensed_text = client_manager.execute_with_retry(op)
        if condensed_text and len(condensed_text) < len(segment["text"]):
            new_wpm = calculate_speaking_rate(condensed_text, duration)
            print(f"[Success] Condensed from {len(segment['text'])} to {len(condensed_text)} chars ({wpm} WPM -> {new_wpm} WPM).")
            segment = segment.copy()
            segment["text"] = condensed_text
    except Exception as e:
        print(f"[Warning] Failed to condense translation: {e}. Falling back to original.")
        
    return segment


