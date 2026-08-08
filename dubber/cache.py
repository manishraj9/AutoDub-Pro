import os
import json
import gzip
import csv
import hashlib
import threading
from typing import Dict, List, Optional, Tuple
from dubber.config import CACHE_DIR

# Ensure cache directory exists on import
os.makedirs(CACHE_DIR, exist_ok=True)

# Thread-safe locks for cache files
cache_lock = threading.Lock()

def generate_text_hash(text: str) -> str:
    """Generates a unique SHA-256 hash key for a given text segment."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

def get_cache_filepath(source_lang: str = "auto", target_lang: str = "English", compressed: bool = False) -> str:
    """
    Returns the absolute file path for the translation cache file.
    Supports standard JSON or compressed gzip JSON formats indexed by target language.
    """
    src_clean = source_lang.strip().lower().replace("/", "_").replace("\\", "_").replace(" ", "_")
    tgt_clean = target_lang.strip().lower().replace("/", "_").replace("\\", "_").replace(" ", "_")
    ext = "json.gz" if compressed else "json"
    return os.path.join(CACHE_DIR, f"translations_{src_clean}_to_{tgt_clean}.{ext}")

def load_translation_cache(source_lang: str = "auto", target_lang: str = "English") -> Dict[str, str]:
    """
    Loads translation pairs (hash -> translation) from local storage for the specified target language.
    Automatically detects and decompresses gzip format if present.
    """
    with cache_lock:
        compressed_path = get_cache_filepath(source_lang, target_lang, compressed=True)
        if os.path.exists(compressed_path):
            try:
                with gzip.open(compressed_path, "rt", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[Warning] Failed to read compressed cache file {compressed_path}: {e}")
                
        plain_path = get_cache_filepath(source_lang, target_lang, compressed=False)
        if os.path.exists(plain_path):
            try:
                with open(plain_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                save_translation_cache_unlocked(data, source_lang, target_lang, compress=True)
                try:
                    os.remove(plain_path)
                except Exception:
                    pass
                return data
            except Exception as e:
                print(f"[Warning] Failed to read plain cache file {plain_path}: {e}")
                
        return {}

def save_translation_cache_unlocked(cache_data: Dict[str, str], source_lang: str = "auto", target_lang: str = "English", compress: bool = True) -> None:
    """
    Internal helper to write cache entries to disk. Must be called inside lock.
    """
    if compress:
        cache_path = get_cache_filepath(source_lang, target_lang, compressed=True)
        try:
            with gzip.open(cache_path, "wt", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Warning] Failed to save compressed cache to file {cache_path}: {e}")
    else:
        cache_path = get_cache_filepath(source_lang, target_lang, compressed=False)
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Warning] Failed to save plain cache to file {cache_path}: {e}")

def save_translation_cache(cache_data: Dict[str, str], source_lang: str = "auto", target_lang: str = "English", compress: bool = True) -> None:
    """
    Saves translation pairs to a local serialized cache file in a thread-safe manner.
    """
    with cache_lock:
        save_translation_cache_unlocked(cache_data, source_lang, target_lang, compress)

def get_cached_translation(text: str, source_lang: str = "auto", target_lang: str = "English") -> Optional[str]:
    """
    Checks if a segment has a cached translation in target_lang and returns it, or None if not found.
    """
    cache_data = load_translation_cache(source_lang, target_lang)
    text_hash = generate_text_hash(text)
    return cache_data.get(text_hash)

def add_to_translation_cache(text: str, translation: str, source_lang: str = "auto", target_lang: str = "English") -> None:
    """
    Adds a new translation pair to the target language persistent cache in a thread-safe manner.
    """
    cache_data = load_translation_cache(source_lang, target_lang)
    text_hash = generate_text_hash(text)
    
    if cache_data.get(text_hash) != translation:
        cache_data[text_hash] = translation.strip()
        save_translation_cache(cache_data, source_lang, target_lang)


def get_cached_segments_count() -> int:
    """
    Counts the total number of translation segments saved across all language caches.
    """
    total = 0
    if not os.path.exists(CACHE_DIR):
        return 0
        
    with cache_lock:
        for filename in os.listdir(CACHE_DIR):
            file_path = os.path.join(CACHE_DIR, filename)
            # Handle compressed json.gz
            if filename.startswith("translations_") and filename.endswith(".json.gz"):
                try:
                    with gzip.open(file_path, "rt", encoding="utf-8") as f:
                        data = json.load(f)
                        total += len(data)
                except Exception:
                    pass
            # Handle plain json
            elif filename.startswith("translations_") and filename.endswith(".json"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        total += len(data)
                except Exception:
                    pass
    return total

def clear_translation_cache() -> None:
    """
    Clears all cached translation records by deleting all JSON/GZIP files in the cache folder.
    """
    print(f"[Info] Clearing translation cache directory: {CACHE_DIR}...")
    if not os.path.exists(CACHE_DIR):
        return
        
    with cache_lock:
        for filename in os.listdir(CACHE_DIR):
            if filename.startswith("translations_") and (filename.endswith(".json") or filename.endswith(".json.gz")):
                file_path = os.path.join(CACHE_DIR, filename)
                try:
                    os.remove(file_path)
                    print(f"  - Deleted cache file: {filename}")
                except Exception as e:
                    print(f"  - [Warning] Failed to delete cache file {filename}: {e}")
    print("[Info] Caches cleared successfully.")

def export_cache_to_csv(csv_path: str, source_lang: str = "auto") -> bool:
    """
    Exports a specific language cache into a CSV format sheet (hash, translated text).
    Useful for review or backup purposes.
    """
    cache_data = load_translation_cache(source_lang)
    if not cache_data:
        print(f"[Info] No cache entries to export for language: {source_lang}.")
        return False
        
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Hash", "Translation"])
            for text_hash, trans_text in cache_data.items():
                writer.writerow([text_hash, trans_text])
        print(f"[Info] Caches exported successfully to CSV: {csv_path}")
        return True
    except Exception as e:
        print(f"[Error] Failed to export cache to CSV: {e}")
        return False

def import_cache_from_csv(csv_path: str, source_lang: str = "auto") -> int:
    """
    Imports translation cache pairs from a formatted CSV sheet.
    Returns:
        int: Number of records successfully imported.
    """
    if not os.path.exists(csv_path):
        return 0
        
    cache_data = load_translation_cache(source_lang)
    imported = 0
    try:
        with open(csv_path, "r", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file)
            # Skip header row
            header = next(reader, None)
            if header and len(header) >= 2:
                for row in reader:
                    if len(row) >= 2:
                        text_hash, trans_text = row[0].strip(), row[1].strip()
                        if text_hash and trans_text:
                            cache_data[text_hash] = trans_text
                            imported += 1
        if imported > 0:
            save_translation_cache(cache_data, source_lang)
            print(f"[Info] Successfully imported {imported} translation pairs from CSV.")
        return imported
    except Exception as e:
        print(f"[Error] Failed to import cache from CSV: {e}")
        return 0
