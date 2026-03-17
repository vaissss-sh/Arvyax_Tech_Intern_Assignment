import re
import pandas as pd
import numpy as np

SLANG_DICT = {
    "tbh": "to be honest",
    "kinda": "kind of",
    "gonna": "going to",
    "wanna": "want to",
    "af": "as fuck",
    "rn": "right now",
    "imo": "in my opinion",
    "omg": "oh my god",
    "lol": "laughing out loud",
    "lmao": "laughing my ass off"
}

def extract_intensity_features(text: str) -> dict:
    """
    Extracts raw intensity features before the text is normalized.
    """
    if not isinstance(text, str):
        return {"caps_ratio": 0.0, "exclamations": 0, "repeats": 0}
        
    words = text.split()
    caps_count = sum(1 for w in words if w.isupper() and len(w) > 1)
    caps_ratio = caps_count / max(len(words), 1)
    
    exclamations = text.count('!')
    
    # Count words with characters repeated >= 3 times (e.g., "soooo")
    repeats = len(re.findall(r'(.)\1{2,}', text))
    
    return {
        "caps_ratio": caps_ratio,
        "exclamations": exclamations,
        "repeats": repeats
    }

def normalize_text(text: str) -> str:
    """
    Cleans and normalizes the text.
    """
    if not isinstance(text, str):
        return ""
        
    # 1. Lowercase
    text = text.lower()
    
    # 2. Translate slang
    for slang, full_phrase in SLANG_DICT.items():
        # Match whole words only
        text = re.sub(rf'\b{slang}\b', full_phrase, text)
        
    # 3. Reduce elongated characters (e.g., "sooo" -> "soo")
    # We reduce to 2 to preserve some emphasis but remove OOV tokens.
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # 4. Remove excessive punctuation, but keep standard sentence structure
    text = re.sub(r'[^\w\s\.,!\?]', '', text)
    
    # 5. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_dataframe(df: pd.DataFrame, text_col="journal_text") -> pd.DataFrame:
    """
    Applies intensity extraction and text normalization to a dataframe.
    """
    # Defensive copy
    processed_df = df.copy()
    
    # Extract features first
    intensity_feats = [extract_intensity_features(t) for t in processed_df[text_col]]
    
    processed_df['feat_caps_ratio'] = [f['caps_ratio'] for f in intensity_feats]
    processed_df['feat_exclamations'] = [f['exclamations'] for f in intensity_feats]
    processed_df['feat_repeats'] = [f['repeats'] for f in intensity_feats]
    
    # Normalize text
    processed_df['normalized_text'] = processed_df[text_col].apply(normalize_text)
    
    return processed_df
