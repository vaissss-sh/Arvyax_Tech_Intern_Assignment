import pandas as pd
import re

train_path = "Sample_arvyax_reflective_dataset.xlsx - Dataset_120.csv"
try:
    df = pd.read_csv(train_path)
    text_data = df['journal_text'].dropna().astype(str).tolist()
    
    # Check for emojis using a regex range for emojis
    emoji_pattern = re.compile("[\U00010000-\U0010ffff]", flags=re.UNICODE)
    texts_with_emojis = [t for t in text_data if emoji_pattern.search(t)]
    print(f"Texts with emojis: {len(texts_with_emojis)} / {len(text_data)}")
    if texts_with_emojis:
        print("Emamples:", texts_with_emojis[:3])
    
    # Check for repeated characters (e.g., soooo)
    repeated_chars = [t for t in text_data if re.search(r'(.)\1{2,}', t)]
    print(f"\nTexts with repeated characters: {len(repeated_chars)} / {len(text_data)}")
    if repeated_chars:
        print("Examples:", repeated_chars[:3])
        
    # Check for slang / informal (look for common ones like 'kinda', 'gonna', 'wanna', 'af', 'rn')
    slang_pattern = re.compile(r'\b(kinda|gonna|wanna|af|rn|tbh|imo|omg|lol|lmao)\b', re.IGNORECASE)
    slang_texts = [t for t in text_data if slang_pattern.search(t)]
    print(f"\nTexts with common slang: {len(slang_texts)} / {len(text_data)}")
    if slang_texts:
        print("Examples:", slang_texts[:3])
        
except Exception as e:
    print(f"Error: {e}")
