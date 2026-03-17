import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
import os

class FeatureExtractor:
    def __init__(self, use_embeddings=True):
        self.use_embeddings = use_embeddings
        self.tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        
        if self.use_embeddings:
            # We use a fast, lightweight local model suitable for small memory footprints
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
    def get_lexicon_features(self, text: str) -> dict:
        """
        Extracts polarity and subjectivity using TextBlob.
        """
        if not isinstance(text, str) or not text.strip():
            return {"polarity": 0.0, "subjectivity": 0.0}
            
        blob = TextBlob(text)
        return {
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity
        }
        
    def fit_transform(self, df: pd.DataFrame, text_col="normalized_text") -> pd.DataFrame:
        """
        Fits the TF-IDF vectorizer and transforms the data, appending to original dataframe.
        """
        # Ensure column exists
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found. Did you run preprocessing?")
            
        texts = df[text_col].fillna("").tolist()
        
        # 1. Lexicon Features
        lex_feats = [self.get_lexicon_features(t) for t in texts]
        df['feat_polarity'] = [f['polarity'] for f in lex_feats]
        df['feat_subjectivity'] = [f['subjectivity'] for f in lex_feats]
        
        # 2. TF-IDF
        tfidf_matrix = self.tfidf.fit_transform(texts)
        tfidf_feats = pd.DataFrame(
            tfidf_matrix.toarray(), 
            columns=[f"tfidf_{w}" for w in self.tfidf.get_feature_names_out()],
            index=df.index
        )
        
        result_df = pd.concat([df, tfidf_feats], axis=1)
        
        # 3. Dense Embeddings
        if self.use_embeddings:
            print("Encoding dense embeddings...")
            embeddings = self.model.encode(texts, show_progress_bar=False)
            emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
            emb_df = pd.DataFrame(embeddings, columns=emb_cols, index=df.index)
            result_df = pd.concat([result_df, emb_df], axis=1)
            
        return result_df

    def transform(self, df: pd.DataFrame, text_col="normalized_text") -> pd.DataFrame:
        """
        Transforms new data (e.g. inference) using pre-fitted transformers.
        """
        texts = df[text_col].fillna("").tolist()
        
        lex_feats = [self.get_lexicon_features(t) for t in texts]
        df['feat_polarity'] = [f['polarity'] for f in lex_feats]
        df['feat_subjectivity'] = [f['subjectivity'] for f in lex_feats]
        
        tfidf_matrix = self.tfidf.transform(texts)
        tfidf_feats = pd.DataFrame(
            tfidf_matrix.toarray(), 
            columns=[f"tfidf_{w}" for w in self.tfidf.get_feature_names_out()],
            index=df.index
        )
        
        result_df = pd.concat([df, tfidf_feats], axis=1)
        
        if self.use_embeddings:
            embeddings = self.model.encode(texts, show_progress_bar=False)
            emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
            emb_df = pd.DataFrame(embeddings, columns=emb_cols, index=df.index)
            result_df = pd.concat([result_df, emb_df], axis=1)
            
        return result_df
