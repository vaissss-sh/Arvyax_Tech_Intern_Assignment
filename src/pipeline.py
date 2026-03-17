import pandas as pd
import os
import joblib
from src.preprocessing import preprocess_dataframe
from src.features import FeatureExtractor
from src.models import EmotionModel, IntensityModel
from src.recommender import RecommenderSystem
from sklearn.metrics import f1_score, mean_absolute_error, accuracy_score

class MLPipeline:
    def __init__(self, use_embeddings=True, model_type="xgboost"):
        self.use_embeddings = use_embeddings
        self.model_type = model_type
        
        self.feature_extractor = FeatureExtractor(use_embeddings=self.use_embeddings)
        self.emotion_model = EmotionModel(model_type=self.model_type)
        self.intensity_model = IntensityModel(model_type=self.model_type)
        self.recommender = RecommenderSystem()
        
    def train(self, train_path: str):
        print(f"Loading training data from {train_path}...")
        df = pd.read_csv(train_path)
        
        # 1. Preprocess
        print("Preprocessing text...")
        df = preprocess_dataframe(df, text_col="journal_text")
        
        # 2. Extract Features
        print("Extracting features (this may take a minute if embeddings are used)...")
        df = self.feature_extractor.fit_transform(df, text_col="normalized_text")
        
        # Define feature columns for modeling
        # Exclude metadata like id, arbitrary strings, and targets
        feature_cols = [c for c in df.columns if c.startswith("feat_") or c.startswith("tfidf_") or c.startswith("emb_")]
        
        # Include numerical indicators
        for col in ["duration_min", "sleep_hours", "energy_level", "stress_level"]:
            if col in df.columns:
                feature_cols.append(col)
                
        X = df[feature_cols].fillna(0)
        y_emotion = df["emotional_state"]
        y_intensity = df["intensity"]
        
        # 3. Train Models
        print(f"Training Emotion Model ({self.model_type})...")
        self.emotion_model.fit(X, y_emotion)
        
        print(f"Training Intensity Model ({self.model_type})...")
        self.intensity_model.fit(X, y_intensity)
        
        # Evaluate on Train Set
        preds_emo = self.emotion_model.predict(X)
        preds_int = self.intensity_model.predict(X)
        
        print("\n--- Training Set Evaluation ---")
        print(f"Emotion F1 (Macro): {f1_score(y_emotion, preds_emo, average='macro'):.4f}")
        print(f"Emotion Accuracy  : {accuracy_score(y_emotion, preds_emo):.4f}")
        print(f"Intensity MAE     : {mean_absolute_error(y_intensity, preds_int):.4f}")
        print("------------------------------\n")
        
        self.feature_cols = feature_cols

    def predict(self, test_path: str, output_path: str):
        print(f"Loading inference data from {test_path}...")
        df = pd.read_csv(test_path)
        
        print("Preprocessing text...")
        df = preprocess_dataframe(df, text_col="journal_text")
        
        print("Extracting features...")
        df = self.feature_extractor.transform(df, text_col="normalized_text")
        
        # Ensure all feature columns are present
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0
                
        X = df[self.feature_cols].fillna(0)
        
        print("Making predictions...")
        df["predicted_emotion"] = self.emotion_model.predict(X)
        df["predicted_intensity"] = self.intensity_model.predict(X)
        
        print("Generating actions and time delays...")
        final_df = self.recommender.get_recommendations(df, "predicted_emotion", "predicted_intensity")
        
        print(f"Saving final output to {output_path}...")
        # Keep relevant columns for readability
        out_cols = [
            "id", "journal_text", "predicted_emotion", "predicted_intensity", 
            "recommended_action", "time_to_act_min", "stress_level", "previous_day_mood"
        ]
        # fallback if some don't exist
        out_cols = [c for c in out_cols if c in final_df.columns]
        
        final_df[out_cols].to_csv(output_path, index=False)
        print("Done!")

    def predict_single(self, text: str) -> dict:
        """
        Runs inference on a single string of text.
        """
        df = pd.DataFrame([{"journal_text": text}])
        
        df = preprocess_dataframe(df, text_col="journal_text")
        df = self.feature_extractor.transform(df, text_col="normalized_text")
        
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0
                
        X = df[self.feature_cols].fillna(0)
        
        emotion = self.emotion_model.predict(X)[0]
        intensity = int(self.intensity_model.predict(X)[0])
        
        # Defaulting meta-vars since we are doing real-time
        df["predicted_emotion"] = [emotion]
        df["predicted_intensity"] = [intensity]
        df["stress_level"] = [3]
        df["duration_min"] = [10]
        df["previous_day_mood"] = ["neutral"]
        
        final_df = self.recommender.get_recommendations(df, "predicted_emotion", "predicted_intensity")
        
        return {
            "emotion": emotion,
            "intensity": intensity,
            "recommended_action": final_df["recommended_action"].iloc[0],
            "time_to_act_min": final_df["time_to_act_min"].iloc[0]
        }

if __name__ == "__main__":
    import argparse
    
    # Simple execution when run as a script
    train_file = "Sample_arvyax_reflective_dataset.xlsx - Dataset_120.csv"
    test_file = "arvyax_test_inputs_120.xlsx - Sheet1.csv"
    output_file = "final_predictions.csv"
    
    pipeline = MLPipeline(use_embeddings=True, model_type="xgboost")
    pipeline.train(train_file)
    pipeline.predict(test_file, output_file)
