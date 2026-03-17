import pandas as pd
import numpy as np

class RecommenderSystem:
    def __init__(self):
        # We can configure default mappings here
        pass

    def get_action(self, emotion: str, intensity: int) -> str:
        """
        Maps (emotion + intensity) to a recommended action.
        """
        emotion = str(emotion).lower()
        
        if emotion == "overwhelmed" and intensity >= 4:
            return "Pause & execute 4-7-8 breathing protocol."
            
        elif emotion == "restless" and intensity >= 4:
            return "Physical switch: Take a 10-minute walk or stand up."
            
        elif emotion == "neutral":
            return "Continue current workflow & monitor."
            
        elif emotion == "mixed" and intensity >= 3:
            return "Log a 3-bullet journal to untangle thoughts."
            
        elif emotion == "focused":
            return "Maintain momentum. Do not break flow."
            
        elif emotion == "calm":
            return "Acknowledge peace. Continue at your own pace."
            
        else:
            if intensity >= 4:
                return "High emotional intensity detected. Take a 5-minute break away from screens."
            else:
                return "Take note of your feelings and continue your day."
                
    def get_time_delay(self, intensity: int, stress_level: int, duration_min: int) -> int:
        """
        Predicts when to act (in minutes).
        Returns 0 for immediate action.
        """
        # Standout concept: Inconsistency logic or direct thresholding
        if intensity >= 4 or stress_level == 5:
            return 0  # Immediate action required
            
        # Otherwise, wait until the current session (duration_min) concludes, or a minimum 30 min buffer
        delay = max(duration_min, 30)
        return delay

    def get_recommendations(self, df: pd.DataFrame, emotion_pred_col: str, intensity_pred_col: str) -> pd.DataFrame:
        """
        Applies logic across a dataframe.
        Expects real or derived 'stress_level' and 'duration_min'.
        """
        result_df = df.copy()
        
        actions = []
        delays = []
        
        for idx, row in result_df.iterrows():
            emotion = row[emotion_pred_col]
            intensity = row[intensity_pred_col]
            
            # Use safe defaults if columns are missing
            stress_level = row.get("stress_level", 3)
            duration_min = row.get("duration_min", 30)
            
            action = self.get_action(emotion, intensity)
            delay = self.get_time_delay(intensity, stress_level, duration_min)
            
            # Incorporating Step 10: "Temporal Context Memory Buffer" concept
            # If previous day was also restless and today is high intensity
            prev_mood = str(row.get("previous_day_mood", "")).lower()
            if emotion == "restless" and prev_mood == "restless" and intensity >= 4:
                action = "MULTIPLE DAYS RESTLESS! " + action + " Also, consider discussing with a mentor or taking an extended break."
            
            actions.append(action)
            delays.append(delay)
            
        result_df["recommended_action"] = actions
        result_df["time_to_act_min"] = delays
        
        return result_df
