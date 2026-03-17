# Emotion Analysis ML System

This is a complete, local Machine Learning system that predicts user emotion, emotional intensity, and recommends personalized actions based on journal entries.

## Features
- **Local Embedded Models**: Uses `sentence-transformers` for dense semantic search without any external APIs.
- **Multi-Task Learning Approach**: Predicts both Emotion (Classification) and Intensity (Regression).
- **Rule-Based Recommender**: Maps predicted emotion/intensity combinations to psychological action protocols.
- **Robust Preprocessing**: Handles slang, repeated characters ("soooo"), and casing extraction.
- **Interactive Web App**: A Streamlit frontend for real-time journal entry analysis with visualization and history.

## Project Structure
- `src/preprocessing.py`: Text normalization and raw intensity feature extraction.
- `src/features.py`: TF-IDF, Semantic Dense Embeddings, and Subjectivity/Polarity.
- `src/models.py`: Extreme Gradient Boosting (XGBoost) classifiers and regressors.
- `src/recommender.py`: Heuristics mapping model outputs to real-world actions and time predictions.
- `src/pipeline.py`: The orchestrator that loads data, trains the models, and executes inference on the test set.
- `app.py`: The interactive Streamlit web application for real-time inference.

## How to Run
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   python -m textblob.download_corpora lite
   ```

2. Run the pipeline:
   ```bash
   python -m src.pipeline
   ```
   This will output training metrics (Macro F1, Accuracy, MAE) and generate an inference file `final_predictions.csv` containing the test dataset predictions and recommended action/delay.

3. Run the Streamlit Application:
   ```bash
   streamlit run app.py
   ```
   This will launch the interactive web interface in your browser, where you can type journal entries and get real-time emotion analysis and recommendations.