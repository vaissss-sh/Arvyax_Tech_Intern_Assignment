import streamlit as st
import pandas as pd
import time
from src.pipeline import MLPipeline

# Must be the first Streamlit command
st.set_page_config(
    page_title="Emotion AI Assistant",
    page_icon="🧠",
    layout="centered"
)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "text_input" not in st.session_state:
    st.session_state.text_input = ""

@st.cache_resource(show_spinner="Loading ML Models... this may take a moment.")
def load_pipeline():
    """ Load the pipeline and pre-trained weights. """
    p = MLPipeline()
    # Path to the training file to fit the transformers correctly
    # Since we are caching, this runs once on startup
    p.train("Sample_arvyax_reflective_dataset.xlsx - Dataset_120.csv")
    return p

pipeline = load_pipeline()

# ---------------------------------------------------------
# UI Layout
# ---------------------------------------------------------
st.title("🧠 Emotion AI Assistant")
st.markdown("Analyze your journal entries to understand your emotional state and receive actionable recommendations.")

# Example Inputs
st.markdown("### Quick Examples")
col1, col2, col3 = st.columns(3)

def set_text(text):
    st.session_state.text_input = text

if col1.button("Overwhelmed & stressed"):
    set_text("Honestly I felt mentally flooded. Everything broke down and I am sooooo exhausted.")
if col2.button("Focused & calm"):
    set_text("Woke up feeling able to prioritize. Mountain visuals made it easier to pause and reflect.")
if col3.button("Restless & mixed"):
    set_text("Started off distracted most of the time. Kinda jumpy tbh.")

# User Input
user_text = st.text_area(
    "How are you feeling right now?", 
    st.session_state.text_input,
    height=120,
    placeholder="Type your journal entry here..."
)

# Analyze Button
if st.button("Analyze Emotion", type="primary"):
    if not user_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing text using NLP and Deep Embeddings..."):
            # Sleep briefly just to make the spinner noticeable if inference is too fast
            time.sleep(0.5) 
            try:
                result = pipeline.predict_single(user_text)
                
                # Add to history (limit to last 5)
                st.session_state.history.insert(0, {"text": user_text, "result": result})
                st.session_state.history = st.session_state.history[:5]
                
                st.markdown("---")
                st.subheader("Analysis Results")
                
                res_col1, res_col2 = st.columns(2)
                
                emotion = result["emotion"].lower()
                intensity = result["intensity"]
                action = result["recommended_action"]
                delay = result["time_to_act_min"]
                
                # Styling maps based on emotion class
                if emotion in ["restless", "mixed"]:
                    res_col1.warning(f"**Emotion:** {emotion.capitalize()}")
                elif emotion == "overwhelmed":
                    res_col1.error(f"**Emotion:** {emotion.capitalize()}")
                elif emotion in ["calm", "focused", "neutral"]:
                    res_col1.success(f"**Emotion:** {emotion.capitalize()}")
                else:
                    res_col1.info(f"**Emotion:** {emotion.capitalize()}")
                    
                # Intensity output
                res_col2.info(f"**Intensity Level:** {intensity} / 5")
                st.progress(intensity / 5.0)
                
                # Action output
                st.markdown("### Recommended Action")
                time_str = "Immediate Action Required" if delay == 0 else f"Consider acting in ~{delay} mins"
                
                st.info(f"**{time_str}**\n\n👉 {action}")
                
            except Exception as e:
                st.error(f"Error during prediction: {e}")

# ---------------------------------------------------------
# History Section
# ---------------------------------------------------------
if st.session_state.history:
    st.markdown("---")
    with st.expander("View Last 5 Inputs (History)"):
        for i, item in enumerate(st.session_state.history):
            st.markdown(f"**Input {i+1}:** *\"{item['text']}\"*")
            st.markdown(f"- **Emotion:** {item['result']['emotion']} (Intensity: {item['result']['intensity']})")
            st.markdown("---")
