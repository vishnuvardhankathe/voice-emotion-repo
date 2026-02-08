import streamlit as st
import speech_recognition as sr
from transformers import pipeline
import pandas as pd
import plotly.express as px
import tempfile
import os

st.set_page_config(page_title="Voice Emotion Analyzer", layout="wide")
st.title("🎤 Voice Emotion Analysis Dashboard")

# Load emotion model
@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base"
    )

emotion_model = load_emotion_model()
recognizer = sr.Recognizer()

uploaded_file = st.file_uploader("Upload Audio File (.wav only)", type=["wav"])

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)

        sentences = text.split(".")
        rows = []

        time = 0
        for sentence in sentences:
            if sentence.strip():
                emotion = emotion_model(sentence)[0]["label"]
                rows.append({
                    "Start Time (sec)": time,
                    "Text": sentence.strip(),
                    "Emotion": emotion
                })
                time += 5   # approximate interval

        df = pd.DataFrame(rows)

        st.success("Analysis Complete!")

        st.subheader("Emotion Timeline")
        st.dataframe(df)

        fig1 = px.line(df, x="Start Time (sec)", y="Emotion", markers=True)
        st.plotly_chart(fig1)

        fig2 = px.bar(df["Emotion"].value_counts().reset_index(),
                      x="index", y="Emotion")
        st.plotly_chart(fig2)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "emotion_report.csv")

    except Exception as e:
        st.error("Please upload a valid WAV audio file.")
        st.write(e)

    finally:
        os.remove(audio_path)
