import streamlit as st
import speech_recognition as sr
from transformers import pipeline
import pandas as pd
import plotly.express as px
import tempfile
import os

st.set_page_config(
    page_title="Voice Emotion Analyzer",
    layout="wide"
)

st.title("🎤 Voice Emotion Analysis Dashboard")

@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base"
    )

emotion_model = load_emotion_model()
recognizer = sr.Recognizer()

uploaded_file = st.file_uploader(
    "Upload Audio File (.wav only)",
    type=["wav"]
)

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    try:
        with st.spinner("Transcribing audio..."):
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
                time += 5

        df = pd.DataFrame(rows)

        st.success("Analysis Complete!")

        st.subheader("Emotion Timeline")
        st.dataframe(df)

        st.subheader("Emotion Changes Over Time")
        fig1 = px.line(
            df,
            x="Start Time (sec)",
            y="Emotion",
            markers=True
        )
        st.plotly_chart(fig1, use_container_width=True)

        emotion_counts = df["Emotion"].value_counts().reset_index()
        emotion_counts.columns = ["Emotion", "Count"]

        st.subheader("Emotion Distribution")
        fig2 = px.bar(
            emotion_counts,
            x="Emotion",
            y="Count"
        )
        st.plotly_chart(fig2, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV Report",
            csv,
            "emotion_report.csv"
        )

    except Exception as e:
        st.error("Please upload a valid WAV audio file.")
        st.write(e)

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
