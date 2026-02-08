import streamlit as st
import whisper
from transformers import pipeline
import pandas as pd
import plotly.express as px
import tempfile
import os

st.set_page_config(page_title="Voice Emotion Analyzer", layout="wide")

st.title("🎤 Voice Emotion Analysis Dashboard")

@st.cache_resource
def load_models():
    whisper_model = whisper.load_model("base")
    emotion_model = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base"
    )
    return whisper_model, emotion_model

whisper_model, emotion_model = load_models()

uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    result = whisper_model.transcribe(audio_path)

    rows = []

    for seg in result["segments"]:
        text = seg["text"]
        emotion = emotion_model(text)[0]["label"]
        rows.append({
            "Start Time (sec)": round(seg["start"],2),
            "End Time (sec)": round(seg["end"],2),
            "Text": text,
            "Emotion": emotion
        })

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
    st.download_button("Download CSV", csv, "report.csv")

    os.remove(audio_path)
