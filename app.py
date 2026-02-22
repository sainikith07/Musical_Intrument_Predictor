# ==========================================================
# InstruNet AI – Professional Dark Deployment Edition
# CNN-Based Musical Instrument Recognition System
# ==========================================================

import streamlit as st
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import matplotlib.pyplot as plt
import json
import os
import requests
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import TableStyle
from datetime import datetime

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="InstruNet AI | Professional Edition",
    page_icon="🎵",
    layout="wide"
)

# =========================
# GOOGLE DRIVE MODEL DOWNLOAD (SAFE FOR LARGE FILES)
# =========================
MODEL_PATH = "instrument_multilabel.keras"
FILE_ID = "1Zwx1bX33jRWPVGre9UxtEIMftkNKBgqZ"

def download_model_from_drive(file_id, destination):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={"id": file_id}, stream=True)

    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

if not os.path.exists(MODEL_PATH):
    st.info("📥 Downloading model from Google Drive...")
    download_model_from_drive(FILE_ID, MODEL_PATH)
    st.success("✅ Model downloaded successfully!")

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model(MODEL_PATH)

# IMPORTANT: Must match training order
CLASSES = ["flute", "guitar", "piano", "violin"]

ICON_MAP = {
    "flute": "🎶",
    "guitar": "🎸",
    "piano": "🎹",
    "violin": "🎻"
}

COLOR_MAP = {
    "flute": "#06B6D4",
    "guitar": "#3B82F6",
    "piano": "#FACC15",
    "violin": "#A855F7"
}

MAX_PAD_LEN = 128
CHUNK_DURATION = 1

# =========================
# DARK THEME UI
# =========================
st.markdown("""
<style>
body { background-color: #0F172A; color: white; }
h1, h2, h3 { color: #06B6D4; }
.stButton>button {
    background: linear-gradient(90deg,#7C3AED,#06B6D4);
    color: white;
    border-radius: 8px;
    border: none;
}
</style>
""", unsafe_allow_html=True)

st.title("🎵 InstruNet AI – Premium Deployment Edition")
st.markdown("CNN-Based Musical Instrument Recognition System")
st.markdown("---")

# =========================
# PREPROCESS FUNCTION
# =========================
def preprocess_chunk(y_chunk, sr):
    mel = librosa.feature.melspectrogram(y=y_chunk, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

    if mel_db.shape[1] < MAX_PAD_LEN:
        pad_width = MAX_PAD_LEN - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0,0),(0,pad_width)), mode='constant')
    else:
        mel_db = mel_db[:, :MAX_PAD_LEN]

    mel_input = mel_db[..., np.newaxis]
    mel_input = np.expand_dims(mel_input, axis=0)
    return mel_input

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file:

    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    y, sr = librosa.load("temp.wav", duration=5)
    st.audio(uploaded_file)

    # =========================
    # TIMELINE PREDICTION
    # =========================
    st.subheader("📊 Instrument Probability Timeline")

    chunk_samples = int(CHUNK_DURATION * sr)
    probabilities = []
    time_axis = []

    for i in range(0, len(y), chunk_samples):
        y_chunk = y[i:i+chunk_samples]
        if len(y_chunk) < chunk_samples:
            break

        mel_input = preprocess_chunk(y_chunk, sr)
        prediction = model.predict(mel_input, verbose=0)[0]
        probabilities.append(prediction)
        time_axis.append(i / sr)

    probabilities = np.array(probabilities)
    avg_prediction = np.mean(probabilities, axis=0)
    top_index = np.argmax(avg_prediction)

    # =========================
    # TIMELINE CHART
    # =========================
    fig_timeline, ax_timeline = plt.subplots()
    for i, instrument in enumerate(CLASSES):
        ax_timeline.plot(time_axis, probabilities[:, i], 
                         label=instrument, 
                         color=COLOR_MAP[instrument])
    ax_timeline.set_ylim([0, 1])
    ax_timeline.set_xlabel("Time (seconds)")
    ax_timeline.set_ylabel("Probability")
    ax_timeline.legend()
    st.pyplot(fig_timeline)

    # =========================
    # DOMINANT INSTRUMENT
    # =========================
    st.success(
        f"Dominant Instrument: {ICON_MAP[CLASSES[top_index]]} "
        f"{CLASSES[top_index].capitalize()}"
    )

    # =========================
    # PROBABILITY METRICS
    # =========================
    st.subheader("Instrument Probabilities")

    for i, instrument in enumerate(CLASSES):
        prob = float(avg_prediction[i])
        st.metric(
            label=f"{ICON_MAP[instrument]} {instrument.capitalize()}",
            value=f"{prob*100:.1f}%"
        )

    # =========================
    # WAVEFORM
    # =========================
    st.subheader("🎵 Audio Waveform")
    fig_wave, ax_wave = plt.subplots()
    ax_wave.plot(y)
    ax_wave.set_title("Amplitude vs Time")
    st.pyplot(fig_wave)

    # =========================
    # SPECTROGRAM
    # =========================
    st.subheader("🎼 Mel Spectrogram")
    mel_full = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_full_db = librosa.power_to_db(mel_full, ref=np.max)

    fig_spec, ax_spec = plt.subplots(figsize=(10,4))
    img = librosa.display.specshow(
        mel_full_db,
        sr=sr,
        x_axis="time",
        y_axis="mel",
        ax=ax_spec
    )
    fig_spec.colorbar(img, ax=ax_spec)
    st.pyplot(fig_spec)

    # =========================
    # PIE CHART
    # =========================
    st.subheader("🥧 Instrument Distribution")

    normalized = avg_prediction / (np.sum(avg_prediction) + 1e-6)
    explode = [0.1 if i == top_index else 0 for i in range(len(CLASSES))]

    fig_pie, ax_pie = plt.subplots()
    wedges, _ = ax_pie.pie(
        normalized,
        explode=explode,
        colors=[COLOR_MAP[i] for i in CLASSES],
        startangle=90
    )
    ax_pie.legend(
        wedges,
        [f"{CLASSES[i]} ({normalized[i]*100:.1f}%)" for i in range(len(CLASSES))],
        loc="center left",
        bbox_to_anchor=(1, 0.5)
    )
    st.pyplot(fig_pie)

    # =========================
    # JSON DOWNLOAD
    # =========================
    report_data = {
        "average_probabilities": {
            instrument: float(avg_prediction[i])
            for i, instrument in enumerate(CLASSES)
        },
        "dominant_instrument": CLASSES[top_index],
        "generated_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    st.download_button(
        label="📥 Download JSON Report",
        data=json.dumps(report_data, indent=4),
        file_name="instrument_analysis.json",
        mime="application/json"
    )
