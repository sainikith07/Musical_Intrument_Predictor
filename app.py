import streamlit as st
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import matplotlib.pyplot as plt
import json
import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import TableStyle

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="InstruNet AI", page_icon="🎵", layout="wide")

MAX_PAD_LEN = 128
CHUNK_DURATION = 1
THRESHOLD = 0.5

ICON_MAP = {
    "flute": "🎶",
    "guitar": "🎸",
    "piano": "🎹",
    "violin": "🎻"
}

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model("instrument_multilabel.keras")

# IMPORTANT: keep class order same as training
CLASSES = sorted([
    folder for folder in os.listdir("dataset")
    if folder != "mixed"
])

# =========================
# DARK UI
# =========================
st.markdown("""
<style>
body { background-color: #0E1117; color: white; }
h1, h2, h3 { color: #00FFAA; }
.stButton>button {
    background-color: #00FFAA;
    color: black;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

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
# UI START
# =========================
st.title("🎵 InstruNet AI - Premium Deployment Edition")

uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file:

    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    y, sr = librosa.load("temp.wav", duration=5)
    st.audio(uploaded_file)

    # =========================
    # CHUNK-BASED TIMELINE
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

    fig_timeline, ax_timeline = plt.subplots()

    for i, instrument in enumerate(CLASSES):
        ax_timeline.plot(time_axis, probabilities[:, i], label=instrument)

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
        confidence = "High" if prob > 0.8 else "Medium" if prob > 0.5 else "Low"

        st.metric(
            label=f"{ICON_MAP[instrument]} {instrument.capitalize()}",
            value=f"{prob*100:.1f}%",
            delta=f"{confidence} Confidence"
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

    fig_spec, ax_spec = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        mel_full_db,
        sr=sr,
        x_axis="time",
        y_axis="mel",
        ax=ax_spec
    )

    fig_spec.colorbar(img, ax=ax_spec, format="%+2.f dB")
    st.pyplot(fig_spec)

    # =========================
    # PIE CHART CLEAN
    # =========================
    st.subheader("🥧 Instrument Distribution")

    normalized = avg_prediction / (np.sum(avg_prediction) + 1e-6)
    explode = [0.1 if i == top_index else 0 for i in range(len(CLASSES))]

    fig_pie, ax_pie = plt.subplots()

    wedges, _ = ax_pie.pie(
        normalized,
        explode=explode,
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
        "dominant_instrument": CLASSES[top_index]
    }

    st.download_button(
        label="Download JSON Report",
        data=json.dumps(report_data, indent=4),
        file_name="instrument_analysis.json",
        mime="application/json"
    )

    # =========================
    # PDF GENERATION
    # =========================
    if st.button("Generate Professional PDF Report"):

        pdf_filename = "InstruNet_AI_Report.pdf"

        # Save figures temporarily
        fig_wave.savefig("waveform.png")
        fig_spec.savefig("spectrogram.png")
        fig_pie.savefig("pie.png")
        fig_timeline.savefig("timeline.png")

        doc = SimpleDocTemplate(pdf_filename)
        elements = []
        styles = getSampleStyleSheet()

        # Logo
        if os.path.exists("logo.png"):
            elements.append(Image("logo.png", width=1.5 * inch, height=1.5 * inch))
            elements.append(Spacer(1, 0.3 * inch))

        elements.append(Paragraph("<b>InstruNet AI - Instrument Analysis Report</b>", styles["Title"]))
        elements.append(Spacer(1, 0.4 * inch))

        elements.append(Paragraph(f"<b>Dominant Instrument:</b> {CLASSES[top_index]}", styles["Heading2"]))
        elements.append(Spacer(1, 0.4 * inch))

        table_data = [["Instrument", "Probability (%)"]]

        for i, instrument in enumerate(CLASSES):
            table_data.append([instrument, f"{avg_prediction[i]*100:.2f}"])

        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))

        elements.append(table)
        elements.append(Spacer(1, 0.5 * inch))

        elements.append(Image("timeline.png", width=5.5 * inch, height=3 * inch))
        elements.append(Spacer(1, 0.4 * inch))

        elements.append(Image("waveform.png", width=5.5 * inch, height=3 * inch))
        elements.append(Spacer(1, 0.4 * inch))

        elements.append(Image("spectrogram.png", width=5.5 * inch, height=3 * inch))
        elements.append(Spacer(1, 0.4 * inch))

        elements.append(Image("pie.png", width=4 * inch, height=4 * inch))

        doc.build(elements)

        with open(pdf_filename, "rb") as pdf_file:
            st.download_button(
                label="Download Full AI PDF Report",
                data=pdf_file,
                file_name="InstruNet_AI_Report.pdf",
                mime="application/pdf"
            )

        # Cleanup
        for file in ["waveform.png", "spectrogram.png", "pie.png", "timeline.png"]:
            if os.path.exists(file):
                os.remove(file)
