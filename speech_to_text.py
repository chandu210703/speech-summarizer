import warnings
warnings.filterwarnings("ignore")

import os
import tempfile
import streamlit as st
import whisper
from openai import OpenAI

# ================= PAGE CONFIG =================
st.set_page_config(page_title="AI Speech Summarizer", layout="wide")

# ================= CSS =================
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ================= HEADER =================
st.markdown(
    """
    <h1 style="text-align:center;">🎤 AI Speech Summarizer</h1>
    <p style="text-align:center; font-size:18px;">
        Upload an audio file to get transcription and AI summary
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ================= CONFIG (SECURE) =================
HF_API_KEY = os.environ.get("HF_API_KEY")

if not HF_API_KEY:
    st.error("HF_API_KEY is not set. Please configure environment variable.")
    st.stop()

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_API_KEY
)

# ================= LOAD WHISPER =================
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

whisper_model = load_whisper()

# ================= FUNCTIONS =================
def speech_to_text(audio_path):
    result = whisper_model.transcribe(audio_path, task="transcribe", language=None)
    return result["text"].strip(), result.get("language", "unknown")

def summarize_transcript(transcript):
    transcript = transcript[:2000]

    prompt = f"""
Read the following transcript and generate one clean paragraph summary.

Transcript:
{transcript}

Summary:
"""

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-1B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1000
    )

    return response.choices[0].message.content.strip()

# ================= UI =================
left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("📤 Upload Audio")
    uploaded_file = st.file_uploader(
        "Choose audio file",
        type=["mp3", "wav", "m4a", "flac", "ogg", "webm"]
    )

    if uploaded_file:
        st.audio(uploaded_file)

with right_col:
    if uploaded_file:
        with st.spinner("Processing audio..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
                tmp.write(uploaded_file.read())
                audio_path = tmp.name

            transcript, language = speech_to_text(audio_path)
            summary = summarize_transcript(transcript)

            os.remove(audio_path)

        st.success("Processing completed!")

        st.subheader("🌍 Detected Language")
        st.write(language.upper())

        st.subheader("📄 Transcript")
        st.text_area("", transcript, height=300)

        st.subheader("✨ Summary")
        st.text_area("", summary, height=200)

        st.download_button(
            "📥 Download Result",
            f"TRANSCRIPT\n{'='*60}\n{transcript}\n\nSUMMARY\n{'='*60}\n{summary}",
            file_name="speech_summary.txt"
        )
