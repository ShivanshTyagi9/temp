import os
import tempfile
import yt_dlp
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI(title="YouTube Audio Transcriber (Gemini 2.5 Flash Test)")

def download_audio_wav(youtube_url: str, output_dir: str) -> str:
    """Download audio directly as WAV using yt-dlp."""
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        title = info.get('title', 'audio')
        return os.path.join(output_dir, f"{title}.wav")

@app.post("/transcribe")
def transcribe_youtube(youtube_url: str = Form(...)):
    """Download audio from YouTube and transcribe using Gemini 2.5 Flash."""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Step 1: Download audio directly as .wav
            audio_path = download_audio_wav(youtube_url, temp_dir)
            if not os.path.exists(audio_path):
                return JSONResponse(content={"error": "Audio download failed"}, status_code=500)

            # Step 2: Read audio bytes
            with open(audio_path, 'rb') as f:
                audio_bytes = f.read()

            # Step 3: Transcribe with Gemini 2.5 Flash
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[
                    'Transcrive this audio clip exactly as it is',
                    types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type='audio/wav',
                    )
                ]
            )

            transcript = response.text.strip()

            return {"youtube_url": youtube_url, "transcript": transcript}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
