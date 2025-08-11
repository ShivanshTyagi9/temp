import os
import tempfile
import yt_dlp
from flask import Flask, request, jsonify
from google import genai
from google.genai import types

app = Flask(__name__)

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

@app.route("/transcribe", methods=["POST"])
def transcribe_youtube():
    youtube_url = request.form.get("youtube_url")
    if not youtube_url:
        return jsonify({"error": "Missing YouTube URL"}), 400

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return jsonify({"error": "Missing GEMINI_API_KEY"}), 500

    client = genai.Client(api_key=api_key)

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = download_audio_wav(youtube_url, temp_dir)
            if not os.path.exists(audio_path):
                return jsonify({"error": "Audio download failed"}), 500

            with open(audio_path, 'rb') as f:
                audio_bytes = f.read()

            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[
                    'Transcribe this audio clip exactly as it is',
                    types.Part.from_bytes(
                        data=audio_bytes,
                        mime_type='audio/wav',
                    )
                ]
            )

            transcript = response.text.strip()
            return jsonify({"youtube_url": youtube_url, "transcript": transcript})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)