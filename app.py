import os
import tempfile
import yt_dlp
from flask import Flask, request, jsonify
from google import genai
from google.genai import types
import traceback
import sys

app = Flask(__name__)
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Error: Missing GEMINI_API_KEY environment variable.")
    sys.exit(1)  # Exit the server if API key is not set

client = genai.Client(api_key=api_key)

client = genai.Client(api_key=api_key)
# -------------------
# Download mp3
# -------------------
def download_audio_mp3(youtube_url: str, output_dir: str) -> str:
    """Download audio directly as WAV using yt-dlp."""
    ydl_opts = {
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'format': 'bestaudio/best',
        'quiet': False, 
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '64',
        }]
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        title = info.get('title', 'audio')
        return os.path.join(output_dir, f"{title}.mp3")

# -------------------
# Transcribe audio
# ----------------

def transcribe_audio(client, audio_path: str) -> str:
    with open(audio_path, 'rb') as f:
        audio_bytes = f.read()

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[
            "Transcribe this audio clip exactly as it is.",
            types.Part.from_bytes(
                data=audio_bytes,
                mime_type='audio/mp3',
            )
        ]
    )
    return response.text.strip()

# -------------------
# Generate quiz     
# -------------------

def generate_mcqs(client,transcript_text, num_questions=10):
    prompt = f"""
You are an AI assistant designed to generate high-quality, professional multiple-choice questions (MCQs) from educational video content.

Your task is to:
- Understand the core concepts from the transcript of a YouTube video.
- Generate concept-based MCQs suitable for undergraduate-level Computer Science students.
- Avoid referring to the transcript directly (e.g., do not use "according to the transcript" or "according to the text" or “according to the video” or “as stated above”).
- Make questions sound natural and exam-ready.

Each question must:
- Be clear and technically accurate.
- Have exactly one correct answer and three related but wrong distractions.
- Target medium-difficulty level.
- Be based on topics typically found in BTech CSE, such as programming, data structures, algorithms, machine learning, and academic subjects also like engineering physics, engineering mathematics, engineering chemistry, fundamentals of electronics and electrical etc.

---

Here are some example questions for reference:

Example 1:
1. What is the time complexity of inserting an element into a max-heap?
    A. O(log n)  
    B. O(1)  
    C. O(n log n)  
    D. O(n²)  
Answer: A

Example 2:
2. Which of the following is NOT a valid use-case for a hash table?
    A. Implementing a dictionary  
    B. Storing hierarchical data like XML  
    C. Caching data  
    D. Checking for duplicates in a list  
Answer: B

Example 3:
3. In supervised learning, which of the following best defines the "training dataset"?
    A. A dataset used only to test the model’s performance  
    B. A dataset containing input-output pairs used to teach the model  
    C. A dataset with only input features and no labels  
    D. A dataset created from real-time user interactions  
Answer: B

---

Now, generate 10 MCQs based on the following transcript:

Transcript:
\"\"\"{transcript_text}\"\"\"

Output format:
1. Question?
    A. Option1
    B. Option2
    C. Option3
    D. Option4
Answer: A
"""

    try:
        model = genai.GenerativeModel(model_name="gemini-2.0-flash")
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents = [prompt]
            )
        return response.text.strip()
    except Exception as e:
        print("Error generating MCQs:", str(e))
        traceback.print_exc()
        return ""

@app.route("/transcribe", methods=["POST"])
def transcribe_youtube():
    youtube_url = request.form.get("youtube_url")
    if not youtube_url:
        return jsonify({"error": "Missing YouTube URL"}), 400

    # api_key = os.getenv("GEMINI_API_KEY")
    # if not api_key:
    #     return jsonify({"error": "Missing GEMINI_API_KEY"}), 500

    # client = genai.Client(api_key=api_key)

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = download_audio_mp3(youtube_url, temp_dir)
            if not os.path.exists(audio_path):
                return jsonify({"error": "Audio download failed"}), 500

            # 2. Transcribe
            response = transcribe_audio(client, audio_path)

            transcript = response.text.strip()
            return jsonify({"youtube_url": youtube_url, "transcript": transcript})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/youtube-quiz", methods=["POST"])
def youtube_quiz():
    youtube_url = request.form.get("youtube_url")
    if not youtube_url:
        return jsonify({"error": "Missing YouTube URL"}), 400

    # api_key = os.getenv("GEMINI_API_KEY")
    # if not api_key:
    #     return jsonify({"error": "Missing GEMINI_API_KEY"}), 500

    # client = genai.Client(api_key=api_key)

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # 1. Download audio
            audio_path = download_audio_mp3(youtube_url, temp_dir)
            if not os.path.exists(audio_path):
                return jsonify({"error": "Audio download failed"}), 500

            # 2. Transcribe
            transcript = transcribe_audio(client, audio_path)

            # 3. Generate quiz
            quiz = generate_mcqs(client, transcript)

            return jsonify({
                "youtube_url": youtube_url,
                "transcript": transcript,
                "quiz": quiz
            })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)