
import os
import re
import torch
import tempfile
import numpy as np
import sounddevice as sd
from flask import Flask, render_template, jsonify, request
from gtts import gTTS
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa

from src.utils.tokenizer import load_nepali_tokenizer, load_english_tokenizer
from src.model.model import build_transformer
from src.utils.translate import translate

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 1. Load Wav2Vec2 Nepali ASR ──────────────────────────────
print("Loading Wav2Vec2 Nepali model...")
MODEL_NAME = "anish-shilpakar/wav2vec2-nepali"
processor  = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
stt_model  = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).to(device)
stt_model.eval()
print("Wav2Vec2 loaded!")

# ── 2. Load Transformer ──────────────────────────────────────
print("Loading Transformer model...")
ne_tokenizer = load_nepali_tokenizer()
en_tokenizer = load_english_tokenizer()

model = build_transformer(
    src_vocab_size=ne_tokenizer.get_vocab_size(),
    tgt_vocab_size=en_tokenizer.get_vocab_size(),
    src_seq_len=128,
    tgt_seq_len=128,
    d_model=512,
    N=4,
    h=8,
    dropout=0.3,
    d_ff=2048
)

_PROJECT_ROOT   = os.path.dirname(os.path.abspath(__file__))
checkpoint_path = os.path.join(_PROJECT_ROOT, "models", "best_nepali_to_eng.pth")

if not os.path.isfile(checkpoint_path):
    raise FileNotFoundError(
        f"Checkpoint not found: {checkpoint_path}. "
        f"Train the model first with: python -m src.train"
    )

try:
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
except TypeError:
    state_dict = torch.load(checkpoint_path, map_location=device)

model.load_state_dict(state_dict)
model.to(device)
model.eval()
print("Transformer model loaded!")


# ── Helper functions ─────────────────────────────────────────

def contains_english(text):
    """Return True if text contains any English alphabet characters."""
    return bool(re.search(r'[a-zA-Z]', text))


def clean_nepali_input(text):
    """Remove extra dandas and whitespace from Nepali input."""
    text = text.strip()
    text = re.sub(r'।+', '।', text)
    return text


def transcribe_audio(audio_path):
    """Wav2Vec2: Nepali audio file → Nepali text"""
    audio, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    ).to(device)
    with torch.no_grad():
        logits = stt_model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(predicted_ids)[0]


# ── Flask routes ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/translate", methods=["POST"])
def run_pipeline():
    try:
        print("\n" + "=" * 50)

        # Step 1: Get audio file
        if "audio" not in request.files:
            return jsonify({
                "success": False,
                "error": "No audio file provided"
            })

        f        = request.files["audio"]
        tmp_path = tempfile.mktemp(suffix=".wav")
        f.save(tmp_path)

        # Step 2: Nepali speech → Nepali text (Wav2Vec2)
        nepali_text = transcribe_audio(tmp_path)
        os.remove(tmp_path)
        print(f"Nepali    : {nepali_text}")

        # Step 3: Validate transcription
        if not nepali_text or nepali_text.strip() == "":
            return jsonify({
                "success": False,
                "error": "Could not hear anything. Please speak louder and try again."
            })

        if contains_english(nepali_text):
            return jsonify({
                "success": False,
                "error": "English detected! Please speak in Nepali only."
            })

        # Step 4: Clean Nepali input
        cleaned_text = clean_nepali_input(nepali_text)
        print(f"Cleaned   : {cleaned_text}")

        # Step 5: Nepali text → English text (Transformer + beam search)
        result = translate(
            model, cleaned_text,
            ne_tokenizer, en_tokenizer,
            device,
            beam_width=5       # beam search for better quality
        )
        print(f"Translated: {result}")

        # Step 6: Validate translation
        if not result or result.strip() == "":
            return jsonify({
                "success": False,
                "error": "Translation failed. Please try again."
            })

        # Step 7: English text → English audio (gTTS)
        static_dir  = os.path.join(_PROJECT_ROOT, "static")
        os.makedirs(static_dir, exist_ok=True)
        output_path = os.path.join(static_dir, "output.mp3")
        tts = gTTS(text=result.strip(), lang="en", slow=False)
        tts.save(output_path)
        print(f"Audio saved!")
        print("=" * 50)

        return jsonify({
            "success":      True,
            "nepali_text":  nepali_text,
            "english_text": result,
            "audio_url":    "/static/output.mp3"
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({
            "success": False,
            "error":   str(e)
        })


# ── Run ──────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(os.path.join(_PROJECT_ROOT, "static"),    exist_ok=True)
    os.makedirs(os.path.join(_PROJECT_ROOT, "templates"), exist_ok=True)
    app.run(debug=True, port=5000)
