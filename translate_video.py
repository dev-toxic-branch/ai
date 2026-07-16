"""Translate video audio from English to French using local models.

Uses:
- Whisper base for transcription
- Helsinki-NLP/opus-mt-en-fr for translation
"""

import tempfile
import wave
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).parent
VIDEO_PATH = BASE_DIR / "Deep sea creatures are found _ Ocean Sunfish Vs Sea Eater.mp4"
OUTPUT_PATH = BASE_DIR / "deep_sea_creatures_french_translation.txt"


def extract_and_transcribe(video_path):
    """Extract audio and transcribe with Whisper."""
    import whisper
    import subprocess
    from moderation import _ffmpeg_exe

    print("Loading Whisper model...")
    model = whisper.load_model("base", device="cpu")

    print("Extracting audio...")
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "audio.wav"
        subprocess.run(
            [_ffmpeg_exe(), "-y", "-i", str(video_path),
             "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(wav)],
            capture_output=True, timeout=300, check=True,
        )
        with wave.open(str(wav), "rb") as w:
            audio = np.frombuffer(w.readframes(w.getnframes()), np.int16)
        audio = audio.astype(np.float32) / 32768.0

    print("Transcribing...")
    result = model.transcribe(audio, fp16=False)
    return result["text"].strip()


def translate_to_french(text):
    """Translate English text to French using MarianMT."""
    from transformers import MarianMTModel, MarianTokenizer

    model_name = "Helsinki-NLP/opus-mt-en-fr"
    print(f"Loading translation model: {model_name}...")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # MarianMT has a max input length, so split into chunks if needed
    max_chunk = 512
    chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]

    translated_chunks = []
    for i, chunk in enumerate(chunks):
        print(f"Translating chunk {i+1}/{len(chunks)}...")
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
        translated = model.generate(**inputs)
        result = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
        translated_chunks.append(result)

    return " ".join(translated_chunks)


def main():
    if not VIDEO_PATH.exists():
        print(f"Error: Video not found at {VIDEO_PATH}")
        return

    # Step 1: Transcribe
    english_text = extract_and_transcribe(VIDEO_PATH)
    print(f"\nEnglish transcript:\n{english_text[:500]}...\n")

    # Step 2: Translate
    french_text = translate_to_french(english_text)
    print(f"\nFrench translation:\n{french_text[:500]}...\n")

    # Step 3: Save to file
    output = f"""Deep Sea Creatures Video - English to French Translation
{'='*60}

ENGLISH TRANSCRIPT:
{english_text}

{'='*60}

FRENCH TRANSLATION:
{french_text}
"""
    OUTPUT_PATH.write_text(output, encoding="utf-8")
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
