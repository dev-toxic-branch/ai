"""Video dubbing pipeline - translates audio from one language to another.

Uses:
- Whisper for transcription
- Helsinki-NLP MarianMT for translation
- edge-tts or similar for TTS (if available)
"""

import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np


def translate_segments(transcript, config):
    """Translate English transcript to target language using MarianMT directly.

    The 'translation' pipeline task was removed in newer transformers versions.
    This uses MarianMTModel and MarianTokenizer directly as a workaround.
    """
    from transformers import MarianMTModel, MarianTokenizer

    target_lang = config.get("target_lang", "fr")
    model_name = f"Helsinki-NLP/opus-mt-en-{target_lang}"

    print(f"Loading translation model: {model_name}...")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # MarianMT has max input length, split into chunks if needed
    max_chunk = 512
    text = transcript if isinstance(transcript, str) else str(transcript)
    chunks = [text[i:i + max_chunk] for i in range(0, len(text), max_chunk)]

    translated_chunks = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
        translated = model.generate(**inputs)
        result = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
        translated_chunks.append(result)

    return " ".join(translated_chunks)


def extract_audio(video_path, output_wav):
    """Extract mono 16kHz audio from video using ffmpeg."""
    from moderation import _ffmpeg_exe

    subprocess.run(
        [_ffmpeg_exe(), "-y", "-i", str(video_path),
         "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(output_wav)],
        capture_output=True, timeout=300, check=True,
    )


def transcribe_audio(video_path):
    """Extract audio and transcribe with Whisper."""
    import whisper

    model = whisper.load_model("base", device="cpu")

    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "audio.wav"
        extract_audio(video_path, wav)
        with wave.open(str(wav), "rb") as w:
            audio = np.frombuffer(w.readframes(w.getnframes()), np.int16)
        audio = audio.astype(np.float32) / 32768.0

    result = model.transcribe(audio, fp16=False)
    return result["text"].strip()


def run_dubbing_pipeline(video_path, output_path, target_lang="fr"):
    """Full pipeline: transcribe -> translate -> (optionally) synthesize.

    For now, returns translated text. TTS synthesis can be added later.
    """
    config = {"target_lang": target_lang}

    # Step 1: Transcribe
    print("Transcribing video...")
    transcript = transcribe_audio(video_path)
    print(f"Transcript: {transcript[:200]}...")

    # Step 2: Translate
    print("Translating...")
    translated = translate_segments(transcript, config)
    print(f"Translation: {translated[:200]}...")

    # Step 3: Save result
    output = Path(output_path)
    output.write_text(
        f"Original: {transcript}\n\nTranslated ({target_lang}): {translated}",
        encoding="utf-8",
    )
    print(f"Saved to: {output}")

    return translated
