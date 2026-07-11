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


def text_to_speech(text, output_path, lang="fr"):
    """Generate speech audio from text using edge-tts."""
    import asyncio
    import edge_tts

    voice_map = {
        "fr": "fr-FR-DeniseNeural",
        "en": "en-US-JennyNeural",
        "es": "es-ES-ElviraNeural",
        "de": "de-DE-KatjaNeural",
        "ar": "ar-SA-ZariyahNeural",
    }
    voice = voice_map.get(lang, f"{lang}-{lang.upper()}-FemaleNeural")

    async def _generate():
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(output_path))

    asyncio.run(_generate())
    return output_path


def has_video_stream(file_path):
    """Check if file has a video stream."""
    from moderation import _ffmpeg_exe

    probe = subprocess.run(
        [_ffmpeg_exe(), "-i", str(file_path)],
        capture_output=True, text=True, timeout=60,
    )
    return "Video:" in probe.stderr


def replace_audio(video_path, audio_path, output_path):
    """Replace video's audio track with new audio using ffmpeg."""
    from moderation import _ffmpeg_exe

    # First convert TTS audio to wav format compatible with video
    with tempfile.TemporaryDirectory() as td:
        wav_audio = Path(td) / "audio.wav"
        subprocess.run(
            [_ffmpeg_exe(), "-y", "-i", str(audio_path),
             "-ar", "44100", "-ac", "2", "-f", "wav", str(wav_audio)],
            capture_output=True, timeout=120, check=True,
        )

        # Check if input has video stream
        if has_video_stream(video_path):
            # Has video - replace audio track
            cmd = [
                _ffmpeg_exe(), "-y",
                "-i", str(video_path),
                "-i", str(wav_audio),
                "-c:v", "copy",
                "-map", "0:v:0", "-map", "1:a:0",
                "-shortest",
                "-loglevel", "error",
                str(output_path),
            ]
        else:
            # Audio-only - just copy the translated audio
            cmd = [
                _ffmpeg_exe(), "-y",
                "-i", str(wav_audio),
                "-c:a", "libmp3lame", "-q:a", "2",
                str(output_path),
            ]

        subprocess.run(cmd, capture_output=True, timeout=300, check=True)

    return output_path


def run_dubbing_pipeline(video_path, output_path, target_lang="fr"):
    """Full pipeline: transcribe -> translate -> TTS -> dubbed video.

    Returns the path to the dubbed video file.
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

    # Step 3: Generate French speech
    print("Generating French audio...")
    with tempfile.TemporaryDirectory() as td:
        tts_audio = Path(td) / "french_speech.mp3"
        text_to_speech(translated, tts_audio, lang=target_lang)
        print(f"TTS audio generated: {tts_audio}")

        # Step 4: Replace video audio
        print("Creating dubbed video...")
        output = Path(output_path)
        replace_audio(video_path, tts_audio, output)
        print(f"Dubbed video saved to: {output}")

    # Also save text translation
    txt_output = Path(str(output_path).rsplit(".", 1)[0] + ".txt")
    txt_output.write_text(
        f"Original: {transcript}\n\nTranslated ({target_lang}): {translated}",
        encoding="utf-8",
    )

    return str(output)
