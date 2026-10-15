"""Whisper a training clip's speech for its caption (the Captions tab's Append speech; the desktop's
"Append Transcription"). Its own process, so the web server never loads torch. Prints one JSON line:
{"text": "..."} or {"error": "..."}.

Gizmo's Whisper, without importing Gizmo (it needs tkinter, which a pod's image doesn't have): the same model
(openai/whisper-base, local snapshot first, so the Preferences downloader's copy works offline), ffmpeg
(imageio-ffmpeg's, then PATH), language (Gizmo's saved choice, English when none - auto-detect's wrong guess on a
short clip is what starts Whisper looping) and loop check (gizmo.Gizmo._whisper_degenerate).
"""
import json
import os
import subprocess
import sys
import tempfile
import wave

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

WHISPER_REPO = "openai/whisper-base"


def _ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        import shutil
        return shutil.which("ffmpeg")


def _local_whisper():
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(WHISPER_REPO, local_files_only=True,
                                 allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors"])
    except Exception:
        return None


def _degenerate(text, span):
    """More characters than the clip can hold (~30 a second), or a loop of the same few words."""
    words = text.split()
    if len(text) > max(80, span * 30):
        return True
    return len(words) >= 10 and len(set(w.lower() for w in words)) / len(words) < 0.4


def _language():
    try:
        from fizgig.user_data import user_file
        with open(user_file("gizmo_settings.json"), encoding="utf-8") as f:
            lang = json.load(f).get("whisper_language")
    except Exception:
        lang = None
    lang = lang or "English"
    return None if lang == "Auto detect" else lang.lower()


def transcribe(path):
    ff = _ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg not found")
    wav = os.path.join(tempfile.gettempdir(), f"fizgig_caption_whisper_{os.getpid()}.wav")
    kw = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    p = subprocess.run([ff, "-y", "-hide_banner", "-loglevel", "error", "-i", path, "-vn", "-ac", "1", "-ar",
                        "16000", "-t", "120", "-c:a", "pcm_s16le", wav], capture_output=True, **kw)
    if p.returncode != 0:
        raise RuntimeError("couldn't read the clip's sound")
    try:
        with wave.open(wav, "rb") as w:
            frames = w.readframes(w.getnframes())
            span = w.getnframes() / float(w.getframerate() or 16000)
    finally:
        try:
            os.remove(wav)
        except OSError:
            pass
    import numpy as np
    from transformers import pipeline
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    pipe = pipeline("automatic-speech-recognition", model=_local_whisper() or WHISPER_REPO, device=-1)

    def hear(language):
        kw = {"generate_kwargs": {"language": language, "task": "transcribe"}} if language else {}
        return (pipe({"array": samples, "sampling_rate": 16000}, chunk_length_s=30, **kw).get("text") or "").strip()

    lang = _language()
    text = hear(lang)
    if lang is None and _degenerate(text, span):
        text = hear("english")
    if _degenerate(text, span):
        raise RuntimeError("Whisper looped on this clip (it made up repeating text). Pick the clip's language in "
                           "Gizmo's settings, or type the words.")
    return text


if __name__ == "__main__":
    try:
        out = {"text": transcribe(sys.argv[1])}
    except Exception as e:
        out = {"error": str(e) or type(e).__name__}
    print(json.dumps(out), flush=True)
