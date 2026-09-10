"""Gemini native-audio speech, shared by the API and the live coach.

Both entry points needed the same call and had their own copy of it. One copy
means the model id, the voice, and the PCM handling can only ever be changed in
one place.

WHY NOT gemini-2.5-flash: that model is text-out only. Asking it for
response_modalities=["AUDIO"] fails -- audio generation lives in a separate
variant of the same family, gemini-2.5-flash-preview-tts, which is what this
targets. Both are 2.5 Flash; only the TTS variant can return a waveform.

The model streams back raw 24 kHz 16-bit mono PCM with no container. That is
not playable as-is, so pcm_to_wav() puts a RIFF header on it. Doing this
natively rather than through a text-to-speech service is what removes the
second network hop -- the audio comes straight out of the same request.
"""

import io
import os
import threading
import wave

# Model ids move. Change them here, nowhere else.
TTS_MODEL = "gemini-2.5-flash-preview-tts"
VOICE = "Kore"

# What the API documents itself as returning. If a future model changes any of
# these, the WAV header goes out of step with the samples and the audio plays
# at the wrong pitch rather than failing outright -- so keep them together.
PCM_RATE = 24000
PCM_WIDTH = 2          # 16-bit
PCM_CHANNELS = 1       # mono

_client = None
_client_lock = threading.Lock()


def api_key():
    """The configured key, or None. GEMINI_API_KEY wins over GOOGLE_API_KEY."""
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def available():
    """True when a speech call could actually succeed."""
    if not api_key():
        return False
    try:
        import google.genai  # noqa: F401
    except ImportError:
        return False
    return True


def client():
    """One cached client for the process.

    live_coach.py used to build a Client per utterance. Construction sets up
    auth and an HTTP session, so paying that on every spoken line added latency
    to the exact path that is supposed to be low latency.
    """
    global _client
    with _client_lock:
        if _client is None:
            key = api_key()
            if not key:
                raise RuntimeError("no GEMINI_API_KEY or GOOGLE_API_KEY set")
            from google import genai
            _client = genai.Client(api_key=key)
        return _client


def reset_client():
    """Drop the cached client, so a changed key is picked up without a restart."""
    global _client
    with _client_lock:
        _client = None


def pcm_to_wav(pcm, rate=PCM_RATE):
    """Raw PCM -> WAV bytes, in memory.

    In memory, not a temp file: a browser is handed the bytes directly, and a
    shared path on disk would race between concurrent callers.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(PCM_CHANNELS)
        w.setsampwidth(PCM_WIDTH)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def synthesize(text, voice=VOICE, model=TTS_MODEL):
    """Blocking. text -> WAV bytes. Raises if the call fails.

    Returns a complete WAV, not PCM, so every caller gets something playable
    without repeating the header logic.
    """
    from google.genai import types

    resp = client().models.generate_content(
        model=model,
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice
                    )
                )
            ),
        ),
    )

    # Reach for the audio defensively: a safety block or a text-only reply
    # leaves candidates present but inline_data absent, and the bare index
    # chain would then raise an IndexError that says nothing useful.
    try:
        parts = resp.candidates[0].content.parts
    except (AttributeError, IndexError, TypeError):
        raise RuntimeError("Gemini returned no audio candidate") from None
    for part in parts or []:
        inline = getattr(part, "inline_data", None)
        if inline is not None and getattr(inline, "data", None):
            return pcm_to_wav(inline.data)
    raise RuntimeError("Gemini response contained no audio data")
