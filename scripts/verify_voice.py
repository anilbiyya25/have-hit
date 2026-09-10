"""Prove the Gemini native-audio path works end to end.

Reads the key from the environment, synthesises one short coaching line, writes
a WAV, and validates the header rather than trusting that bytes came back.

    .\\run.ps1 scripts\\verify_voice.py
    .\\run.ps1 scripts\\verify_voice.py --play          # also play it
    .\\run.ps1 scripts\\verify_voice.py --text "..."    # custom line

The key is never printed. Only its length and last four characters are shown,
which is enough to tell two keys apart without putting one in a terminal
scrollback or a screen share.
"""

import argparse
import io
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, ".")
import voice

DEFAULT_TEXT = ("Nice work. Keep your chest up and drive through the heels on "
                "the next repetition.")
OUT_PATH = Path("checkpoints") / "voice_check.wav"


def fail(msg, fix=None):
    print(f"\n  FAILED: {msg}")
    if fix:
        print(f"  Fix   : {fix}")
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--play", action="store_true", help="play the result")
    ap.add_argument("--voice", default=voice.VOICE)
    args = ap.parse_args()

    print("=" * 62)
    print("GEMINI NATIVE AUDIO -- VERIFICATION")
    print("=" * 62)

    # ---- 1. key -----------------------------------------------------------
    key = voice.api_key()
    if not key:
        return fail(
            "no API key in the environment",
            'set it, then open a NEW shell:\n'
            '          [Environment]::SetEnvironmentVariable('
            '"GEMINI_API_KEY", "<key>", "User")',
        )
    import os
    which = "GEMINI_API_KEY" if os.environ.get("GEMINI_API_KEY") else "GOOGLE_API_KEY"
    print(f"  key source     : {which}")
    print(f"  key            : {len(key)} chars, ends ...{key[-4:]}")

    # ---- 2. sdk -----------------------------------------------------------
    try:
        import google.genai as genai
        ver = getattr(genai, "__version__", "unknown")
    except ImportError:
        return fail("google-genai is not installed",
                    r"D:\have-hit\venv\Scripts\pip.exe install google-genai")
    print(f"  google-genai   : {ver}")
    print(f"  model          : {voice.TTS_MODEL}")
    print(f"  voice          : {args.voice}")

    # ---- 3. synthesise ----------------------------------------------------
    print(f'\n  synthesising   : "{args.text[:52]}..."')
    t0 = time.time()
    try:
        wav = voice.synthesize(args.text, voice=args.voice)
    except Exception as exc:                                  # noqa: BLE001
        name = exc.__class__.__name__
        msg = str(exc)
        hint = None
        low = msg.lower()
        if "api key" in low or "unauthenticated" in low or "401" in low:
            hint = "the key was rejected -- check it is an AI Studio key"
        elif "permission" in low or "403" in low:
            hint = "key valid but lacks access to this model"
        elif "quota" in low or "429" in low:
            hint = "rate limited or out of quota; retry shortly"
        elif "not found" in low or "404" in low:
            hint = (f"{voice.TTS_MODEL} was not found -- model ids move, "
                    "update TTS_MODEL in voice.py")
        return fail(f"{name}: {msg[:200]}", hint)
    dt = time.time() - t0

    # ---- 4. validate ------------------------------------------------------
    if not wav or wav[:4] != b"RIFF" or wav[8:12] != b"WAVE":
        return fail("returned bytes are not a RIFF/WAVE file")
    with wave.open(io.BytesIO(wav), "rb") as w:
        ch, width, rate, frames = (w.getnchannels(), w.getsampwidth(),
                                   w.getframerate(), w.getnframes())
    seconds = frames / rate if rate else 0
    if frames == 0:
        return fail("WAV parsed but contains zero audio frames")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_bytes(wav)

    print(f"\n  latency        : {dt:.2f}s")
    print(f"  bytes          : {len(wav):,}")
    print(f"  format         : {ch}ch {width * 8}-bit {rate} Hz")
    print(f"  duration       : {seconds:.2f}s ({frames:,} frames)")
    print(f"  written        : {OUT_PATH}")

    expected = (voice.PCM_CHANNELS, voice.PCM_WIDTH, voice.PCM_RATE)
    if (ch, width, rate) != expected:
        print(f"\n  !!! format is {(ch, width, rate)}, voice.py declares "
              f"{expected}.")
        print("      Playback pitch will be wrong. Update the constants there.")

    if args.play:
        try:
            import winsound
            print("\n  playing...")
            winsound.PlaySound(str(OUT_PATH), winsound.SND_FILENAME)
        except Exception as exc:                              # noqa: BLE001
            print(f"  (playback unavailable: {exc})")

    print("\n" + "=" * 62)
    print("  SUCCESS -- native Gemini audio is active.")
    print("=" * 62)
    print("  The API server picks this up on restart: /health reports")
    print('  "gemini": true and the dashboard voice button goes live.')
    return 0


if __name__ == "__main__":
    sys.exit(main())
