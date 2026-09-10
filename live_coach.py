"""Live webcam exercise coach.  [DEPRECATED as a product path]

DEPRECATION
    The product moved to one asynchronous architecture: record or upload a
    whole clip, POST it to /api/v1/analyze-workout, get a full breakdown back.
    The rolling-window live loop this file implements is no longer part of it,
    and the server-side /predict endpoint it mirrors has been removed.

    The file is kept, and still runs, for two reasons that are not the product:

      --selftest  verifies encoder + head + label alignment end to end without
                  a camera. That is the fastest way to tell whether a
                  checkpoint is sane, and nothing has replaced it.
      the loop    is the only remaining way to watch the 27-class head react in
                  real time, which is useful when judging whether a retrain
                  actually helped.

    Do not build on it. It is a diagnostic, not a feature, and the rolling
    1500 ms window is exactly the design the pivot moved away from.

Rolling webcam frames -> frozen V-JEPA 2.1 -> trained head -> spoken coaching.

    .\\run.ps1 live_coach.py                 # start coaching
    .\\run.ps1 live_coach.py --selftest      # verify the pipeline, no camera
    .\\run.ps1 live_coach.py --voice off     # on-screen only
    .\\run.ps1 live_coach.py --voice gemini  # force Gemini TTS

Press q (or Esc) in the preview window to quit.

Three threads, because a naive single-threaded loop stutters badly:
  main       capture + preview, must stay at camera framerate
  inference  ~200 ms per clip on this GPU, so it cannot block the preview
  voice      a TTS round trip is far slower still, so it gets its own queue

Robustness at inference time is TTA plus adaptive contrast, NOT augmentation.
Augmentation is a training-time tool: it perturbs inputs so a model learns
invariance while its weights update. Nothing trains here -- the encoder is
frozen and the head is already fitted -- so a random perturbation per clip
would only make the same pose score differently from one moment to the next.
Instead the mirror is applied deterministically and both views are scored,
which uses the same symmetry to steady predictions rather than shake them.

Label decoding goes through config.ID_TO_LABEL, and the checkpoint's sidecar
classes.json is checked against it at startup. Those two disagreed once
already (ids came from sorted directory order, config listed them in a
different order), which silently mislabelled every prediction.
"""

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, ".")
import config
import voice
from models.classifier import load_classifier

# The speech model, voice name and PCM format now live in voice.py, shared with
# the API server. Note gemini-2.5-flash itself cannot emit audio -- the TTS
# variant of the same family can, and that is what voice.py targets.

CLASSES_SIDECAR = Path(config.CHECKPOINT_DIR) / "classes.json"


# --------------------------------------------------------------------------
# Voice
# --------------------------------------------------------------------------

class Voice:
    """Speaks short coaching lines, newest-wins, never blocking the capture loop.

    Two backends. Gemini is the requested one; Windows SAPI is the fallback so
    the coach still talks on a machine with no API key. Only the coaching
    sentence is sent to Google -- never webcam frames.
    """

    def __init__(self, backend="auto"):
        self.q = queue.Queue(maxsize=1)
        self.backend = self._resolve(backend)
        self._warned = False
        if self.backend != "off":
            threading.Thread(target=self._run, daemon=True).start()

    def _resolve(self, backend):
        if backend == "off":
            return "off"

        has_key = bool(voice.api_key())
        try:
            import google.genai  # noqa: F401
            has_sdk = True
        except ImportError:
            has_sdk = False

        if backend == "gemini":
            if has_sdk and has_key:
                return "gemini"
            missing = []
            if not has_sdk:
                missing.append("google-genai not installed "
                               "(pip install google-genai)")
            if not has_key:
                missing.append("GEMINI_API_KEY not set")
            print(f"  !!! --voice gemini unavailable: {'; '.join(missing)}")
            print("      falling back to the system voice")
            return "system"

        if has_sdk and has_key:
            return "gemini"
        return "system"

    def say(self, text):
        """Queue a line. Drops it if one is already pending -- stale coaching
        is worse than silence."""
        if self.backend == "off":
            return
        try:
            self.q.put_nowait(text)
        except queue.Full:
            pass

    def _run(self):
        while True:
            text = self.q.get()
            try:
                if self.backend == "gemini":
                    self._gemini(text)
                else:
                    self._system(text)
            except Exception as exc:                      # noqa: BLE001
                if not self._warned:
                    print(f"  !!! voice backend failed ({exc.__class__.__name__}: "
                          f"{exc}); falling back to the system voice")
                    self._warned = True
                self.backend = "system"
                try:
                    self._system(text)
                except Exception:                         # noqa: BLE001
                    self.backend = "off"

    def _gemini(self, text):
        """Native Gemini audio: one request, waveform straight back.

        The call lives in voice.py, shared with the API server, so the model
        id, voice and PCM format cannot drift between the two. It also caches
        the client -- this used to construct one per spoken line, paying auth
        and HTTP setup on the latency-sensitive path.
        """
        self._play_wav(voice.synthesize(text))

    @staticmethod
    def _play_wav(wav_bytes):
        """Play a complete WAV. winsound is stdlib and needs a real path."""
        import winsound

        path = Path(os.environ.get("TMP", ".")) / "have_hit_coach.wav"
        path.write_bytes(wav_bytes)
        winsound.PlaySound(str(path), winsound.SND_FILENAME)

    @staticmethod
    def _system(text):
        """Windows SAPI via PowerShell -- no packages, no API key."""
        safe = text.replace("'", "''")
        subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Add-Type -AssemblyName System.Speech; "
             "(New-Object System.Speech.Synthesis.SpeechSynthesizer)"
             f".Speak('{safe}')"],
            capture_output=True, timeout=30,
        )


# --------------------------------------------------------------------------
# Rolling clip buffer
# --------------------------------------------------------------------------

class RollingClip:
    """Keeps the last `seconds` of frames, subsampled to NUM_FRAMES on demand.

    Not simply the last 16 frames: at 30 fps that is half a second, far too
    short to contain a rep. Training clips were 16 frames spread across a
    multi-second span, so the live window is sampled the same way to keep the
    input distribution close to what the head was trained on.
    """

    def __init__(self, seconds, num_frames=config.NUM_FRAMES):
        self.seconds = seconds
        self.num_frames = num_frames
        self.buf = deque(maxlen=600)
        self.lock = threading.Lock()

    def push(self, frame_rgb):
        now = time.time()
        with self.lock:
            self.buf.append((now, frame_rgb))
            while self.buf and now - self.buf[0][0] > self.seconds:
                self.buf.popleft()

    def clip(self):
        """(NUM_FRAMES, H, W, 3) uint8 RGB, or None if not enough frames yet."""
        with self.lock:
            frames = [f for _, f in self.buf]
        if len(frames) < self.num_frames:
            return None
        idx = np.linspace(0, len(frames) - 1, self.num_frames).round().astype(int)
        return np.stack([frames[i] for i in idx])


# --------------------------------------------------------------------------
# Input conditioning
# --------------------------------------------------------------------------

# Mean luma below this counts as a dark room worth correcting. 0-255 scale.
DARK_LUMA = 90.0


def mean_luma(frames):
    """Rec.601 luma of a clip, sampled cheaply."""
    f = frames[::4].astype(np.float32)
    return float((0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]).mean())


def enhance_clip(frames, mode, clahe):
    """Lift contrast in a dark clip. Returns (frames, applied).

    CLAHE on the L channel of LAB only -- equalising RGB channels separately
    shifts colour, and the encoder's ImageNet normalisation is a fixed affine
    transform that cannot compensate for an underexposed room.

    Default is "auto" rather than "always" on purpose: the training clips never
    went through CLAHE, so applying it to an already well-lit frame pushes the
    input away from the distribution the head was fitted on for no gain. In a
    dark room the frame is off-distribution anyway, so correcting it is the
    lesser of two mismatches.
    """
    if mode == "off":
        return frames, False
    if mode == "auto" and mean_luma(frames) >= DARK_LUMA:
        return frames, False

    out = []
    for f in frames:
        lab = cv2.cvtColor(f, cv2.COLOR_RGB2LAB)
        lab[..., 0] = clahe.apply(lab[..., 0])
        out.append(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
    return np.stack(out), True


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------

def check_label_alignment():
    """Refuse to run if the checkpoint's label order is not config's."""
    if not CLASSES_SIDECAR.exists():
        print(f"  !!! {CLASSES_SIDECAR} missing -- cannot confirm label order.")
        print("      Retrain to regenerate it. Continuing on config order.")
        return
    saved = json.loads(CLASSES_SIDECAR.read_text(encoding="utf-8"))["classes"]
    if saved != list(config.EXERCISES):
        print("  !!! checkpoint label order does not match config.EXERCISES:")
        print(f"      checkpoint: {', '.join(saved)}")
        print(f"      config    : {', '.join(config.EXERCISES)}")
        print("      Every prediction would be mislabelled. Retrain first.")
        sys.exit(1)
    print(f"  label order   : verified against {CLASSES_SIDECAR}")


class Predictor:
    """Frozen encoder + trained head, running on its own thread."""

    def __init__(self, smooth=3, tta=True, enhance="auto"):
        from models.world_model import VJEPAEncoder

        print("  loading encoder...")
        self.enc = VJEPAEncoder()
        for p in self.enc.model.parameters():
            p.requires_grad = False

        print("  loading head...")
        self.head = load_classifier()

        self.tta = tta
        self.enhance = enhance
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self.history = deque(maxlen=smooth)
        self.result = (None, 0.0, 0.0)     # label, confidence, latency ms
        self.enhanced = False
        self.lock = threading.Lock()
        self.stop = threading.Event()

    def _probs(self, x):
        """x: preprocessed (1, 3, T, H, W) tensor -> class probabilities."""
        emb = self.enc.embed_preprocessed(x)
        with torch.no_grad():
            logits = self.head(torch.tensor(emb, device=config.DEVICE).unsqueeze(0))
            return torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    def predict(self, frames):
        """frames -> (label, confidence, latency ms).

        Probabilities are averaged over the mirrored pair (if TTA is on) and
        then over the last few clips, so a momentary bad frame cannot flip the
        announced label on its own.
        """
        t0 = time.time()
        frames, applied = enhance_clip(frames, self.enhance, self.clahe)
        self.enhanced = applied

        # Preprocess once, then mirror the tensor on the GPU. Mirroring the
        # uint8 frames instead would allocate a second full-size clip on the
        # host -- that 14 MiB copy is what made this fall over on a machine
        # already tight on memory -- and would repeat 16 cv2.resize calls.
        # Flip commutes with resize and centre crop, so the two agree to
        # within the one-pixel crop offset on odd-width frames.
        x = self.enc.preprocess(frames)
        probs = self._probs(x)
        if self.tta:
            # Deterministic: identical input always gives identical output.
            probs = (probs + self._probs(torch.flip(x, dims=[-1]))) / 2.0

        self.history.append(probs)
        mean = np.mean(self.history, axis=0)
        idx = int(mean.argmax())
        return config.ID_TO_LABEL[idx], float(mean[idx]), (time.time() - t0) * 1000

    def run(self, rolling, interval):
        while not self.stop.is_set():
            frames = rolling.clip()
            if frames is None:
                time.sleep(0.05)
                continue
            try:
                out = self.predict(frames)
            except Exception as exc:                      # noqa: BLE001
                print(f"  !!! inference error: {exc}")
                time.sleep(0.5)
                continue
            with self.lock:
                self.result = out
            time.sleep(interval)

    def latest(self):
        with self.lock:
            return self.result


# --------------------------------------------------------------------------

def spoken_line(label, conf):
    pretty = label.replace("_", " ")
    return f"You are doing {pretty}. Confidence {int(conf * 100)} percent."


def draw(frame_bgr, label, conf, threshold, latency, fps, enhanced=False):
    h, w = frame_bgr.shape[:2]
    cv2.rectangle(frame_bgr, (0, 0), (w, 96), (0, 0, 0), -1)

    if label is None:
        cv2.putText(frame_bgr, "warming up...", (16, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
    else:
        hit = conf >= threshold
        colour = (80, 230, 80) if hit else (60, 160, 230)
        text = label.replace("_", " ").upper()
        cv2.putText(frame_bgr, text, (16, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, colour, 3)
        cv2.putText(frame_bgr, f"{conf * 100:5.1f}%", (16, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
        bar_x = 200
        cv2.rectangle(frame_bgr, (bar_x, 62), (bar_x + 320, 84), (70, 70, 70), 1)
        cv2.rectangle(frame_bgr, (bar_x, 62),
                      (bar_x + int(320 * conf), 84), colour, -1)
        gate = bar_x + int(320 * threshold)
        cv2.line(frame_bgr, (gate, 58), (gate, 88), (255, 255, 255), 2)

    cv2.putText(frame_bgr, f"{latency:.0f} ms | {fps:.1f} fps | q to quit",
                (w - 330, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 170), 1)
    if enhanced:
        cv2.putText(frame_bgr, "low light: contrast lifted", (w - 330, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 200, 255), 1)
    return frame_bgr


def selftest(args):
    """Prove the whole chain works without needing a camera or a person."""
    print("=" * 62)
    print("SELF TEST (no camera)")
    print("=" * 62)
    check_label_alignment()

    pred = Predictor(smooth=1, tta=args.tta, enhance=args.enhance)
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 255, (config.NUM_FRAMES, 480, 640, 3), dtype=np.uint8)

    label, conf, ms = pred.predict(frames)          # first call warms CUDA up
    label, conf, ms = pred.predict(frames)
    print(f"\n  synthetic clip -> {label}  ({conf * 100:.1f}%, {ms:.0f} ms)")
    print(f"  clip shape    : {frames.shape}")
    print(f"  crop          : {config.CROP_SIZE}x{config.CROP_SIZE}")
    print(f"  device        : {config.DEVICE}")
    print(f"  mirror TTA    : {'on (2 views/clip)' if args.tta else 'off'}")
    print(f"  contrast lift : {args.enhance}")

    # Determinism matters: a random flip would make these two disagree.
    a = pred.predict(frames)[1]
    b = pred.predict(frames)[1]
    print(f"  repeatability : {a * 100:.2f}% then {b * 100:.2f}% "
          f"-> {'identical' if abs(a - b) < 1e-6 else 'DIFFERS'}")

    # Dark-room path: does the contrast lift actually engage?
    dark = (frames * 0.25).astype(np.uint8)
    _, applied = enhance_clip(dark, args.enhance, pred.clahe)
    print(f"  dark clip     : luma {mean_luma(dark):.0f} -> "
          f"contrast {'lifted' if applied else 'not lifted'}")

    v = Voice(args.voice)
    print(f"  voice backend : {v.backend}")
    if args.voice != "off":
        v.say("Have hit coach is online.")
        time.sleep(6)

    print("\n  Pipeline is functional. Noise gives a meaningless label by")
    print("  design -- this checks plumbing, not accuracy.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.80,
                    help="confidence needed to announce (default 0.80)")
    # 10 s rolling window. Training clips spanned duration/3 of each video,
    # measured across all 678 clips as median 15.5 s (p10 5.3, p90 32.9) -- so
    # there is no single "correct" value, only a distribution. 10 s sits inside
    # it and keeps the coach responsive; 15 s matches the median more closely
    # but delays the first prediction and reacts slowly to a change of exercise.
    ap.add_argument("--seconds", type=float, default=10.0,
                    help="length of the rolling window fed to the encoder "
                         "(training median was 15.5s)")
    ap.add_argument("--interval", type=float, default=0.4,
                    help="seconds between inferences")
    ap.add_argument("--repeat-after", type=float, default=20.0,
                    help="re-announce the same exercise after this many seconds")
    ap.add_argument("--smooth", type=int, default=3,
                    help="clips averaged together to steady the prediction")
    ap.add_argument("--tta", action=argparse.BooleanOptionalAction, default=True,
                    help="score each clip and its mirror, average the two "
                         "(--no-tta halves latency)")
    ap.add_argument("--enhance", choices=["auto", "always", "off"], default="auto",
                    help="CLAHE contrast lift; auto applies it only when the "
                         "room is actually dark")
    ap.add_argument("--voice", choices=["auto", "gemini", "system", "off"],
                    default="auto")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    print("=" * 62)
    print("HAVE HIT -- LIVE COACH")
    print("=" * 62)

    if args.selftest:
        return selftest(args)

    check_label_alignment()
    pred = Predictor(smooth=args.smooth, tta=args.tta, enhance=args.enhance)
    voice = Voice(args.voice)
    print(f"  voice backend : {voice.backend}")
    print(f"  mirror TTA    : {'on (2 views/clip)' if args.tta else 'off'}")
    print(f"  contrast lift : {args.enhance}"
          + (f" (triggers below luma {DARK_LUMA:.0f})" if args.enhance == "auto" else ""))

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"\n  !!! could not open camera {args.camera}.")
        print("      Check no other app holds it, or try --camera 1.")
        return 1

    rolling = RollingClip(args.seconds)
    worker = threading.Thread(target=pred.run, args=(rolling, args.interval),
                              daemon=True)
    worker.start()

    print(f"\n  window {args.seconds:.0f}s -> {config.NUM_FRAMES} frames, "
          f"announce above {args.threshold * 100:.0f}%")
    print("  press q or Esc in the preview window to quit\n")

    last_spoken, last_spoken_at = None, 0.0
    frames_seen, t_fps, fps = 0, time.time(), 0.0

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("  !!! camera read failed")
                break

            frame_bgr = cv2.flip(frame_bgr, 1)            # mirror, feels natural
            rolling.push(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

            frames_seen += 1
            if frames_seen % 15 == 0:
                now = time.time()
                fps = 15 / (now - t_fps)
                t_fps = now

            label, conf, ms = pred.latest()

            if label is not None and conf >= args.threshold:
                now = time.time()
                if label != last_spoken or now - last_spoken_at >= args.repeat_after:
                    line = spoken_line(label, conf)
                    print(f"  [{time.strftime('%H:%M:%S')}]  {line}")
                    voice.say(line)
                    last_spoken, last_spoken_at = label, now

            cv2.imshow("Have Hit - live coach",
                       draw(frame_bgr, label, conf, args.threshold, ms, fps,
                            pred.enhanced))
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        pred.stop.set()
        cap.release()
        cv2.destroyAllWindows()

    print("\n  stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
