"""
ai_backend.py  -  BrailleAI Module 4: Cloud AI Orchestration
============================================================
Coordinates the cloud services described in Section IV:
    Speech-to-Text  ->  LLM (Claude)  ->  Text-to-Speech
plus tone/emotion classification used by Mode 1 (Face-to-Face).

Each external service sits behind a small Protocol so the real device can
plug in Whisper/Deepgram (STT), Anthropic Claude (LLM) and ElevenLabs/Polly
(TTS), while this file ships deterministic MOCK providers so the end-to-end
flow runs offline during development.

Emotion classification supports two modes:
  - Keyword-based (MockEmotionClassifier) for offline testing
  - Pi serial (PiEmotionReceiver) for real hardware — the Raspberry Pi runs
    the trained AVEmotionNet model and sends labels over UART

Public API maps 1:1 to the three device modes:
    summarize_for_braille(audio)  -> (emotion, short_text)   # Mode 1
    tutor_step(user_text)         -> tutor reply             # Mode 2
    assistant_query(user_text)    -> assistant answer         # Mode 3
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Tuple, List, Optional
import threading
import time


# --- Service interfaces ----------------------------------------------------
class STTProvider(Protocol):
    def transcribe(self, audio: bytes) -> str: ...


class LLMProvider(Protocol):
    def complete(self, system: str, user: str, max_words: int = 60) -> str: ...


class TTSProvider(Protocol):
    def synthesize(self, text: str) -> bytes: ...


# --- Mock providers (offline, deterministic) -------------------------------
class MockSTT:
    """Pretend audio bytes already carry UTF-8 text (for testing)."""
    def transcribe(self, audio: bytes) -> str:
        return audio.decode("utf-8", errors="ignore").strip()


class MockLLM:
    """Rule-based stand-in for Claude: summarizes / answers briefly."""
    def complete(self, system: str, user: str, max_words: int = 60) -> str:
        if "summarize" in system.lower():
            words = user.split()
            return " ".join(words[:max_words])
        if "tutor" in system.lower():
            return f"Good. Now try spelling: {user.strip().upper()[:1] or 'A'}"
        return f"You asked: {user.strip()}. (mock answer)"


class MockTTS:
    def synthesize(self, text: str) -> bytes:
        return text.encode("utf-8")


# --- Emotion classifier interface ------------------------------------------
class EmotionClassifier(Protocol):
    def get_emotion(self, text: str) -> str: ...


# --- Keyword-based classifier (offline/mock) --------------------------------
_EMOTION_KEYWORDS = {
    "happy":   ["great", "love", "happy", "wonderful", "thanks", "yay", "!"],
    "sad":     ["sad", "sorry", "miss", "unfortunately", "cry"],
    "angry":   ["angry", "no!", "stop", "hate", "furious"],
    "fearful": ["afraid", "scared", "help", "danger", "worried"],
    "question":["?", "what", "why", "how", "when", "where", "who"],
    "urgent":  ["urgent", "now", "emergency", "immediately", "hurry"],
}


class MockEmotionClassifier:
    """Lightweight keyword tone classifier for offline testing."""
    def get_emotion(self, text: str) -> str:
        low = text.lower()
        best, score = "neutral", 0
        for emo, kws in _EMOTION_KEYWORDS.items():
            s = sum(low.count(k) for k in kws)
            if s > score:
                best, score = emo, s
        return best


def classify_emotion(text: str) -> str:
    """Legacy function — uses keyword matching. Prefer MockEmotionClassifier."""
    return MockEmotionClassifier().get_emotion(text)


# --- Pi serial emotion receiver (real hardware) -----------------------------
class PiEmotionReceiver:
    """Receives emotion labels from the Raspberry Pi over UART serial.

    The Pi runs emotion_inference.py which sends:
        EMOTION:<label>:<confidence>\n
    e.g. "EMOTION:happy:0.62\n"

    This class runs a background thread that continuously reads from the
    serial port and caches the latest emotion. get_emotion() returns
    the most recent prediction (ignores the text arg — uses audio+video
    from the Pi's own camera/mic).
    """
    def __init__(self, port: str = "/dev/ttyUSB0", baud: int = 115200):
        self._port = port
        self._baud = baud
        self._latest_emotion: str = "neutral"
        self._latest_confidence: float = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._serial = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Open serial port and start background reader thread."""
        try:
            import serial
            self._serial = serial.Serial(self._port, self._baud, timeout=1)
            self._running = True
            self._thread = threading.Thread(
                target=self._read_loop, daemon=True
            )
            self._thread.start()
            print(f"[PiEmotion] Listening on {self._port} @ {self._baud}")
        except ImportError:
            print("[PiEmotion] pyserial not installed. "
                  "Run: pip install pyserial")
        except Exception as e:
            print(f"[PiEmotion] Cannot open {self._port}: {e}")

    def stop(self) -> None:
        """Stop the background reader and close the serial port."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._serial:
            self._serial.close()

    def _read_loop(self) -> None:
        """Background thread: read serial lines and parse emotion."""
        while self._running:
            try:
                if self._serial and self._serial.in_waiting:
                    line = self._serial.readline().decode("utf-8").strip()
                    self._parse_line(line)
                else:
                    time.sleep(0.05)
            except Exception:
                time.sleep(0.1)

    def _parse_line(self, line: str) -> None:
        """Parse 'EMOTION:<label>:<confidence>' format."""
        if not line.startswith("EMOTION:"):
            return
        parts = line.split(":")
        if len(parts) >= 3:
            label = parts[1]
            try:
                confidence = float(parts[2])
            except ValueError:
                confidence = 0.0
            with self._lock:
                self._latest_emotion = label
                self._latest_confidence = confidence

    def get_emotion(self, text: str = "") -> str:
        """Return the most recent emotion from the Pi.
        The text arg is accepted for interface compatibility but ignored
        (the Pi uses its own camera+mic for classification)."""
        with self._lock:
            return self._latest_emotion

    @property
    def confidence(self) -> float:
        with self._lock:
            return self._latest_confidence


# --- Orchestrator ----------------------------------------------------------
@dataclass
class AIBackend:
    stt: STTProvider
    llm: LLMProvider
    tts: TTSProvider
    emotion_classifier: EmotionClassifier = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.emotion_classifier is None:
            self.emotion_classifier = MockEmotionClassifier()

    # Mode 1: hearing person speaks -> short Braille text + emotion tag
    def summarize_for_braille(self, audio: bytes,
                              max_words: int = 12) -> Tuple[str, str]:
        transcript = self.stt.transcribe(audio)
        emotion = self.emotion_classifier.get_emotion(transcript)
        summary = self.llm.complete(
            system="You summarize speech for a Braille display. Be concise.",
            user=transcript, max_words=max_words)
        return emotion, summary

    # Mode 1 reverse: Braille user text -> spoken audio for the hearing person
    def speak(self, text: str) -> bytes:
        return self.tts.synthesize(text)

    # Mode 2: AI Braille tutor
    def tutor_step(self, user_text: str) -> str:
        return self.llm.complete(
            system="You are a patient Braille tutor.", user=user_text)

    # Mode 3: AI personal assistant
    def assistant_query(self, user_text: str) -> str:
        return self.llm.complete(
            system="You are a helpful assistant for a DeafBlind user.",
            user=user_text)


if __name__ == "__main__":
    # --- Demo with mock (keyword) classifier ---
    print("=== Mock (keyword) emotion classifier ===")
    ai = AIBackend(stt=MockSTT(), llm=MockLLM(), tts=MockTTS())

    spoken = "I am so happy to finally meet you! How are you today?"
    emo, summ = ai.summarize_for_braille(spoken.encode())
    print("Mode1  emotion:", emo)
    print("Mode1  summary:", summ)

    reply = ai.tutor_step("apple")
    print("Mode2  tutor  :", reply)

    ans = ai.assistant_query("What time is my meeting?")
    print("Mode3  answer :", ans)

    audio = ai.speak("hello there")
    print("Reverse TTS bytes:", audio)

    # --- Demo: how to use PiEmotionReceiver on real hardware ---
    print("\n=== Pi emotion receiver (hardware mode) ===")
    print("To use with real hardware:")
    print("  receiver = PiEmotionReceiver(port='/dev/ttyUSB0')")
    print("  receiver.start()")
    print("  ai = AIBackend(stt=RealSTT(), llm=RealLLM(), tts=RealTTS(),")
    print("                 emotion_classifier=receiver)")
    print("  # Now ai.summarize_for_braille() uses Pi's camera+mic model")
    print("  # receiver.stop()  # when done")
