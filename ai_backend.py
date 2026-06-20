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

Public API maps 1:1 to the three device modes:
    summarize_for_braille(audio)  -> (emotion, short_text)   # Mode 1
    tutor_step(user_text)         -> tutor reply             # Mode 2
    assistant_query(user_text)    -> assistant answer        # Mode 3
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Tuple, List


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


# --- Emotion classifier ----------------------------------------------------
_EMOTION_KEYWORDS = {
    "happy":   ["great", "love", "happy", "wonderful", "thanks", "yay", "!"],
    "sad":     ["sad", "sorry", "miss", "unfortunately", "cry"],
    "angry":   ["angry", "no!", "stop", "hate", "furious"],
    "fearful": ["afraid", "scared", "help", "danger", "worried"],
    "question":["?", "what", "why", "how", "when", "where", "who"],
    "urgent":  ["urgent", "now", "emergency", "immediately", "hurry"],
}


def classify_emotion(text: str) -> str:
    """Lightweight keyword tone classifier (real device uses the LLM)."""
    low = text.lower()
    best, score = "neutral", 0
    for emo, kws in _EMOTION_KEYWORDS.items():
        s = sum(low.count(k) for k in kws)
        if s > score:
            best, score = emo, s
    return best


# --- Orchestrator ----------------------------------------------------------
@dataclass
class AIBackend:
    stt: STTProvider
    llm: LLMProvider
    tts: TTSProvider

    # Mode 1: hearing person speaks -> short Braille text + emotion tag
    def summarize_for_braille(self, audio: bytes,
                              max_words: int = 12) -> Tuple[str, str]:
        transcript = self.stt.transcribe(audio)
        emotion = classify_emotion(transcript)
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