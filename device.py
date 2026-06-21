"""
device.py  -  BrailleAI Module 6: Main Device Controller / State Machine
=======================================================================
Integrates Modules 1-5 into the complete device described in the outline.
This is the PC-runnable analogue of the ESP32-S3 firmware main loop.

Modes (mode button cycles through them):
  1. FACE_TO_FACE - hearing person speaks -> emotion + summary on Braille;
                    DeafBlind user types (Perkins) -> spoken via TTS.
  2. TUTOR        - AI Braille tutor drills the user.
  3. ASSISTANT    - AI personal assistant answers typed questions.

Every interaction is logged through CloudLogger for remote monitoring.
"""
from __future__ import annotations
import os
import time
from enum import Enum
from typing import List

from braille_core import emotion_to_prefix
from display_driver import BrailleDisplay
from input_handler import ChordInput
from ai_backend import (AIBackend, MockSTT, MockLLM, MockTTS,
                        PiEmotionReceiver, MockEmotionClassifier)
from cloud_sync import CloudLogger, InteractionEvent, fake_poster_factory


class Mode(Enum):
    FACE_TO_FACE = 1
    TUTOR = 2
    ASSISTANT = 3


class BrailleAIDevice:
    def __init__(self, display: BrailleDisplay, ai: AIBackend,
                 logger: CloudLogger, num_cells: int = 4):
        self.display = display
        self.ai = ai
        self.logger = logger
        self.mode = Mode.FACE_TO_FACE
        self.kbd = ChordInput()

    # -- controls --------------------------------------------------------
    def cycle_mode(self) -> Mode:
        nxt = self.mode.value % len(Mode) + 1
        self.mode = Mode(nxt)
        self.display.blank()
        return self.mode

    def _render(self, text: str, emotion: str = "neutral") -> None:
        """Show emotion prefix cell (if any) then the text on the display."""
        prefix = emotion_to_prefix(emotion)
        if prefix:
            self.display.show_cells([prefix])
        self.display.render_text(text)

    # -- Mode 1 ----------------------------------------------------------
    def on_speech(self, audio: bytes) -> None:
        emotion, summary = self.ai.summarize_for_braille(audio)
        self._render(summary, emotion)
        self.logger.log(InteractionEvent(time.time(), "face_to_face", "in",
                                         summary, emotion))
        print(f"[FACE2FACE] ({emotion}) display -> {summary!r}")

    def on_user_message(self, text: str) -> bytes:
        """DeafBlind user finished typing a message; speak it aloud."""
        audio = self.ai.speak(text)
        self.logger.log(InteractionEvent(time.time(), "face_to_face", "out", text))
        print(f"[FACE2FACE] speak -> {text!r}")
        return audio

    # -- Mode 2 ----------------------------------------------------------
    def tutor(self, user_text: str) -> None:
        reply = self.ai.tutor_step(user_text)
        self._render(reply)
        self.logger.log(InteractionEvent(time.time(), "tutor", "out", reply))
        print(f"[TUTOR] display -> {reply!r}")

    # -- Mode 3 ----------------------------------------------------------
    def assistant(self, user_text: str) -> None:
        answer = self.ai.assistant_query(user_text)
        self._render(answer)
        self.logger.log(InteractionEvent(time.time(), "assistant", "out", answer))
        print(f"[ASSIST] display -> {answer!r}")


def build_default_device(log_path: str,
                         use_pi_emotion: bool = False,
                         pi_serial_port: str = "/dev/ttyUSB0") -> BrailleAIDevice:
    """Build device with mock providers (testing) or Pi emotion (hardware).

    Args:
        log_path: Path for the interaction log file.
        use_pi_emotion: If True, use PiEmotionReceiver for real-time emotion
                        from the Raspberry Pi's camera+mic model.
        pi_serial_port: Serial port for Pi communication.
    """
    if os.path.exists(log_path):
        os.remove(log_path)
    display = BrailleDisplay(num_cells=4)

    if use_pi_emotion:
        emotion_clf = PiEmotionReceiver(port=pi_serial_port)
        emotion_clf.start()
    else:
        emotion_clf = MockEmotionClassifier()

    ai = AIBackend(stt=MockSTT(), llm=MockLLM(), tts=MockTTS(),
                   emotion_classifier=emotion_clf)
    logger = CloudLogger(sd_path=log_path,
                         webhook_url="https://script.google.com/macros/s/MOCK/exec",
                         poster=fake_poster_factory([]), batch_size=20)
    return BrailleAIDevice(display, ai, logger)


if __name__ == "__main__":
    log_path = os.path.join(os.path.dirname(__file__), "_device_log.jsonl")
    dev = build_default_device(log_path)

    print("=== Mode 1: Face-to-Face ===")
    dev.on_speech(b"I am so happy to meet you! Thanks for coming.")
    dev.on_user_message("nice to meet you too")

    print("\n=== Mode 2: Tutor ===")
    dev.cycle_mode()
    dev.tutor("apple")

    print("\n=== Mode 3: Assistant ===")
    dev.cycle_mode()
    dev.assistant("what is the weather?")

    print("\nPending log events to sync:", dev.logger.pending_count())
    print("Synced now:", dev.logger.sync())