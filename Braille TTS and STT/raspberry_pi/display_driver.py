"""
display_driver.py  -  BrailleAI Module 2: Refreshable Braille Display Driver
===========================================================================
Hardware-abstraction layer for the row of servo-actuated 6-dot cells.

Each cell = 2x MG90S servos driven through a PCA9685 16-channel PWM
controller over I2C (Section II/III). Pipeline:
    cells (dot sets) -> per-servo angles -> PWM duty cycles

Handles physical realities: limited cell count (paging), servo settle
time, and blanking on idle. A pluggable PWMBackend lets the same logic
run on a simulator now and a real PCA9685 on-device.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import List, Protocol, Tuple

from braille_core import (
    text_to_cells, cell_to_servo_angles, dots_to_halfcell_states,
)

SERVO_FREQ_HZ = 50
PWM_MIN_US = 500
PWM_MAX_US = 2500


def angle_to_duty_12bit(angle: float) -> int:
    """Convert a 0..180 deg angle to a 12-bit PCA9685 duty (0..4095)."""
    angle = max(0.0, min(180.0, angle))
    pulse_us = PWM_MIN_US + (PWM_MAX_US - PWM_MIN_US) * angle / 180.0
    period_us = 1000000 / SERVO_FREQ_HZ
    return int(round(pulse_us / period_us * 4096))


class PWMBackend(Protocol):
    def set_channel(self, channel: int, duty_12bit: int) -> None: ...


class SimulatedPCA9685:
    """In-memory stand-in for the PCA9685; records last duty per channel."""
    def __init__(self, channels: int = 16):
        self.duty: List[int] = [0] * channels

    def set_channel(self, channel: int, duty_12bit: int) -> None:
        self.duty[channel] = duty_12bit


@dataclass
class BrailleDisplay:
    num_cells: int = 4
    backend: PWMBackend = field(default_factory=SimulatedPCA9685)
    settle_per_step_s: float = 0.06
    _last_states: List[Tuple[int, int]] = field(default_factory=list)

    def __post_init__(self):
        self._last_states = [(0, 0)] * self.num_cells

    def _write_cell(self, index: int, dots) -> float:
        left, right = dots_to_halfcell_states(dots)
        l_ang, r_ang = cell_to_servo_angles(dots)
        self.backend.set_channel(2 * index, angle_to_duty_12bit(l_ang))
        self.backend.set_channel(2 * index + 1, angle_to_duty_12bit(r_ang))
        pl, pr = self._last_states[index]
        steps = max(abs(left - pl), abs(right - pr))
        self._last_states[index] = (left, right)
        return steps * self.settle_per_step_s

    def show_cells(self, cells, *, simulate_time: bool = False) -> float:
        worst = 0.0
        for i in range(self.num_cells):
            dots = cells[i] if i < len(cells) else frozenset()
            worst = max(worst, self._write_cell(i, dots))
        if simulate_time:
            time.sleep(worst)
        return worst

    def render_text(self, text: str, *, on_window=None,
                    simulate_time: bool = False) -> int:
        cells = text_to_cells(text)
        windows = [cells[i:i + self.num_cells]
                   for i in range(0, max(len(cells), 1), self.num_cells)]
        for w, win in enumerate(windows):
            settle = self.show_cells(win, simulate_time=simulate_time)
            if on_window:
                on_window(w, win)
            else:
                states = [dots_to_halfcell_states(c) for c in win]
                print(f"  window {w}: states={states} settle={settle:.2f}s")
        return len(windows)

    def blank(self) -> None:
        for i in range(self.num_cells):
            self._write_cell(i, frozenset())


if __name__ == "__main__":
    print("angle->duty:", angle_to_duty_12bit(0),
          angle_to_duty_12bit(90), angle_to_duty_12bit(180))
    disp = BrailleDisplay(num_cells=4)
    msg = "hello world"
    print(f"Rendering {msg!r} on a {disp.num_cells}-cell display:")
    n = disp.render_text(msg)
    print(f"Shown in {n} windows.")
    disp.blank()
    print("Blanked. Channel duties:", disp.backend.duty[:8])