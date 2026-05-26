"""Synthesized audio cue for saccade-mode "go" beep.

Generates a short sine-tone Sound object via numpy + pygame.sndarray. No
external WAV asset needed. Initializes pygame.mixer on first use.
"""

import numpy as np
import pygame


_MIXER_READY = False


def _ensure_mixer():
    global _MIXER_READY
    if _MIXER_READY:
        return True
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        _MIXER_READY = True
        return True
    except Exception as e:
        print(f"[audio_cue] ⚠ pygame.mixer.init falló: {e}")
        return False


def make_tone(frequency_hz: float = 880.0, duration_ms: int = 80,
              volume: float = 0.4) -> pygame.mixer.Sound | None:
    """Return a pygame.mixer.Sound that plays a short sine tone.

    Applies a ~5 ms linear envelope at start and end to avoid clicks.
    """
    if not _ensure_mixer():
        return None

    sample_rate = 44100
    n = int(sample_rate * float(duration_ms) / 1000.0)
    if n <= 0:
        return None

    t = np.arange(n, dtype=np.float64) / sample_rate
    wave = np.sin(2.0 * np.pi * float(frequency_hz) * t)

    # Envelope: 5 ms attack + 5 ms release to prevent click artifacts
    env_n = max(1, int(sample_rate * 0.005))
    env = np.ones(n, dtype=np.float64)
    env[:env_n] = np.linspace(0.0, 1.0, env_n)
    env[-env_n:] = np.linspace(1.0, 0.0, env_n)
    wave = wave * env * float(volume)

    # 16-bit stereo
    samples = (wave * 32767.0).astype(np.int16)
    stereo = np.column_stack((samples, samples)).copy()

    return pygame.sndarray.make_sound(stereo)


def from_config(cfg: dict | None) -> pygame.mixer.Sound | None:
    """Build a Sound from a `saccade.audio_cue` config block. Returns None if
    disabled or any step fails."""
    if not cfg or not cfg.get("enabled", False):
        return None
    try:
        return make_tone(
            frequency_hz=float(cfg.get("frequency_hz", 880.0)),
            duration_ms=int(cfg.get("duration_ms", 80)),
            volume=float(cfg.get("volume", 0.4)),
        )
    except Exception as e:
        print(f"[audio_cue] ⚠ make_tone falló: {e}")
        return None
