"""Non-blocking feedback sounds for classification decisions."""

import threading


def play_decision_buzzer() -> None:
    """Play a short two-tone buzzer without blocking the Qt interface."""
    def play() -> None:
        try:
            import winsound

            winsound.Beep(880, 110)
            winsound.Beep(660, 170)
        except (ImportError, RuntimeError):
            pass

    threading.Thread(target=play, daemon=True).start()