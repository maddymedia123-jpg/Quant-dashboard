"""The one rule shared by the rail and the phone strip: both controls name the same open screen."""
from __future__ import annotations


def pick_from_strip(open_screen: str, picked: str | None) -> tuple[str, str]:
    """(screen, strip) after the phone strip changes. Tapping the open pill clears it; that must leave
    the page on the screen it was showing, with that pill still marked - never on no screen at all."""
    screen = picked or open_screen
    return screen, screen
