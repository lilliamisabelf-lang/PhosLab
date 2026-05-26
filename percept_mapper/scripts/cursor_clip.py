"""Windows cursor confinement helper.

Constrains the system cursor to a screen rectangle using the user32.ClipCursor
API. Used while the drawing tablet screen is active so that a multi-monitor
setup doesn't let the tablet/mouse drift onto a secondary display.

No-op on non-Windows platforms.
"""

import ctypes
import sys
from ctypes import wintypes

_IS_WIN = sys.platform == "win32"

if _IS_WIN:
    _user32 = ctypes.windll.user32

    _MonitorEnumProc = ctypes.WINFUNCTYPE(
        ctypes.c_int,
        wintypes.HMONITOR,
        wintypes.HDC,
        ctypes.POINTER(wintypes.RECT),
        wintypes.LPARAM,
    )

    _SM_CXSCREEN = 0
    _SM_CYSCREEN = 1


def primary_monitor_rect():
    """(left, top, right, bottom) of the primary monitor. Primary is anchored
    at (0, 0) by Windows convention."""
    if not _IS_WIN:
        return None
    w = _user32.GetSystemMetrics(_SM_CXSCREEN)
    h = _user32.GetSystemMetrics(_SM_CYSCREEN)
    return (0, 0, int(w), int(h))


def enumerate_monitor_rects():
    """List of (left, top, right, bottom) for every attached monitor, in the
    order Windows enumerates them. Index 0 is typically the primary."""
    if not _IS_WIN:
        return []
    rects = []

    def cb(_hmon, _hdc, lprc, _lp):
        r = lprc.contents
        rects.append((int(r.left), int(r.top), int(r.right), int(r.bottom)))
        return 1

    callback = _MonitorEnumProc(cb)
    _user32.EnumDisplayMonitors(None, None, callback, 0)
    return rects


def resolve_target_rect(monitor):
    """Pick a clip rect based on a config value.

    Accepts: 'primary' | 'none' | int index | None.
    Returns: (l, t, r, b) tuple, or None to disable clipping.
    """
    if monitor is None or (isinstance(monitor, str) and monitor.lower() in ("none", "off", "")):
        return None
    if isinstance(monitor, str) and monitor.lower() == "primary":
        return primary_monitor_rect()
    try:
        idx = int(monitor)
    except (TypeError, ValueError):
        return primary_monitor_rect()
    rects = enumerate_monitor_rects()
    if 0 <= idx < len(rects):
        return rects[idx]
    print(f"[cursor_clip] ⚠ monitor index {idx} fuera de rango; usando primario")
    return primary_monitor_rect()


def clip_cursor(rect):
    """Confine the cursor to `rect = (l, t, r, b)`, or release if rect is None."""
    if not _IS_WIN:
        return False
    if rect is None:
        _user32.ClipCursor(None)
        return False
    l, t, r, b = rect
    w_rect = wintypes.RECT(int(l), int(t), int(r), int(b))
    ok = bool(_user32.ClipCursor(ctypes.byref(w_rect)))
    return ok
