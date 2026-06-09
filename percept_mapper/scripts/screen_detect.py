"""Automatic screen-geometry detection for PhosLab.

The experiment config (`config/params.yaml`, `screen:` block) carries
physical display parameters — resolution and `screen_diagonal_inches` —
that are *per machine*. They drift silently when a config is copied
between PCs (e.g. a 13.3" laptop value reused on a 27" desktop), which
throws off `validate_eye_tracker` and any physical-size reasoning.

This module reads the real display geometry from the OS and can write it
back into `params.yaml`. It is a *calibrate-once-per-PC* utility, sibling
to `validate_eye_tracker.py` / `validate_protocol.py`.

What is detected:
- resolution (px)           — reliable on every platform
- physical size (cm)        — Windows: EDID via WMI; elsewhere best-effort
- diagonal (inches)         — derived from physical size

The core px/deg mapping does NOT use the diagonal (it uses
`screen_width / (2 * vf_scope_deg)`), so detection is about keeping the
recorded physical geometry honest, not about changing stimulus positions.

CLI:
    python -m scripts.screen_detect            # print detected geometry + diff
    python -m scripts.screen_detect --write    # persist into params.yaml
    python -m scripts.screen_detect --params <path>
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DisplayInfo:
    index: int
    name: str
    width_cm: float | None
    height_cm: float | None
    resolution_px: tuple[int, int] | None
    is_primary: bool = False
    active: bool = True

    @property
    def diagonal_inches(self) -> float | None:
        if not self.width_cm or not self.height_cm:
            return None
        return math.hypot(self.width_cm, self.height_cm) / 2.54

    def describe(self) -> str:
        res = (
            f"{self.resolution_px[0]}x{self.resolution_px[1]}"
            if self.resolution_px
            else "?x?"
        )
        if self.diagonal_inches:
            phys = (
                f"{self.diagonal_inches:.1f}\" "
                f"({self.width_cm:.1f}x{self.height_cm:.1f} cm)"
            )
        else:
            phys = "physical size unavailable"
        tag = " [primary]" if self.is_primary else ""
        name = self.name or "Unknown"
        return f"Monitor {self.index}{tag}: {name} — {res}, {phys}"


# ── Detection ────────────────────────────────────────────────────────────

_PS_SCRIPT = r"""
$OutputEncoding = [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false
$ErrorActionPreference = 'Stop'
$params = @(Get-CimInstance -Namespace root\wmi -ClassName WmiMonitorBasicDisplayParams)
$ids    = @(Get-CimInstance -Namespace root\wmi -ClassName WmiMonitorID)
Add-Type -AssemblyName System.Windows.Forms
$screens = @([System.Windows.Forms.Screen]::AllScreens)
$out = @()
for ($i = 0; $i -lt $params.Count; $i++) {
    $p = $params[$i]
    $name = ''
    if ($i -lt $ids.Count -and $ids[$i].UserFriendlyName) {
        $name = -join ($ids[$i].UserFriendlyName | Where-Object { $_ -ne 0 } | ForEach-Object { [char]$_ })
    }
    $res = $null
    if ($i -lt $screens.Count) {
        $res = @{ w = $screens[$i].Bounds.Width; h = $screens[$i].Bounds.Height; primary = [bool]$screens[$i].Primary }
    }
    $out += [PSCustomObject]@{
        index     = $i
        name      = $name
        width_cm  = [double]$p.MaxHorizontalImageSize
        height_cm = [double]$p.MaxVerticalImageSize
        active    = [bool]$p.Active
        res       = $res
    }
}
$out | ConvertTo-Json -Depth 5 -Compress
"""


def _detect_windows() -> list[DisplayInfo]:
    proc = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            _PS_SCRIPT,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
        # Avoid flashing a console window when called mid-experiment (fullscreen)
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    raw = (proc.stdout or "").strip()
    if not raw:
        raise RuntimeError(
            f"no output from WMI query (stderr: {(proc.stderr or '').strip()[:200]})"
        )
    data = json.loads(raw)
    if isinstance(data, dict):  # single monitor -> ConvertTo-Json emits an object
        data = [data]

    displays: list[DisplayInfo] = []
    for entry in data:
        res = entry.get("res")
        resolution = None
        is_primary = False
        if res:
            resolution = (int(res["w"]), int(res["h"]))
            is_primary = bool(res.get("primary", False))
        w_cm = float(entry.get("width_cm") or 0) or None
        h_cm = float(entry.get("height_cm") or 0) or None
        displays.append(
            DisplayInfo(
                index=int(entry.get("index", len(displays))),
                name=str(entry.get("name") or "").strip(),
                width_cm=w_cm,
                height_cm=h_cm,
                resolution_px=resolution,
                is_primary=is_primary,
                active=bool(entry.get("active", True)),
            )
        )
    return displays


def _detect_fallback() -> list[DisplayInfo]:
    """Resolution-only fallback for non-Windows / when EDID is unavailable.

    Physical size is left as None so callers keep the manually configured
    value instead of guessing.
    """
    resolution = None
    try:
        import tkinter

        root = tkinter.Tk()
        resolution = (root.winfo_screenwidth(), root.winfo_screenheight())
        root.destroy()
    except Exception:
        resolution = None
    if resolution is None:
        return []
    return [
        DisplayInfo(
            index=0,
            name="",
            width_cm=None,
            height_cm=None,
            resolution_px=resolution,
            is_primary=True,
        )
    ]


def detect_displays() -> list[DisplayInfo]:
    """Return detected displays, primary first. Empty list if nothing found."""
    displays: list[DisplayInfo] = []
    if platform.system() == "Windows":
        try:
            displays = _detect_windows()
        except Exception as e:  # noqa: BLE001 — detection is best-effort
            print(f"[screen_detect] ⚠ WMI/EDID detection failed: {e}")
            displays = _detect_fallback()
    else:
        displays = _detect_fallback()

    displays.sort(key=lambda d: (not d.is_primary, d.index))
    return displays


def primary_display(displays: list[DisplayInfo]) -> DisplayInfo | None:
    for d in displays:
        if d.is_primary:
            return d
    return displays[0] if displays else None


# ── params.yaml writer (block-aware, comment-preserving) ───────────────────

def update_params_screen_block(
    text: str,
    *,
    width: int | None = None,
    height: int | None = None,
    diagonal_inches: float | None = None,
) -> str:
    """Return `text` with the given keys updated inside the top-level
    `screen:` block. Only *direct* children of `screen:` are touched
    (so nested blocks like `anchor_circle:` are left alone). Missing keys
    are inserted at the child indentation. Other lines, comments and key
    order are preserved.
    """
    updates: dict[str, str] = {}
    if width is not None:
        updates["width"] = str(int(width))
    if height is not None:
        updates["height"] = str(int(height))
    if diagonal_inches is not None:
        updates["screen_diagonal_inches"] = f"{round(float(diagonal_inches), 2)}"
    if not updates:
        return text

    lines = text.splitlines(keepends=True)
    out: list[str] = []
    in_screen = False
    screen_indent = 0
    child_indent: int | None = None
    seen: set[str] = set()

    def _flush_missing() -> None:
        ci = child_indent if child_indent is not None else screen_indent + 2
        for k, v in updates.items():
            if k not in seen:
                out.append(" " * ci + f"{k}: {v}\n")
                seen.add(k)

    for line in lines:
        body = line.rstrip("\r\n")
        if not in_screen:
            m = re.match(r"^(\s*)screen:\s*$", body)
            if m:
                in_screen = True
                screen_indent = len(m.group(1))
            out.append(line)
            continue

        # inside the screen block
        if body.strip() == "":
            out.append(line)
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent <= screen_indent:  # block ended
            _flush_missing()
            in_screen = False
            out.append(line)
            continue
        if child_indent is None:
            child_indent = indent
        mk = re.match(r"^(\s*)([A-Za-z0-9_]+):(.*)$", body)
        if mk and indent == child_indent and mk.group(2) in updates:
            key = mk.group(2)
            out.append(" " * child_indent + f"{key}: {updates[key]}\n")
            seen.add(key)
            continue
        out.append(line)

    if in_screen:  # file ended while still inside the block
        _flush_missing()

    return "".join(out)


def default_params_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "params.yaml"


def write_to_params(params_path: Path, display: DisplayInfo) -> bool:
    """Persist `display`'s resolution + diagonal into the params file.
    Returns True if the file content changed."""
    text = params_path.read_text(encoding="utf-8")
    res = display.resolution_px
    new_text = update_params_screen_block(
        text,
        width=res[0] if res else None,
        height=res[1] if res else None,
        diagonal_inches=display.diagonal_inches,
    )
    if new_text == text:
        return False
    params_path.write_text(new_text, encoding="utf-8")
    return True


# ── CLI ────────────────────────────────────────────────────────────────────

def _read_configured_screen(params_path: Path) -> dict:
    try:
        import yaml

        with open(params_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("screen", {}) or {}
    except Exception:
        return {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Detect screen geometry from the OS.")
    parser.add_argument(
        "--write",
        action="store_true",
        help="write detected resolution + diagonal into params.yaml",
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=default_params_path(),
        help="path to params.yaml (default: config/params.yaml)",
    )
    args = parser.parse_args(argv)

    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    displays = detect_displays()
    if not displays:
        print("[screen_detect] ✗ no displays detected")
        return 1

    print("Detected displays:")
    for d in displays:
        print(f"  {d.describe()}")

    target = primary_display(displays)
    if target is None:
        return 1

    # Compare to current config
    screen_cfg = _read_configured_screen(args.params)
    cfg_diag = screen_cfg.get("screen_diagonal_inches")
    cfg_w, cfg_h = screen_cfg.get("width"), screen_cfg.get("height")
    print("\nConfigured vs detected (primary):")
    print(f"  resolution: config {cfg_w}x{cfg_h}  →  detected "
          f"{target.resolution_px[0] if target.resolution_px else '?'}"
          f"x{target.resolution_px[1] if target.resolution_px else '?'}")
    det_diag = target.diagonal_inches
    print(f"  diagonal:   config {cfg_diag}\"  →  detected "
          f"{f'{det_diag:.2f}' if det_diag else 'n/a'}\"")
    if det_diag and cfg_diag and abs(float(cfg_diag) - det_diag) / det_diag > 0.10:
        print("  ⚠ configured diagonal is off by >10% from the real display")

    if args.write:
        if target.diagonal_inches is None and target.resolution_px is None:
            print("\n[screen_detect] nothing to write (no detected geometry)")
            return 1
        changed = write_to_params(args.params, target)
        if changed:
            print(f"\n✓ Updated {args.params} (screen: block)")
        else:
            print(f"\n= {args.params} already matches detected geometry")
    else:
        print("\n(run with --write to persist into params.yaml)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
