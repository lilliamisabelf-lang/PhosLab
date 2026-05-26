# Wacom Tablet — Setup & Usage Guide

End-to-end guide for installing the Wacom driver and using a Wacom tablet to draw percepts in PhosLab, alongside or instead of a mouse.

## Architecture

```
Wacom tablet (pen) ── USB ──▶ Wacom driver + WTabletServicePro ──▶ Windows HID/Pointer ──▶ pygame mouse events ──▶ DrawingTablet
                              (libwintab32.dll on disk —          (Wacom Pointer device     (MOUSEBUTTONDOWN /
                               not used directly by PhosLab,       reports x,y to the OS)    MOUSEBUTTONUP /
                               but installed by the driver)                                  pygame.mouse.get_pos)
```

PhosLab does **not** use Wintab or any Wacom-specific SDK. The tablet's pen is consumed via the same pygame mouse-event path used by an actual mouse. The `drawing_input` flag in `params.yaml` selects UI cues (brush size, title text, cursor visibility) for the active device — it does *not* filter the underlying event stream, because pygame cannot distinguish a mouse pointer movement from a pen pointer movement at the SDL level.

---

## Part 1 — One-time driver install

### 1.1 Download

Get the **Pen Tablet Driver** from <https://www.wacom.com/en-es/support/product-support/drivers>. A current build at the time of writing: `WacomTablet_6.4.13-4.exe`.

### 1.2 Install

1. Disconnect the tablet (recommended by the Wacom installer).
2. Run the installer as Administrator.
3. Reboot when prompted. *The Wacom service `WTabletServicePro` only starts cleanly after a full restart on Windows.*
4. Reconnect the tablet via USB.

### 1.3 Verify

Run these in PowerShell — they should all return positive:

```powershell
# Device enumeration (VID_056A = Wacom)
Get-PnpDevice -PresentOnly |
  Where-Object { $_.InstanceId -match 'VID_056A' } |
  Select-Object Class, FriendlyName, Manufacturer, Status |
  Format-Table -AutoSize -Wrap
```

Expected: at least one entry with `FriendlyName: Wacom Tablet` (Manufacturer `Wacom`) and one `Wacom Pointer` (Manufacturer `Wacom Technology`), all `Status: OK`. If you see only generic `Dispositivo de entrada USB` entries instead, the driver did not install correctly.

```powershell
# Driver service
Get-Service WTabletServicePro |
  Select-Object Name, Status, StartType |
  Format-Table -AutoSize
```

Expected: `Status: Running`, `StartType: Automatic`. If `Stopped`, reboot (the service does not always start on first install without a restart).

```powershell
# Wintab DLLs (used by some apps; PhosLab doesn't, but they confirm driver completeness)
Test-Path 'C:\Windows\System32\wintab32.dll', 'C:\Windows\SysWOW64\wintab32.dll'
```

Expected: `True, True`.

If the tablet shows `Present: False` / `Status: Unknown`, the cable is unplugged or the USB port is bad — reconnect.

---

## Part 2 — PhosLab configuration

### 2.1 Top-level flag

In [`percept_mapper/config/params.yaml`](config/params.yaml):

```yaml
drawing_input: tablet   # mouse | tablet | both
```

| Value    | Title shown on the drawing screen                                       | Default brush size |
|----------|--------------------------------------------------------------------------|--------------------|
| `mouse`  | "Dibuja con el ratón y presiona ENTER"                                  | 2 px               |
| `tablet` | "Dibuja con la tablet y presiona ENTER"                                 | 4 px               |
| `both`   | "Dibuja (ratón o tablet) y presiona ENTER"                              | 3 px               |

### 2.2 Per-mode overrides

Configurable in the `drawing_tablet:` block:

```yaml
drawing_tablet:
  brush:
    size: 2                  # base default if no per-mode override
    color: [255, 255, 0]
  mouse:
    brush: { size: 2 }
    hide_cursor: false
  tablet:
    brush: { size: 4 }
    hide_cursor: false       # set true if Wacom driver overlay duplicates cursor
  both:
    brush: { size: 3 }
    hide_cursor: false
```

The `<mode>.brush.size` and `<mode>.hide_cursor` keys override the base values when `drawing_input` matches the mode.

### 2.3 Multi-monitor cursor confinement

With two screens, the tablet pen normally drifts across both displays — which means strokes intended for the experiment can end up on the wrong monitor. PhosLab confines the cursor to one monitor while the drawing screen is active (released on ENTER):

```yaml
drawing_tablet:
  cursor_clip:
    enabled: true
    monitor: primary    # primary | 0 | 1 | ... | none
```

Resolution rules:
- `primary` — the OS-designated primary monitor (where Windows places `(0, 0)`).
- An integer — index into `EnumDisplayMonitors` order (`0` is typically primary).
- `none` / `off` — disable confinement entirely.

Implementation uses `user32.ClipCursor` ([scripts/cursor_clip.py](scripts/cursor_clip.py)). Clip applies on `DrawingTablet.reset()` (entering the drawing phase) and releases on ENTER confirmation or `DrawingTablet.close()`. No-op on non-Windows platforms.

### 2.4 What the flag does and doesn't do

- **Does**: pick the title text, pick the default brush size for the active mode, optionally hide the system cursor.
- **Does not**: prevent the other device from drawing. If `drawing_input: tablet`, the mouse will still draw if you click — pygame routes both through the same `MOUSEBUTTONDOWN` event with no source field. Adding source-discrimination would require the Wintab or Windows Pointer Input APIs (not implemented; not currently in scope).

---

## Part 3 — Using the tablet during an experiment

1. Plug in the tablet before launching PhosLab.
2. (Optional) Open Wacom Tablet Properties from the system tray to configure tablet area mapping. Default is full-tablet-to-full-screen, which is usually what you want for percept drawing.
3. Run the experiment as usual:
   ```
   cd percept_mapper
   uv run python main.py
   ```
4. When the drawing screen appears, hover the pen over the tablet to position the cursor, then touch the tablet surface to start a stroke. Lift to end a stroke. **ENTER** to confirm, **X** to undo the last stroke.

---

## Known limitations

| Feature              | Status | Reason / Workaround |
|----------------------|--------|---------------------|
| Pressure-sensitive brush | Not implemented | Pygame mouse events don't carry pressure. Would require Wintab integration via Python ctypes shim. ~80–120 LoC. |
| Tilt-sensitive brush     | Not implemented | Same reason as above. |
| Eraser end of pen        | Not implemented | Wacom driver maps eraser to right-click; would need event-source filtering to distinguish from mouse right-click. |
| Tablet button mapping    | Configurable in Wacom Tablet Properties, but not consumed by PhosLab. |
| Source discrimination (tablet-only or mouse-only) | Not enforced | pygame can't distinguish. Both devices always physically active regardless of `drawing_input`. |

---

## Troubleshooting

### Pen moves cursor but tapping the tablet doesn't draw

The tablet's pen-touch isn't generating `MOUSEBUTTONDOWN`. Check Wacom Tablet Properties → Pen → "Tip Feel" mapping is set to "Click". If you reassigned the tip button to something else, drawing won't fire.

### Cursor jumps unpredictably when using the pen

Tablet-to-screen mapping is misconfigured. Open Wacom Tablet Properties → Mapping. Set screen area = the stimulus monitor, and tablet area = full tablet. Aspect ratio mismatches can cause warped cursor movement.

### Tablet works in other apps but not in PhosLab

PhosLab uses standard pygame mouse events — if the cursor follows the pen in any other app (Notepad, browser), it should also follow it in PhosLab. If only PhosLab fails to receive events, check that `drawing_input` and the `drawing_tablet:` block parse without YAML errors. Look for `[CONFIG] Pincel de dibujo: mode=...` near the top of the experiment startup output.

### Status: Unknown after a reboot

The tablet is unplugged (or USB cable is loose). Replug.

### Service `WTabletServicePro` won't start

Reboot. The service usually does not start cleanly on a fresh install without a full Windows restart. If it still won't start after restart, reinstall the driver as Administrator.

### Pen doesn't show in Wacom Tablet Properties

The driver doesn't see the pen. Common causes:
- Pen battery (some Wacom pens need a battery — Bamboo CTH-series pens are passive, but newer pens use batteries).
- Pen tip worn out or stuck.
- USB cable only providing power, not data.

---

## File reference

| Path | Role |
|------|------|
| [`scripts/tablet.py`](scripts/tablet.py) | `DrawingTablet` class. Consumes pygame mouse events; mode-aware title and brush. |
| [`config/params.yaml`](config/params.yaml) | `drawing_input` flag + per-mode overrides. |
| [`main.py`](main.py) (around line 758) | Reads `drawing_input`, picks per-mode brush size, passes to `DrawingTablet`. |
