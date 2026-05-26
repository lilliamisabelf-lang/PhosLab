"""One-shot: extract 4 individual AprilTag PNGs from the Pupil-Labs JPG sheet
and bake in a white quiet-zone border so they're robust on a black background.

Source sheet: C:\\Users\\admin\\pupil\\aprilTags\\apriltags_tag36h11_0-23.jpg
              (24 tags in a 6-row x 4-col grid, IDs 0..23 left-to-right, top-to-bottom)

Output:       <percept_mapper>/assets/apriltags/tag_0.png ... tag_3.png

Run from repo root:
    uv run --project percept_mapper python percept_mapper/scripts/extract_apriltags.py
"""

import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cv2

SHEET = Path(r"C:\Users\admin\pupil\aprilTags\apriltags_tag36h11_0-23.jpg")
GRID_ROWS = 6
GRID_COLS = 4
TAG_IDS = [0, 1, 2, 3]  # which IDs to extract (first row = TL, TR, BL, BR pairs ok)
OUT_DIR = Path(__file__).resolve().parents[1] / "assets" / "apriltags"
TARGET_TAG_PX = 200          # final tag art size (the black/white tag itself)
QUIET_ZONE_FRACTION = 0.20   # white border = 20% of tag width (way over the spec)


def find_tag_bboxes(sheet_bgr):
    """Detect the dark tag bounding boxes on the sheet via thresholding + contours.
    Returns list of (x, y, w, h) sorted top-to-bottom, left-to-right.
    """
    gray = cv2.cvtColor(sheet_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 40 or h < 40:
            continue
        if abs(w - h) / max(w, h) > 0.2:
            continue
        bboxes.append((x, y, w, h))

    if len(bboxes) != GRID_ROWS * GRID_COLS:
        raise RuntimeError(
            f"Expected {GRID_ROWS * GRID_COLS} tag bboxes, found {len(bboxes)}. "
            "Sheet layout assumption is wrong; inspect manually."
        )

    avg_h = float(np.mean([b[3] for b in bboxes]))
    bboxes.sort(key=lambda b: (round(b[1] / (avg_h * 0.5)), b[0]))
    return bboxes


def crop_and_pad(sheet_bgr, bbox, tag_id):
    x, y, w, h = bbox
    crop = sheet_bgr[y:y + h, x:x + w]
    crop = cv2.resize(crop, (TARGET_TAG_PX, TARGET_TAG_PX), interpolation=cv2.INTER_NEAREST)

    quiet = int(TARGET_TAG_PX * QUIET_ZONE_FRACTION)
    padded_size = TARGET_TAG_PX + 2 * quiet
    canvas = np.full((padded_size, padded_size, 3), 255, dtype=np.uint8)
    canvas[quiet:quiet + TARGET_TAG_PX, quiet:quiet + TARGET_TAG_PX] = crop
    return canvas


def main():
    if not SHEET.exists():
        print(f"[extract] ✗ No existe: {SHEET}")
        sys.exit(1)

    sheet = cv2.imread(str(SHEET), cv2.IMREAD_COLOR)
    if sheet is None:
        print(f"[extract] ✗ No se pudo cargar JPG: {SHEET}")
        sys.exit(1)

    print(f"[extract] Hoja cargada: {sheet.shape[1]}x{sheet.shape[0]}")
    bboxes = find_tag_bboxes(sheet)
    print(f"[extract] {len(bboxes)} tags detectados en la rejilla.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for tag_id in TAG_IDS:
        if tag_id >= len(bboxes):
            print(f"[extract] ⚠ tag_id {tag_id} fuera de rango (max {len(bboxes) - 1})")
            continue
        out_img = crop_and_pad(sheet, bboxes[tag_id], tag_id)
        out_path = OUT_DIR / f"tag_{tag_id}.png"
        cv2.imwrite(str(out_path), out_img)
        print(f"[extract] ✓ {out_path}  ({out_img.shape[1]}x{out_img.shape[0]})")

    print(f"[extract] Listo. PNGs en {OUT_DIR}")


if __name__ == "__main__":
    main()
