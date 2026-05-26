"""Smoke tests for response metadata and analyzer feature resolution."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw

from scripts.response_capture import ResponseResult, resolve_response_features


def _extract_centroid(path: Path):
    img = Image.open(path).convert("RGB")
    pixels = img.load()
    xs = []
    ys = []
    for y in range(img.height):
        for x in range(img.width):
            if sum(pixels[x, y]) > 10:
                xs.append(x)
                ys.append(y)
    if not xs:
        return None
    return {
        "centroid": (sum(xs) / len(xs), sum(ys) / len(ys)),
        "n_pixels": len(xs),
        "intensity_sum": float(len(xs)),
        "bbox": {
            "left": min(xs),
            "top": min(ys),
            "right": max(xs),
            "bottom": max(ys),
            "width": max(xs) - min(xs) + 1,
            "height": max(ys) - min(ys) + 1,
            "area": (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1),
        },
        "fill_ratio": 1.0,
    }


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        drawing = base / "drawing_1.png"
        img = Image.new("RGB", (20, 20), "black")
        ImageDraw.Draw(img).point((7, 9), fill="white")
        img.save(drawing)

        drawing_meta = ResponseResult(
            mode="drawing",
            response_file=drawing.name,
            response_file_type="png",
        ).to_metadata()
        drawing_result = resolve_response_features(drawing_meta, base, _extract_centroid)
        assert drawing_result["ok"]
        assert drawing_result["features"]["centroid"] == (7.0, 9.0)

        saccade_meta = ResponseResult(
            mode="saccade",
            status="ok",
            response_xy=(11.0, 13.0),
            response_file="saccade_samples_1.json",
            response_file_type="json",
        ).to_metadata()
        saccade_result = resolve_response_features(saccade_meta, base, _extract_centroid)
        assert saccade_result["ok"]
        assert saccade_result["features"]["centroid"] == (11.0, 13.0)

        missing_drawing = resolve_response_features(
            {"response_mode": "drawing", "response_file": "missing.png"},
            base,
            _extract_centroid,
        )
        assert not missing_drawing["ok"]

        missing_saccade = resolve_response_features(
            {"response_mode": "saccade", "response_status": "no_fixation"},
            base,
            _extract_centroid,
        )
        assert not missing_saccade["ok"]

    print("[response_capture_smoke_test] ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
