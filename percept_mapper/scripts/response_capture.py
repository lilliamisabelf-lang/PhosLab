"""Response capture adapters and response-analysis helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class ResponseResult:
    mode: str
    status: str = "ok"
    response_xy: tuple[float, float] | None = None
    response_file: str | None = None
    response_file_type: str | None = None
    raw_file: str | None = None
    attempts: int | None = None
    debug: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        metadata = {
            "response_mode": self.mode,
            "response_status": self.status,
            "response_xy": (
                [float(self.response_xy[0]), float(self.response_xy[1])]
                if self.response_xy is not None
                else None
            ),
            "response_file": self.response_file,
            "response_file_type": self.response_file_type,
        }
        if self.raw_file:
            metadata["raw_file"] = self.raw_file
        if self.attempts is not None:
            metadata["response_attempts"] = self.attempts
        if "extraction" in self.debug:
            metadata["response_extraction"] = self.debug["extraction"]

        # Legacy compatibility for existing analyzers, scripts, and saved data.
        if self.mode == "drawing":
            metadata["drawing_file"] = self.response_file
        elif self.mode == "saccade":
            metadata["saccade_samples_file"] = self.response_file
        return metadata


class DrawingResponseCapture:
    mode = "drawing"

    def __init__(self, response_screen):
        self.response_screen = response_screen
        self._canvas = None
        self.last_status = "ok"

    def reset(self):
        self._canvas = None
        self.response_screen.reset()
        self.last_status = "unknown"

    def update(self, screen, events) -> bool:
        finished, output = self.response_screen.update(screen, events)
        if not finished:
            return False
        self._canvas = output
        status = getattr(self.response_screen, "last_status", None)
        self.last_status = status if status and status != "unknown" else "ok"
        return True

    def save_result(
        self,
        output_dir: Path,
        *,
        drawing_filename: str,
        saccade_filename: str | None = None,
    ) -> ResponseResult:
        if self._canvas is None:
            raise RuntimeError("Drawing response finished without a canvas")
        import pygame

        output_dir = Path(output_dir)
        filename = drawing_filename
        pygame.image.save(self._canvas, str(output_dir / filename))
        return ResponseResult(
            mode="drawing",
            status=self.last_status or "ok",
            response_file=filename,
            response_file_type="png",
        )

    def close(self):
        if hasattr(self.response_screen, "close"):
            self.response_screen.close()


class SaccadeResponseCapture:
    mode = "saccade"

    def __init__(self, response_screen):
        self.response_screen = response_screen
        self._payload: dict[str, Any] | None = None
        self.last_status = "unknown"

    def reset(self):
        self._payload = None
        self.last_status = "unknown"
        self.response_screen.reset()

    def update(self, screen, events) -> bool:
        finished, output = self.response_screen.update(screen, events)
        if not finished:
            return False
        if not isinstance(output, dict):
            raise TypeError("Saccade response must return a dict payload")
        status = output.get("status", "unknown")
        if (
            status != "ok"
            and hasattr(self.response_screen, "should_rerun")
            and self.response_screen.should_rerun()
        ):
            print(
                f"      [SaccadeScreen] retry silencioso "
                f"({output.get('attempts')}/{output.get('max_attempts')}) "
                f"motivo={status}"
            )
            self.response_screen.reset()
            return False
        self._payload = output
        self.last_status = status
        return True

    def save_result(
        self,
        output_dir: Path,
        *,
        drawing_filename: str | None = None,
        saccade_filename: str,
    ) -> ResponseResult:
        if self._payload is None:
            raise RuntimeError("Saccade response finished without a payload")
        output_dir = Path(output_dir)
        filename = saccade_filename
        record = {
            "response_xy": self._payload.get("response_xy"),
            "status": self._payload.get("status"),
            "extraction": self._payload.get("extraction"),
            "attempts": self._payload.get("attempts"),
            "max_attempts": self._payload.get("max_attempts"),
            "capture_duration_ms": self._payload.get("capture_duration_ms"),
            "anchor_xy": self._payload.get("anchor_xy"),
            "samples": self._payload.get("samples", []),
        }
        with open(output_dir / filename, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        response_xy = self._payload.get("response_xy")
        return ResponseResult(
            mode="saccade",
            status=self._payload.get("status", "unknown"),
            response_xy=(
                (float(response_xy[0]), float(response_xy[1]))
                if response_xy is not None
                else None
            ),
            response_file=filename,
            response_file_type="json",
            raw_file=filename,
            attempts=self._payload.get("attempts"),
            debug={"extraction": self._payload.get("extraction")},
        )

    def close(self):
        if hasattr(self.response_screen, "close"):
            self.response_screen.close()


def apply_response_metadata(metadata: dict[str, Any], result: ResponseResult) -> None:
    metadata.update(result.to_metadata())


def write_response_summary(f, metadata: dict[str, Any]) -> None:
    mode = metadata.get("response_mode", "drawing")
    if mode == "saccade":
        f.write(
            f"  Modo respuesta: saccade ({metadata.get('response_extraction')}, "
            f"status={metadata.get('response_status')}, "
            f"intentos={metadata.get('response_attempts')})\n"
        )
        f.write(f"  response_xy: {metadata.get('response_xy')}\n")
        f.write(
            f"  Archivo respuesta: {metadata.get('response_file') or metadata.get('saccade_samples_file', '-')}\n"
        )
        return

    f.write(
        f"  Archivo respuesta: {metadata.get('response_file') or metadata.get('drawing_file', '-')}\n"
    )


def _point_features(response_xy) -> dict[str, Any] | None:
    if not response_xy:
        return None
    x = float(response_xy[0])
    y = float(response_xy[1])
    return {
        "centroid": (x, y),
        "n_pixels": 1,
        "intensity_sum": 1.0,
        "bbox": {
            "left": int(x),
            "top": int(y),
            "right": int(x) + 1,
            "bottom": int(y) + 1,
            "width": 1,
            "height": 1,
            "area": 1,
        },
        "fill_ratio": 1.0,
    }


def resolve_response_features(
    metadata: dict[str, Any],
    base_dir: Path,
    extract_drawing_features: Callable[[Path], dict[str, Any] | None],
) -> dict[str, Any]:
    mode = metadata.get("response_mode", "drawing")
    if mode == "saccade":
        features = _point_features(metadata.get("response_xy"))
        if features is None:
            return {
                "ok": False,
                "mode": mode,
                "error": f"Sin respuesta saccade valida (status={metadata.get('response_status', 'unknown')})",
            }
        return {
            "ok": True,
            "mode": mode,
            "features": features,
            "source_file": metadata.get("response_file")
            or metadata.get("saccade_samples_file", ""),
        }

    filename = metadata.get("response_file") or metadata.get("drawing_file")
    if not filename:
        return {"ok": False, "mode": mode, "error": "Sin archivo de dibujo"}
    drawing_path = Path(base_dir) / filename
    if not drawing_path.exists():
        return {
            "ok": False,
            "mode": mode,
            "error": f"No encontrado: {drawing_path.name}",
        }
    features = extract_drawing_features(drawing_path)
    if features is None:
        return {"ok": False, "mode": mode, "error": "Dibujo vacio"}
    return {
        "ok": True,
        "mode": mode,
        "features": features,
        "source_file": drawing_path.name,
    }
