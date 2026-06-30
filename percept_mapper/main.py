"""
Simulador de Prótesis Cortical Visual

"""

# REGION IMPORTS
import sys
import os

# Fix Windows console encoding for Unicode characters
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pygame
import time
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import yaml

_ACTIVE_RESPONSE_SCREEN = None
_APRILTAG_OVERLAY = None
# Overlay de MAPPING DEBUG MODE (rejilla + marcador de fosfeno). None salvo que
# debug.mapping_debug_mode esté activo. Se dibuja en cada _display_flip, así
# que aparece en todas las fases (prestim/stim/poststim/drawing).
_MAPPING_DEBUG_OVERLAY = None


def _set_active_response_screen(response_screen):
    global _ACTIVE_RESPONSE_SCREEN
    _ACTIVE_RESPONSE_SCREEN = response_screen


def _close_response_screen(response_screen=None):
    global _ACTIVE_RESPONSE_SCREEN
    target = response_screen or _ACTIVE_RESPONSE_SCREEN
    if target is not None and hasattr(target, "close"):
        try:
            target.close()
        except Exception as e:
            print(f"[CLEANUP] ⚠ Error cerrando pantalla de respuesta: {e}")
    if target is _ACTIVE_RESPONSE_SCREEN:
        _ACTIVE_RESPONSE_SCREEN = None


def _display_flip(screen=None):
    target = screen or pygame.display.get_surface()
    if target is not None:
        # La rejilla de debug va primero (fondo); los AprilTags de las esquinas
        # quedan por encima para no perder el surface tracking.
        if _MAPPING_DEBUG_OVERLAY is not None:
            _MAPPING_DEBUG_OVERLAY.draw(target)
        if _APRILTAG_OVERLAY is not None:
            _APRILTAG_OVERLAY.draw(target)
    pygame.display.flip()


def _set_mapping_debug_marker(stimulation_screen):
    """Fija el marcador del overlay de debug en la posición del fosfeno actual.
    No-op si el debug está desactivado o es un catch trial (sin fosfeno)."""
    overlay = _MAPPING_DEBUG_OVERLAY
    if overlay is None or not getattr(stimulation_screen, "show_phosphene", False):
        return
    px = stimulation_screen.phosphene_position
    mapper = getattr(stimulation_screen, "dynaphos_mapper", None)
    idx = getattr(stimulation_screen, "active_electrode_index", None)
    deg = ecc = polar = None
    if mapper is not None and idx is not None:
        try:
            info = mapper.get_electrode_info(idx)
            # El marcador se dibuja en la posición MOSTRADA (px con error simulado),
            # así que se anota con los grados MOSTRADOS para que px y grados coincidan.
            deg = info.get("displayed_position_deg") or info["visual_position_deg"]
            ecc = info.get("displayed_eccentricity_deg", info.get("eccentricity_deg"))
            polar = float(np.degrees(np.arctan2(deg[1], deg[0])))
        except Exception:
            deg = ecc = polar = None
    if ecc is None:  # fallback: derivar de px con la geometría del overlay
        dx = (px[0] - overlay.cx) / max(overlay.ppd_x, 1e-6)
        dy = -(px[1] - overlay.cy) / max(overlay.ppd_y, 1e-6)
        deg = (dx, dy)
        ecc = float(np.hypot(dx, dy))
        polar = float(np.degrees(np.arctan2(dy, dx)))
    overlay.set_phosphene(px, deg, ecc, polar, electrode_index=idx)


def _assert_phosphenes_onscreen(mapper, screen_w, screen_h, vf_scope_deg, allow_offscreen):
    """Feasibility gate antes de lanzar: si algún electrodo ACTIVO cae fuera de
    la pantalla con la geometría actual, su fosfeno sería invisible (y se
    registraría como si se hubiera mostrado). Por defecto se ABORTA listando los
    electrodos afectados; con screen.allow_offscreen: true solo se avisa.

    Con el mapeo isotrópico anclado al lado corto, el ecc máximo visible en
    CUALQUIER ángulo es ~vf_scope_deg (el meridiano del lado corto es el límite).
    """
    offscreen = []
    for idx, is_active in enumerate(mapper.active_electrodes):
        if not is_active:
            continue
        try:
            px = mapper.get_phosphene_position(idx)
        except Exception:
            px = None
        if px is None:
            continue
        x, y = float(px[0]), float(px[1])
        if 0 <= x < screen_w and 0 <= y < screen_h:
            continue
        ecc = None
        try:
            ecc = mapper.get_electrode_info(idx).get("eccentricity_deg")
        except Exception:
            pass
        offscreen.append((idx, x, y, ecc))

    if not offscreen:
        return

    head = (
        f"FOSFENOS FUERA DE PANTALLA: con vf_scope_deg={vf_scope_deg:g}°, "
        f"{len(offscreen)} de los electrodos seleccionados caen fuera de "
        f"{screen_w}x{screen_h} y NO serían visibles:"
    )
    body = [
        f"    electrodo {idx}: px=({x:.0f},{y:.0f})"
        + (f", ecc={ecc:.1f}°" if ecc is not None else "")
        for idx, x, y, ecc in offscreen
    ]
    tail = (
        f"  El ecc máximo visible (en cualquier ángulo) con esta geometría es "
        f"~{vf_scope_deg:g}°. Soluciones: sube vf_scope_deg (o ponlo a 'auto'), "
        f"acerca la pantalla, o reduce la excentricidad de los electrodos.\n"
        f"  Para ejecutar igualmente: screen.allow_offscreen: true"
    )
    msg = "\n".join([head, *body, tail])

    if allow_offscreen:
        print("[INIT] ⚠ " + msg)
        print("[INIT] allow_offscreen=true → se ejecuta de todas formas.")
        return
    print("[INIT] ✗ " + msg)
    raise SystemExit(1)


# ============================================
# HELPERS: corrientes por electrodo (sparse)
# ============================================


def _parse_sparse_currents_mapping(currents_cfg):
    """Normaliza configuraciones sparse de corrientes.

    Soporta 2 formatos:
      1) Dict directo: {10: 10, 20: 30, 30: 40, default_uA: 0}
      2) Dict anidado: {default_uA: 0, overrides: {10: 10, 20: 30}}

    Devuelve: (default_uA: float, overrides: dict[int,float])
    """
    if not isinstance(currents_cfg, dict):
        raise TypeError("currents_cfg debe ser dict")

    # Formato anidado
    if "overrides" in currents_cfg and isinstance(currents_cfg.get("overrides"), dict):
        overrides_raw = currents_cfg.get("overrides") or {}
        default_uA = currents_cfg.get("default_uA", currents_cfg.get("default", 0.0))
    else:
        overrides_raw = dict(currents_cfg)
        default_uA = overrides_raw.pop(
            "default_uA",
            overrides_raw.pop("default", overrides_raw.pop("_default", 0.0)),
        )

    try:
        default_uA = float(default_uA)
    except (TypeError, ValueError):
        default_uA = 0.0

    overrides = {}
    for k, v in (overrides_raw or {}).items():
        # Ignorar claves no numéricas (p.ej. metadatos)
        try:
            idx = int(k)
        except (TypeError, ValueError):
            continue
        try:
            overrides[idx] = float(v)
        except (TypeError, ValueError):
            continue

    return default_uA, overrides


def _resolve_currents_uA(currents_cfg, total_electrodes, default_uA_fallback=90.0):
    """Devuelve siempre una LISTA de corrientes.

    - Si currents_cfg es list: se devuelve tal cual.
      *Si su longitud == total_electrodes, se interpreta como "por índice".
      *Si no, se interpreta como "por orden" (comportamiento existente).

    - Si currents_cfg es dict: se expande a longitud total_electrodes usando default_uA.
      Esto permite definir solo unos pocos electrodos sin un array enorme.
    """
    if isinstance(currents_cfg, list):
        return currents_cfg

    if isinstance(currents_cfg, dict):
        default_uA, overrides = _parse_sparse_currents_mapping(currents_cfg)
        if total_electrodes is None:
            raise ValueError(
                "total_electrodes es requerido para corrientes sparse (dict)"
            )
        n = int(total_electrodes)
        if n <= 0:
            return []
        out = [float(default_uA)] * n
        for idx, val in overrides.items():
            if 0 <= idx < n:
                out[idx] = float(val)
        return out

    # None u otros tipos: usar fallback
    if total_electrodes is None:
        return [float(default_uA_fallback)]
    return [float(default_uA_fallback)] * int(total_electrodes)


def _select_current_uA(
    currents_uA, electrode_index, order_index, total_electrodes, default_current_uA
):
    """Resuelve la corriente final para un electrodo.

    Reglas:
    - Lista vacia/None: default_current_uA
    - Lista len == total_electrodes: se indexa por electrode_index
    - Lista len != total_electrodes: se interpreta por orden (order_index),
      y los restantes usan default_current_uA
    - Fuera de rango: default_current_uA
    """
    if not currents_uA:
        return float(default_current_uA)

    if len(currents_uA) == total_electrodes:
        if 0 <= electrode_index < len(currents_uA):
            return float(currents_uA[electrode_index])
        return float(default_current_uA)

    if 0 <= order_index < len(currents_uA):
        return float(currents_uA[order_index])

    return float(default_current_uA)


def _resolve_mapping_electrode_indices(
    mapping_config, cli_electrode, total_electrodes, electrode_index_map=None
):
    """Resuelve qué electrodos se van a mapear en modo 'mapping'.

    Compatibilidad:
    - CLI manda siempre si está presente.
    - mapping.electrode_index puede ser:
        * int
        * list[int]
        * "all" / "todos"
        * dict con:
            - mode: "manual" + indices: [...]
            - mode: "range" + start/end/step
            - mode: "all"
    """
    if cli_electrode is not None:
        return cli_electrode if isinstance(cli_electrode, list) else [cli_electrode]

    mapping_config = mapping_config or {}
    print(f"DEBUG electrodes_by_implant: {mapping_config.get('electrodes_by_implant')}")
    print(f"DEBUG electrode_index_map es None: {electrode_index_map is None}")

    electrodes_by_implant = mapping_config.get("electrodes_by_implant", None)
    if (
        electrodes_by_implant
        and isinstance(electrodes_by_implant, list)
        and electrode_index_map
    ):
        inverse_map = {
            (str(v[0]), int(v[1])): k for k, v in electrode_index_map.items()
        }
        global_indices = []
        for block in electrodes_by_implant:
            imp_id = str(block.get("implant_id", "")).strip()
            indices = block.get("electrode_index", [])
            if isinstance(indices, int):
                indices = [indices]
            for elec_idx in indices:
                key = (imp_id, int(elec_idx))
                if key in inverse_map:
                    global_indices.append(inverse_map[key])
                    print(
                        f" [{imp_id}] electrodo {elec_idx} → índice global {inverse_map[key]}"
                    )
                else:
                    print(
                        f"  [{imp_id}] electrodo {elec_idx} no encontrado en CSV, saltando"
                    )
        return global_indices

    spec = mapping_config.get("electrode_index", 0)

    # Caso string: all/todos
    if isinstance(spec, str):
        key = spec.strip().lower()
        if key in {"all", "todos", "todo"}:
            return list(range(int(total_electrodes)))
        # intento: "range:0-100" (no documentado, pero tolerante)
        if key.startswith("range:"):
            try:
                body = key.split(":", 1)[1]
                start_s, end_s = body.split("-", 1)
                start = int(start_s)
                end = int(end_s)
                return list(range(max(0, start), min(int(total_electrodes), end + 1)))
            except Exception:
                pass

    # Caso dict: modo manual/range/all
    if isinstance(spec, dict):
        mode = str(spec.get("mode", "manual")).strip().lower()
        if mode in {"all", "todos", "todo"}:
            return list(range(int(total_electrodes)))
        if mode == "range":
            start = int(spec.get("start", 0))
            end = int(spec.get("end", int(total_electrodes)))
            step = int(spec.get("step", 1))
            start = max(0, start)
            end = min(int(total_electrodes), end)
            step = max(1, step)
            return list(range(start, end, step))
        # manual
        indices = spec.get("indices", spec.get("electrodes", spec.get("list", [])))
        if isinstance(indices, list):
            return [int(x) for x in indices]
        if indices is None:
            return []
        return [int(indices)]

    # Caso int/list (modo antiguo)
    if isinstance(spec, list):
        return [int(x) for x in spec]
    return [int(spec)]


import argparse
from PIL import Image as PILImage
import platform
import ctypes

# Importar módulos propios
from core.eye_tracker import EyeTracker
from core.mouse_tracker import MouseTracker
from core.pupil_tracker import PupilTracker
from scripts.anchor_screen import AnchorScreen
from scripts.stimulation_screen import StimulationScreen
from scripts.debug_overlay import MappingDebugOverlay
from scripts.tablet import DrawingTablet
from scripts.webcam_viewer import WebcamViewer
from scripts.dynaphos_adapter import (
    DynaphosMapper,
    load_active_electrodes_config,
    load_timing_config,
)
from scripts.gaze_trace import GazeTrace
from scripts.phosphene_mapping import PhospheneMappingExperiment
from scripts.response_capture import (
    DrawingResponseCapture,
    SaccadeResponseCapture,
    apply_response_metadata,
    write_response_summary,
)
from scripts.trial_sequence import build_trial_list, summary as trial_summary
from scripts.schemas import SessionMetadata, TrialSequenceConfig

# endregion

# REGION CONFIGURACIÓN
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
FULLSCREEN = True
FPS = 60  # Valor por defecto
GAZE_TRACE_DURATION_MS = 300  # Duration of the raw gaze trace tail in ms

# Cargar tiempos desde params.yaml
timing_config = load_timing_config()
PRESTIMULATION_MS = timing_config["prestimulation_ms"]
STIMULATION_MS = timing_config["stimulation_ms"]
POSTSTIMULATION_MS = timing_config["poststimulation_ms"]
INTERSTIMULATION_MS = timing_config["interstimulation_ms"]
MAX_FIXATION_WAIT_MS = timing_config["max_fixation_wait_ms"]

# Debug toggles cargados perezosamente. Se rellenan al iniciar main() desde
# `config` (que se carga ahí). Default False: no debug noise por defecto.
DEBUG_SHOW_INTERSTIM_TEXT = False

# endregion


# ============================================
# VISUALIZACIÓN DE RESULTADOS EN PYGAME
# ============================================


def show_mapping_results_pygame(
    screen, clock, results, electrode_dir, screen_width, screen_height
):
    """
    Muestra los resultados del mapeo en la pantalla negra de pygame

    Args:
        screen: Superficie de pygame
        clock: Reloj de pygame
        results: Diccionario con resultados del análisis
        electrode_dir: Path a la carpeta del electrodo con los dibujos
        screen_width: Ancho de la pantalla
        screen_height: Alto de la pantalla
    """
    # Colores
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    YELLOW = (255, 255, 0)
    CYAN = (0, 255, 255)
    LIGHT_GRAY = (180, 180, 180)
    RED = (255, 0, 0)

    # Crear superficie para composición
    composite_surface = pygame.Surface((screen_width, screen_height))
    composite_surface.fill(BLACK)

    # Cargar y superponer todos los dibujos en gris claro
    electrode_dir = Path(electrode_dir)
    num_reps = results["num_total_repetitions"]

    for i in range(1, num_reps + 1):
        drawing_file = electrode_dir / f"repetition_{i:03d}.png"
        if drawing_file.exists():
            try:
                # Cargar imagen PIL
                pil_img = PILImage.open(drawing_file).convert("RGB")
                img_array = np.array(pil_img)

                # Crear superficie pygame
                drawing_surface = pygame.surfarray.make_surface(
                    img_array.swapaxes(0, 1)
                )

                # Convertir píxeles no negros a gris claro
                pixels = pygame.surfarray.pixels3d(drawing_surface)
                # Identificar píxeles dibujados (no negros)
                drawn_mask = (
                    (pixels[:, :, 0] > 10)
                    | (pixels[:, :, 1] > 10)
                    | (pixels[:, :, 2] > 10)
                )
                # Cambiar a gris claro
                pixels[drawn_mask] = LIGHT_GRAY
                del pixels  # Liberar bloqueo de superficie

                # Blit con transparencia
                composite_surface.blit(
                    drawing_surface, (0, 0), special_flags=pygame.BLEND_ADD
                )
            except Exception as e:
                print(f"⚠ Error cargando {drawing_file.name}: {e}")

    # Obtener posiciones
    mean_pos = (int(results["mean_position"]["x"]), int(results["mean_position"]["y"]))
    stim_pos = tuple(results["stimulation_position"])
    centroids = results["centroids"]

    # Fuentes
    font_large = pygame.font.Font(None, 76)
    font_medium = pygame.font.Font(None, 56)
    font_small = pygame.font.Font(None, 44)

    running = True
    while running:
        # Dibujar base
        screen.blit(composite_surface, (0, 0))

        # Dibujar centroides individuales (pequeños puntos amarillos)
        for centroid in centroids:
            cx, cy = int(centroid[0]), int(centroid[1])
            pygame.draw.circle(screen, (200, 200, 100), (cx, cy), 4)

        # Dibujar posición del estímulo (estrella cyan)
        star_size = 20
        pygame.draw.circle(screen, CYAN, stim_pos, star_size, 3)
        pygame.draw.line(
            screen,
            CYAN,
            (stim_pos[0] - star_size, stim_pos[1]),
            (stim_pos[0] + star_size, stim_pos[1]),
            3,
        )
        pygame.draw.line(
            screen,
            CYAN,
            (stim_pos[0], stim_pos[1] - star_size),
            (stim_pos[0], stim_pos[1] + star_size),
            3,
        )

        # Dibujar posición promedio (cruz amarilla grande)
        cross_size = 25
        pygame.draw.line(
            screen,
            YELLOW,
            (mean_pos[0] - cross_size, mean_pos[1]),
            (mean_pos[0] + cross_size, mean_pos[1]),
            5,
        )
        pygame.draw.line(
            screen,
            YELLOW,
            (mean_pos[0], mean_pos[1] - cross_size),
            (mean_pos[0], mean_pos[1] + cross_size),
            5,
        )
        pygame.draw.circle(screen, YELLOW, mean_pos, cross_size, 5)

        # Dibujar círculo de desviación estándar
        std_x = results["std_position"]["x"]
        std_y = results["std_position"]["y"]
        std_radius = int(np.sqrt(std_x**2 + std_y**2))
        if std_radius > 0:
            pygame.draw.circle(screen, RED, mean_pos, std_radius, 2)

        # Título
        title = f"Análisis de Mapeo - Electrodo {results['electrode_index']}"
        title_surface = font_large.render(title, True, WHITE)
        title_rect = title_surface.get_rect(center=(screen_width // 2, 40))
        screen.blit(title_surface, title_rect)

        # Subtítulo
        subtitle = f"{results['num_valid_repetitions']} repeticiones válidas"
        subtitle_surface = font_medium.render(subtitle, True, WHITE)
        subtitle_rect = subtitle_surface.get_rect(center=(screen_width // 2, 85))
        screen.blit(subtitle_surface, subtitle_rect)

        # Leyenda (lado izquierdo superior)
        legend_x = 30
        legend_y = 140
        line_height = 35

        # Leyenda - Dibujos individuales (gris)
        pygame.draw.circle(screen, LIGHT_GRAY, (legend_x + 10, legend_y), 8)
        legend_surface = font_small.render("Dibujos individuales", True, WHITE)
        screen.blit(legend_surface, (legend_x + 30, legend_y - 12))

        # Leyenda - Posición promedio (amarillo)
        legend_y += line_height
        pygame.draw.circle(screen, YELLOW, (legend_x + 10, legend_y), 10, 3)
        legend_surface = font_small.render("Posición promedio (media)", True, WHITE)
        screen.blit(legend_surface, (legend_x + 30, legend_y - 12))

        # Leyenda - Posición del estímulo (cyan)
        legend_y += line_height
        pygame.draw.circle(screen, CYAN, (legend_x + 10, legend_y), 10, 2)
        legend_surface = font_small.render("Posición del estímulo", True, WHITE)
        screen.blit(legend_surface, (legend_x + 30, legend_y - 12))

        # Leyenda - Desviación estándar (círculo rojo)
        legend_y += line_height
        pygame.draw.circle(screen, RED, (legend_x + 10, legend_y), 8, 2)
        legend_surface = font_small.render("Radio de desviación estándar", True, WHITE)
        screen.blit(legend_surface, (legend_x + 30, legend_y - 12))

        # Estadísticas (cuadro inferior derecho)
        stats_x = screen_width - 450
        stats_y = screen_height - 200
        stats_width = 420
        stats_height = 170

        # Fondo semi-transparente
        stats_bg = pygame.Surface((stats_width, stats_height))
        stats_bg.set_alpha(200)
        stats_bg.fill((30, 30, 30))
        screen.blit(stats_bg, (stats_x, stats_y))

        # Borde
        pygame.draw.rect(
            screen, WHITE, (stats_x, stats_y, stats_width, stats_height), 2
        )

        # Texto de estadísticas
        stats_text_x = stats_x + 15
        stats_text_y = stats_y + 15
        stats_line_height = 32

        stats = [
            f"Media: ({mean_pos[0]:.1f}, {mean_pos[1]:.1f}) px",
            f"Desv. Est.: ({std_x:.1f}, {std_y:.1f}) px",
            f"Dist. media: {results['mean_distance_from_average']:.1f} px",
            f"Dist. máxima: {results['max_distance_from_average']:.1f} px",
        ]

        for i, stat in enumerate(stats):
            stat_surface = font_small.render(stat, True, WHITE)
            screen.blit(
                stat_surface, (stats_text_x, stats_text_y + i * stats_line_height)
            )

        # Instrucciones (parte inferior)
        instr_text = "Presiona ENTER o ESC para continuar"
        instr_surface = font_medium.render(instr_text, True, (200, 200, 200))
        instr_rect = instr_surface.get_rect(
            center=(screen_width // 2, screen_height - 40)
        )
        screen.blit(instr_surface, instr_rect)

        # Eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN or event.key == pygame.K_ESCAPE:
                    running = False

        _display_flip(screen)
        clock.tick(30)


# FUNCIÓN PRINCIPAL


def main():
    global FPS  # ← Declarar que vamos a modificar la variable global

    """
    Función principal del simulador
    """
    # PARSEAR ARGUMENTOS DE LÍNEA DE COMANDOS
    parser = argparse.ArgumentParser(
        description="Simulador de Prótesis Cortical Visual",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python main.py                    # Modo según params.yaml
  python main.py 85 5              # Modo mapping: electrodo 85, 5 repeticiones
  python main.py --electrode 85    # Modo mapping: electrodo 85 (repeticiones por defecto)
        """,
    )
    parser.add_argument(
        "electrode_index",
        type=int,
        nargs="?",
        help="Índice del electrodo a mapear (activa modo mapping automáticamente)",
    )
    parser.add_argument(
        "num_repetitions",
        type=int,
        nargs="?",
        help="Número de repeticiones para el mapeo (default: desde params.yaml)",
    )
    parser.add_argument(
        "--electrode",
        type=int,
        dest="electrode_index_flag",
        help="Índice del electrodo a mapear (alternativa)",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        dest="num_repetitions_flag",
        help="Número de repeticiones (alternativa)",
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        dest="no_save",
        help="No guardar los datos del experimento en disco",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Reanudar una sesión: ruta al directorio mapping_experiments/mapping_*/. "
        "Lee session_metadata.json y salta trials cuyo trial_idx ya tiene "
        "artefactos guardados.",
    )

    args = parser.parse_args()
    SAVE_RESULTS = not args.no_save
    resume_dir = Path(args.resume) if args.resume else None

    # CARGAR CONFIGURACIÓN DESDE YAML
    print("=" * 70)
    print("CARGANDO CONFIGURACIÓN...")
    print("=" * 70)

    config_path = Path("config/params.yaml")
    if not config_path.exists():
        print(f"ERROR: No se encontró {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Debug toggles (módulo-globales para que run_interstimulation, etc.,
    # los puedan leer sin pasarlos por parámetro).
    global DEBUG_SHOW_INTERSTIM_TEXT
    DEBUG_SHOW_INTERSTIM_TEXT = bool(
        (config.get("debug") or {}).get("show_interstim_text", False)
    )

    # Extraer parámetros de pantalla
    screen_config = config["screen"]
    SCREEN_WIDTH = screen_config["width"]
    SCREEN_HEIGHT = screen_config["height"]
    FULLSCREEN = screen_config["fullscreen"]

    # Extraer parámetros de estimulación
    stim_config = config.get("stimulation", {})
    PULSE_WIDTH_US = stim_config.get("pulse_width", 200)  # microsegundos
    FREQUENCY_HZ = stim_config.get("frequency", 50)  # Hz

    # NOTA: Las corrientes se resolverán más tarde, cuando sepamos cuántos
    # electrodos existen (total_electrodes). Aquí solo guardamos la config.
    STIMULATION_CURRENTS_CFG = stim_config.get("stimulation_currents_uA", [90] * 10)
    STIMULATION_CURRENTS_STANDARD_CFG = stim_config.get(
        "stimulation_currents_standard_uA", [90] * 10
    )

    print(f"✓ Configuración cargada desde {config_path}")
    print(f"  - FPS: {FPS}")
    print(f"  - Resolución: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
    print(f"  - Fullscreen: {FULLSCREEN}")
    print(f"  - Pulse width: {PULSE_WIDTH_US} µs")
    print(f"  - Frecuencia: {FREQUENCY_HZ} Hz")
    if isinstance(STIMULATION_CURRENTS_CFG, dict):
        print("  - Corrientes MAPPING: modo sparse (dict)")
    else:
        print(
            f"  - Corrientes MAPPING: {len(STIMULATION_CURRENTS_CFG)} valores → {STIMULATION_CURRENTS_CFG[:3]}..."
        )
    if isinstance(STIMULATION_CURRENTS_STANDARD_CFG, dict):
        print("  - Corrientes STANDARD: modo sparse (dict)")
    else:
        print(
            f"  - Corrientes STANDARD: {len(STIMULATION_CURRENTS_STANDARD_CFG)} valores → {STIMULATION_CURRENTS_STANDARD_CFG[:3]}..."
        )
    print()

    # INICIALIZACIÓN

    print("[INIT] Inicializando Pygame...")

    # En Windows, el escalado DPI puede hacer que SDL/Pygame trabajen en una resolución
    # "lógica" (p.ej. 1600x900 con escala 120%) aunque el panel sea 1920x1080.
    # Para intentar trabajar en píxeles reales, marcamos el proceso como DPI-aware.
    if platform.system().lower() == "windows":
        try:
            # Windows 8.1+
            ctypes.windll.shcore.SetProcessDpiAwareness(
                2
            )  # PROCESS_PER_MONITOR_DPI_AWARE
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
    pygame.init()

    # Info del monitor
    display_info = pygame.display.Info()
    native_width = display_info.current_w
    native_height = display_info.current_h
    print(f"[INIT] Resolución nativa del monitor: {native_width}x{native_height}")

    # Aviso best-effort si la geometría física del config está desfasada
    # respecto al monitor real (típico al copiar params.yaml entre PCs). No
    # cambia el comportamiento: el mapeo px/deg usa width/vf_scope, pero
    # screen_diagonal_inches alimenta validate_eye_tracker. Corrige con:
    #   python -m scripts.screen_detect --write
    try:
        from scripts.screen_detect import detect_displays, primary_display

        _disp = primary_display(detect_displays())
        _cfg_diag = screen_config.get("screen_diagonal_inches")
        if _disp is not None and _disp.diagonal_inches and _cfg_diag:
            _det = _disp.diagonal_inches
            if abs(float(_cfg_diag) - _det) / _det > 0.10:
                print(
                    f"[INIT] ⚠ screen_diagonal_inches={_cfg_diag} no coincide con el "
                    f"monitor detectado (~{_det:.1f}\"). "
                    f"Ejecuta: python -m scripts.screen_detect --write"
                )
    except Exception:
        pass  # detección best-effort; nunca debe romper el arranque

    # Tamaño de ventana adaptativo: por defecto la ventana se ajusta a la
    # pantalla actual para funcionar en distintos PCs y monitores. Se puede
    # desactivar poniendo `screen.adaptive: false` en params.yaml para usar
    # la resolución fija indicada por width/height.
    adaptive = screen_config.get("adaptive", True)
    if adaptive and native_width > 0 and native_height > 0:
        if FULLSCREEN:
            # Fullscreen ocupa toda la pantalla nativa.
            SCREEN_WIDTH = native_width
            SCREEN_HEIGHT = native_height
        else:
            # En modo ventana dejamos margen para la barra de título / bordes.
            SCREEN_WIDTH = min(SCREEN_WIDTH, int(native_width * 0.95))
            SCREEN_HEIGHT = min(SCREEN_HEIGHT, int(native_height * 0.92))
        print(f"[INIT] Tamaño adaptativo de ventana: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")

    if FULLSCREEN:
        screen = pygame.display.set_mode(
            (SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN
        )
    else:
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    pygame.display.set_caption("Simulador Prótesis Cortical")
    clock = pygame.time.Clock()

    # AprilTag corner overlay (siempre visible en cada frame) para Pupil Surface Tracker
    global _APRILTAG_OVERLAY
    from scripts.apriltag_overlay import from_config as _build_apriltag_overlay

    try:
        _APRILTAG_OVERLAY = _build_apriltag_overlay(config)
    except Exception as e:
        print(f"[INIT] ⚠ AprilTag overlay no disponible: {e}")

    actual_width = screen.get_width()
    actual_height = screen.get_height()
    print(f"[INIT] Tamaño real de pantalla: {actual_width}x{actual_height}")
    if (actual_width, actual_height) != (SCREEN_WIDTH, SCREEN_HEIGHT):
        print(
            f"⚠ WARNING: La resolución solicitada ({SCREEN_WIDTH}x{SCREEN_HEIGHT}) no coincide con la real ({actual_width}x{actual_height})."
        )

    # Modo de entrada: 'mouse' o 'gaze'
    input_mode = config.get("input_mode", "mouse")
    print(f"[INIT] Modo de entrada: {input_mode}")

    # Inicializar tracker según modo
    tracker = None
    webcam_viewer = None

    if input_mode == "gaze":
        print("[INIT] Inicializando eye tracker (webcam)...")
        print("       Esto puede tardar unos segundos...")
        try:
            tracker = EyeTracker(camera_index=0)
            print("       ✓ Eye tracker iniciado correctamente")
        except Exception as e:
            print("=" * 70)
            print("⚠ ERROR al iniciar eye tracker:")
            print(str(e))
            print("=" * 70)
            print("\n⚠ Cayendo a modo mouse...\n")
            tracker = MouseTracker()

        # Webcam viewer solo en modo gaze con EyeTracker real
        if isinstance(tracker, EyeTracker):
            print("[INIT] Inicializando visualizador de webcam...")
            try:
                webcam_viewer = WebcamViewer(tracker)
                print("       ✓ Visualizador de webcam iniciado")
            except Exception as e:
                print(f"       ⚠ Webcam viewer no disponible: {e}")
    elif input_mode == "pupil":
        print("[INIT] Inicializando Pupil tracker (ZMQ a Pupil Capture)...")
        pupil_cfg = config.get("pupil", {}) or {}
        try:
            tracker = PupilTracker(
                address=pupil_cfg.get("address", "127.0.0.1"),
                port=pupil_cfg.get("port", 50020),
                surface_name=pupil_cfg.get("surface_name", "phoslab_screen"),
                min_confidence=pupil_cfg.get("min_confidence", 0.6),
                one_euro=pupil_cfg.get("one_euro"),
                max_sample_age_s=pupil_cfg.get("max_sample_age_s", 0.25),
            )
            print("       ✓ Pupil tracker iniciado correctamente")
        except Exception as e:
            print("=" * 70)
            print("⚠ ERROR al iniciar Pupil tracker:")
            print(str(e))
            print("=" * 70)
            if not bool(pupil_cfg.get("allow_mouse_fallback", False)):
                print(
                    "✗ No se inicia el experimento sin Pupil. Activa pupil.allow_mouse_fallback solo para pruebas."
                )
                cleanup_and_exit(None, webcam_viewer)
                return
            print(
                "⚠ Cayendo a modo mouse por configuración pupil.allow_mouse_fallback=true."
            )
            tracker = MouseTracker()
    else:
        tracker = MouseTracker()

    # Alias para compatibilidad
    eye_tracker = tracker

    # Crear params con el tamaño REAL
    params = {
        "screen": {
            "width": actual_width,
            "height": actual_height,
            "background_color": screen_config["background_color"],
            "anchor_circle": {
                "radius": screen_config["anchor_circle"]["radius"],  # ← Desde YAML
                "color": screen_config["anchor_circle"]["color"],  # ← Desde YAML
                "thickness": screen_config["anchor_circle"][
                    "thickness"
                ],  # ← Desde YAML
                "tolerance_radius": screen_config["anchor_circle"][
                    "tolerance_radius"
                ],  # ← Desde YAML
            },
        }
    }

    print(f"[CONFIG] Círculo de anclaje:")
    print(f"         - Radio visual: {params['screen']['anchor_circle']['radius']}px")
    print(
        f"         - Radio tolerancia: {params['screen']['anchor_circle']['tolerance_radius']}px"
    )
    print()

    # Crear pantallas
    print("[INIT] Creando pantallas...")
    ui_cfg = config.get("ui", {}) or {}
    try:
        from scripts.audio_cue import make_fixation_tick

        fixation_tick = make_fixation_tick(ui_cfg.get("fixation_tick"))
        if fixation_tick is not None:
            print("[INIT] Fixation tick activado")
    except Exception as e:
        print(f"[INIT] ⚠ fixation_tick no disponible: {e}")
        fixation_tick = None
    anchor_screen = AnchorScreen(params, eye_tracker, fixation_tick=fixation_tick)

    # Leer configuración del pincel desde params.yaml
    drawing_tablet_cfg = config.get("drawing_tablet", {}) or {}
    tablet_brush_cfg = drawing_tablet_cfg.get("brush", {}) or {}
    brush_size = tablet_brush_cfg.get("size", 5)  # Default: 5
    brush_color = tuple(
        tablet_brush_cfg.get("color", [255, 255, 0])
    )  # Default: amarillo

    # Modo de entrada de dibujo: 'mouse' | 'tablet' | 'both' (default both).
    # pygame trata ratón y stylus como el mismo dispositivo, así que el flag
    # solo ajusta UI (texto, cursor, tamaño de pincel por defecto). Override
    # opcional de tamaño de pincel por modo en drawing_tablet.{mode}.brush.size.
    drawing_input = (config.get("drawing_input") or "both").lower()
    mode_override = drawing_tablet_cfg.get(drawing_input, {}) or {}
    mode_brush = mode_override.get("brush") or {}
    if "size" in mode_brush:
        brush_size = mode_brush["size"]
    if "color" in mode_brush:
        brush_color = tuple(mode_brush["color"])
    instructions_text = (drawing_tablet_cfg.get("instructions") or {}).get("text")
    if instructions_text and instructions_text.strip().startswith("Dibuja el punto"):
        # Default config string is too generic; let mode pick the wording.
        instructions_text = None
    hide_cursor = bool(mode_override.get("hide_cursor", False))
    allow_empty = bool(
        mode_override.get("allow_empty", drawing_tablet_cfg.get("allow_empty", False))
    )

    # Cursor clipping (multi-monitor): mantener el puntero en un único monitor
    # mientras la pantalla de dibujo esté activa. Default: monitor primario.
    clip_cfg = drawing_tablet_cfg.get("cursor_clip") or {}
    mode_clip_cfg = mode_override.get("cursor_clip") or {}
    clip_enabled = bool(mode_clip_cfg.get("enabled", clip_cfg.get("enabled", True)))
    clip_monitor = mode_clip_cfg.get("monitor", clip_cfg.get("monitor", "primary"))
    cursor_clip_rect = None
    if clip_enabled:
        try:
            from scripts.cursor_clip import resolve_target_rect

            cursor_clip_rect = resolve_target_rect(clip_monitor)
        except Exception as e:
            print(f"[INIT] ⚠ cursor_clip no disponible: {e}")
            cursor_clip_rect = None

    print(
        f"[CONFIG] Pincel de dibujo: mode={drawing_input} tamaño={brush_size} "
        f"color={brush_color} hide_cursor={hide_cursor} cursor_clip={cursor_clip_rect}"
    )

    # Modo de respuesta: 'drawing' (DrawingTablet) o 'saccade' (SaccadeScreen).
    # Ambos cumplen la misma interfaz (reset / update(screen, events) -> (bool, payload) / close)
    # así que el resto del experimento las consume con el mismo nombre.
    response_mode = (config.get("response_mode") or "drawing").lower()
    mapping_method = (config.get("mapping_method") or "absolute").lower()
    print(f"[CONFIG] Modo de respuesta: {response_mode}")
    print(f"[CONFIG] Método de mapeo:   {mapping_method}")

    def _build_audio_cue(cue_cfg):
        if not cue_cfg or not cue_cfg.get("enabled", False):
            return None
        try:
            from scripts.audio_cue import from_config as _ac_from_config

            return _ac_from_config(cue_cfg)
        except Exception as e:
            print(f"[INIT] ⚠ audio_cue no disponible: {e}")
            return None

    if response_mode == "saccade":
        from scripts.saccade_screen import SaccadeScreen

        saccade_cfg = config.get("saccade", {}) or {}
        idt_cfg = saccade_cfg.get("idt", {}) or {}
        vel_cfg = saccade_cfg.get("velocity", {}) or {}
        on_failure = (saccade_cfg.get("on_failure") or "rerun_max_3").lower()
        if on_failure.startswith("rerun_max_"):
            try:
                max_attempts = int(on_failure.split("_")[-1])
            except (ValueError, IndexError):
                max_attempts = 3
        elif on_failure == "rerun":
            max_attempts = 999
        else:  # 'skip' or unknown
            max_attempts = 1

        response_screen = SaccadeScreen(
            screen_width=actual_width,
            screen_height=actual_height,
            anchor_xy=(actual_width // 2, actual_height // 2),
            eye_tracker=eye_tracker,
            capture_duration_ms=saccade_cfg.get("capture_duration_ms", 1500),
            extraction=saccade_cfg.get("extraction", "idt_first_fixation"),
            extractor_params={"idt": idt_cfg, "velocity": vel_cfg},
            min_response_distance_px=saccade_cfg.get("min_response_distance_px", 30.0),
            max_attempts=max_attempts,
            show_gaze_trace=saccade_cfg.get("show_gaze_trace", True),
            audio_cue=_build_audio_cue(saccade_cfg.get("audio_cue")),
            # Debug fallback: when input_mode is mouse, the tracker doesn't
            # update last_smooth_gaze outside is_looking_at_point — so let
            # SaccadeScreen poll pygame.mouse directly. Off in production
            # eye-tracker modes to avoid silent mouse contamination.
            allow_mouse_fallback=bool(
                saccade_cfg.get(
                    "allow_mouse_fallback",
                    input_mode in ("mouse", "wacom"),
                )
            ),
        )
        response_capture = SaccadeResponseCapture(response_screen)
    elif mapping_method == "forced_adjustment":
        from scripts.tablet import ForcedAdjustmentTablet

        fa_cfg = config.get("forced_adjustment", {})
        fa_min_deg = float(fa_cfg.get("anchor_min_dist_deg", 2.0))
        fa_max_deg = float(fa_cfg.get("anchor_max_dist_deg", 4.0))
        # ppd se calculará después del mapper; se pasa None por ahora y se
        # actualiza en configure_forced_adjustment_ppd() más adelante
        response_screen = ForcedAdjustmentTablet(
            actual_width, actual_height,
            brush_size=brush_size,
            brush_color=brush_color,
            min_dist_px=fa_min_deg * 60,   # estimación provisional (60 px/deg)
            max_dist_px=fa_max_deg * 60,
        )
        response_capture = DrawingResponseCapture(response_screen)
    else:
        response_screen = DrawingTablet(
            actual_width,
            actual_height,
            brush_size,
            brush_color,
            mode=drawing_input,
            instructions_text=instructions_text,
            hide_cursor=hide_cursor,
            cursor_clip_rect=cursor_clip_rect,
            allow_empty=allow_empty,
        )
        response_capture = DrawingResponseCapture(response_screen)

    _set_active_response_screen(response_capture)

    # Gaze trace overlay
    gaze_trace_config = config.get("eye_tracker", {}).get("gaze_trace", {})
    gaze_trace_enabled = gaze_trace_config.get("enabled", True)
    gaze_trace_duration = gaze_trace_config.get(
        "trace_duration_ms", GAZE_TRACE_DURATION_MS
    )
    gaze_filter_name = gaze_trace_config.get("filter", "ema")
    gaze_filter_params = gaze_trace_config.get(gaze_filter_name, {})
    if gaze_trace_enabled:
        gaze_trace = GazeTrace(
            trace_duration_ms=gaze_trace_duration,
            filter_name=gaze_filter_name,
            filter_params=gaze_filter_params,
        )
        print(
            f"        Gaze trace: ON ({gaze_trace_duration}ms, filter={gaze_filter_name}, params={gaze_filter_params})"
        )
    else:
        gaze_trace = None
        print("        Gaze trace: OFF")
    print("        Pantallas creadas")

    # ============================================
    # INICIALIZAR MAPEO RETINOTÓPICO CON DYNAPHOS
    # ============================================
    print("[INIT] Inicializando mapeo retinotópico (Dynaphos)...")

    # Cargar configuración de electrodos
    electrode_config = load_active_electrodes_config()
    array_type = electrode_config["array_type"]
    coord_file = electrode_config["coordinate_files"][array_type]
    dropout = electrode_config.get("dropout", 0.0)  # Obtener dropout del params.yaml

    # Crear mapper con dropout
    screen_cfg = config.get("screen", {})

    # Convenio de FOV (pipeline): vf_scope_deg es el semiancho que se mapea al
    # lado MENOR de la pantalla (ppd = min(W,H)/(2*vf_scope_deg), isotrópico).
    # Para que los grados sean FÍSICAMENTE reales, vf_scope_deg debe igualar el
    # half-FOV físico del lado corto a la distancia de visionado. Pon
    # `vf_scope_deg: auto` y se deriva solo de la geometría + dist_to_screen_cm
    # (no puede desfasarse); con un número, se avisa si es incoherente.
    def _as_float_or_none(v):
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    _vf_raw = screen_cfg.get("vf_scope_deg")
    _vf_auto = _vf_raw is None or (
        isinstance(_vf_raw, str)
        and _vf_raw.strip().lower() in {"auto", "physical", "screen", "max"}
    )
    _phys_vf = None
    try:
        from scripts.screen_detect import coherent_vf_scope_deg

        _phys_vf = coherent_vf_scope_deg(
            screen_cfg.get("width"),
            screen_cfg.get("height"),
            screen_cfg.get("screen_diagonal_inches"),
            screen_cfg.get("dist_to_screen_cm"),
        )
    except Exception:
        _phys_vf = None

    _dist = screen_cfg.get("dist_to_screen_cm")
    if _vf_auto:
        if _phys_vf is not None:
            vf_scope_deg = _phys_vf
            print(
                f"[INIT] vf_scope_deg=auto → {vf_scope_deg:g}° "
                f"(half-FOV físico del lado corto a {_dist}cm)"
            )
        else:
            vf_scope_deg = 15.0
            print(
                "[INIT] ⚠ vf_scope_deg=auto pero falta geometría física "
                "(width/height/screen_diagonal_inches/dist_to_screen_cm); usando 15°"
            )
    else:
        vf_scope_deg = _as_float_or_none(_vf_raw)
        if vf_scope_deg is not None:
            vf_scope_deg = abs(vf_scope_deg)
        if vf_scope_deg is None:  # Default histórico: 30° total => 15
            vf_scope_deg = 15.0
        if _phys_vf and abs(vf_scope_deg - _phys_vf) / _phys_vf > 0.05:
            print(
                f"[INIT] ⚠ vf_scope_deg={vf_scope_deg:g}° no coincide con el half-FOV "
                f"físico del lado corto a {_dist}cm (~{_phys_vf:g}°). "
                f"Pon `vf_scope_deg: auto` para calibrarlo automáticamente."
            )

    # fuente de coordenadas
    coords_source = electrode_config.get("coordinate_source", "dynaphos_yaml")
    coords_csv_path = electrode_config.get("coords_csv_path", "")
    implant_id_filter = electrode_config.get("implant_id_filter", "all")
    if coords_source == "phoslab_csv" and coords_csv_path:
        coord_file_to_use = coords_csv_path
        print(f"[COORDS] Usando CSV de phoslab: {coords_csv_path}")
    else:
        coord_file_to_use = coord_file
        print(f"[COORDS] Usando coordenadas de Dynaphos: {coord_file}")

    # Error simulado conocido (bias + ruido) inyectado en la posición MOSTRADA del
    # fosfeno, para que el bayesiano tenga un sesgo real que aprender. La verdad
    # del CSV (implant_explorer) se conserva como `pred`. Ver
    # retinotopic_mapping.simulated_display_error en params.yaml.
    sim_err_cfg = electrode_config.get("simulated_display_error", {}) or {}
    display_bias_deg = sim_err_cfg.get("bias_deg", [0.0, 0.0])
    display_noise_std_deg = sim_err_cfg.get("noise_std_deg", 0.0)
    display_noise_seed = sim_err_cfg.get("noise_seed", None)
    display_error_enabled = bool(sim_err_cfg.get("enabled", False))

    mapper = DynaphosMapper(
        electrode_coords_file=coord_file_to_use,
        screen_width=actual_width,
        screen_height=actual_height,
        dropout=dropout,  # Pasar dropout a Dynaphos
        screen_diagonal_inches=screen_cfg.get("screen_diagonal_inches"),
        dist_to_screen_cm=screen_cfg.get("dist_to_screen_cm"),
        vf_scope_deg=vf_scope_deg,
        implant_id_filter=implant_id_filter,
        display_bias_deg=display_bias_deg,
        display_noise_std_deg=display_noise_std_deg,
        display_noise_seed=display_noise_seed,
        display_error_enabled=display_error_enabled,
    )

    # Actualizar ppd en ForcedAdjustmentTablet ahora que vf_scope_deg es conocido
    if mapping_method == "forced_adjustment":
        _fa_ppd = min(actual_width, actual_height) / (2.0 * vf_scope_deg)
        response_screen._ppd = _fa_ppd
        response_screen._screen_cx = actual_width // 2
        response_screen._screen_cy = actual_height // 2
        fa_cfg = config.get("forced_adjustment", {})
        fa_min_deg = float(fa_cfg.get("anchor_min_dist_deg", 2.0))
        fa_max_deg = float(fa_cfg.get("anchor_max_dist_deg", 4.0))
        response_screen.min_dist_px = fa_min_deg * _fa_ppd
        response_screen.max_dist_px = fa_max_deg * _fa_ppd
        print(
            f"[ForcedAdjustment] ppd={_fa_ppd:.1f} px/deg  "
            f"anchor_range=[{response_screen.min_dist_px:.0f}, "
            f"{response_screen.max_dist_px:.0f}] px  "
            f"({fa_min_deg}°–{fa_max_deg}°)"
        )

    # MAPPING DEBUG MODE: rejilla + marcador de fosfeno. Se construye desde la
    # geometría del mapper (mismo px/grado y centro que las posiciones de los
    # fosfenos) para que la rejilla quede alineada con los marcadores.
    global _MAPPING_DEBUG_OVERLAY
    _MAPPING_DEBUG_OVERLAY = MappingDebugOverlay.from_config_and_mapper(
        config, mapper, (actual_width, actual_height)
    )
    if _MAPPING_DEBUG_OVERLAY is not None:
        print(
            "[INIT] 🔧 MAPPING DEBUG MODE activo "
            f"(anillos cada {_MAPPING_DEBUG_OVERLAY.ring_step_deg:g}°)"
        )

    print(f"       ✓ Mapper inicializado")
    display_metadata = mapper.get_display_metadata()
    print()

    # ============================================
    # RESOLVER CORRIENTES (ahora ya sabemos total_electrodes)
    # ============================================
    # total_electrodes = tamaño COMPLETO del array (antes de filtrar por selection)
    # Se usa para resolver corrientes mapping y como referencia global.

    total_electrodes = len(mapper.active_electrodes)
    default_uA_sparse = float(stim_config.get("default_current_uA", 0.0))

    STIMULATION_CURRENTS_UA = _resolve_currents_uA(
        STIMULATION_CURRENTS_CFG,
        total_electrodes=total_electrodes,
        default_uA_fallback=90.0,
    )
    # STANDARD_UA se resolverá de nuevo tras configure_electrodes_from_selection
    # para que total_electrodes refleje solo los electrodos activos del selection.
    STIMULATION_CURRENTS_STANDARD_UA = _resolve_currents_uA(
        STIMULATION_CURRENTS_STANDARD_CFG,
        total_electrodes=total_electrodes,
        default_uA_fallback=90.0,
    )

    # Si el usuario usa dict sparse sin especificar default en el dict,
    # permitimos setearlo globalmente desde stimulation.default_current_uA.
    if isinstance(STIMULATION_CURRENTS_CFG, dict) and default_uA_sparse != 0.0:
        # Re-resolver para aplicar default global si el dict no lo trae.
        # (Si el dict sí lo trae, _parse_sparse_currents_mapping ya manda.)
        if (
            "default_uA" not in STIMULATION_CURRENTS_CFG
            and "default" not in STIMULATION_CURRENTS_CFG
            and "_default" not in STIMULATION_CURRENTS_CFG
        ):
            cfg = {
                "default_uA": default_uA_sparse,
                "overrides": STIMULATION_CURRENTS_CFG.get(
                    "overrides", STIMULATION_CURRENTS_CFG
                ),
            }
            STIMULATION_CURRENTS_UA = _resolve_currents_uA(
                cfg, total_electrodes=total_electrodes
            )

    if isinstance(STIMULATION_CURRENTS_STANDARD_CFG, dict) and default_uA_sparse != 0.0:
        if (
            "default_uA" not in STIMULATION_CURRENTS_STANDARD_CFG
            and "default" not in STIMULATION_CURRENTS_STANDARD_CFG
            and "_default" not in STIMULATION_CURRENTS_STANDARD_CFG
        ):
            cfg = {
                "default_uA": default_uA_sparse,
                "overrides": STIMULATION_CURRENTS_STANDARD_CFG.get(
                    "overrides", STIMULATION_CURRENTS_STANDARD_CFG
                ),
            }
            STIMULATION_CURRENTS_STANDARD_UA = _resolve_currents_uA(
                cfg, total_electrodes=total_electrodes
            )

    # ============================================
    # CHECKEAR MODO DE EXPERIMENTO Y ARGUMENTOS CLI
    # ============================================
    # IMPORTANTE: Esto ANTES de configurar electrodos, para eligir qué configuración usar
    cli_electrode = args.electrode_index or args.electrode_index_flag
    cli_repetitions = args.num_repetitions or args.num_repetitions_flag

    if cli_electrode is not None:
        # Modo mapping activado por CLI
        experiment_mode = "mapping"
        print(f"[CLI] Modo mapping activado por argumentos de línea de comandos")
    else:
        # Usar modo desde params.yaml
        experiment_mode = config.get("experiment_mode", "standard")

    print(f"[MODO] Modo de experimento: {experiment_mode}")
    print()

    # ════════════════════════════════════════════════════════════════
    # Activar electrodos según modo (SEPARACIÓN COMPLETA)
    # ════════════════════════════════════════════════════════════════
    if experiment_mode == "mapping":
        # MODO MAPPING: Los electrodos son COMPLETAMENTE LIBRES
        # No hay restricción de electrode_selection
        mapping_config = config.get("phosphene_mapping", {})

        # Obtener electrodos de phosphene_mapping (soporta 'all' y 'range')
        mapping_electrode_indices = _resolve_mapping_electrode_indices(
            mapping_config=mapping_config,
            cli_electrode=cli_electrode,
            total_electrodes=len(mapper.active_electrodes),
            electrode_index_map=getattr(mapper, "_electrode_index_map", None),
        )

        # Activar solo los electrodos que se van a mapear
        mapper.set_active_electrodes(mapping_electrode_indices)
        print(
            f"[MAPPING] Electrodos a mapear: {mapping_electrode_indices} (LIBRES, sin restricción)"
        )

    else:
        # MODO STANDARD: Los electrodos son los de electrode_selection
        mapper.configure_electrodes_from_selection(
            electrode_config["electrode_selection"]
        )
        print(f"[STANDARD] Electrodos según electrode_selection configurados")

    # Obtener posiciones de fosfenos
    PHOSPHENE_POSITIONS = mapper.get_active_phosphene_positions()
    NUM_PHOSPHENES = len(PHOSPHENE_POSITIONS)

    print(f"       ✓ Fosfenos generados: {NUM_PHOSPHENES}")
    print()

    # Feasibility gate: ningún fosfeno seleccionado debe caer fuera de pantalla.
    # Aborta por defecto (override: screen.allow_offscreen: true).
    _assert_phosphenes_onscreen(
        mapper,
        actual_width,
        actual_height,
        vf_scope_deg,
        bool(screen_cfg.get("allow_offscreen", False)),
    )

    if experiment_mode == "mapping":
        # ============================================
        # MODO MAPEO: N REPETICIONES DE UNO O MÁS ELECTRODOS
        # ============================================
        # NOTA: Los electrodos ya fueron activados anteriormente con mapper.set_active_electrodes()
        # Aquí continuamos con la lógica del experimento usando los electrodos del mapping
        mapping_config = config.get("phosphene_mapping", {})
        electrode_indices = mapping_electrode_indices
        print(f"[MAPPING] Electrodos a mapear: {electrode_indices}")

        if cli_repetitions is not None:
            num_repetitions = cli_repetitions
            print(f"[CLI] Repeticiones desde CLI: {num_repetitions}")
        else:
            num_repetitions = mapping_config.get("num_repetitions", 5)
            print(f"[YAML] Repeticiones desde phosphene_mapping: {num_repetitions}")

        print()

        # Detectar si es mapeo múltiple o simple
        is_multi_electrode = len(electrode_indices) > 1

        # ════════════════════════════════════════
        # MODO: ELECTRODOS (guardado consolidado)
        # ════════════════════════════════════════
        print("=" * 70)
        print("MODO MAPEO: ELECTRODOS (guardado consolidado)")
        print(f"Electrodos a mapear: {electrode_indices}")
        print(f"Repeticiones por electrodo: {num_repetitions}")
        print("=" * 70)
        print()

        # Carpeta del experimento — nueva o reanudada
        resumed_session_meta = None
        if resume_dir is not None:
            if not resume_dir.exists():
                print(f"✗ ERROR: --resume {resume_dir} no existe")
                return
            session_meta_path = resume_dir / "session_metadata.json"
            if not session_meta_path.exists():
                print(
                    f"✗ ERROR: {session_meta_path} no encontrado — no se puede reanudar"
                )
                return
            with open(session_meta_path, encoding="utf-8") as f:
                resumed_session = SessionMetadata.from_dict(json.load(f))
            resumed_session_meta = resumed_session.to_dict()  # dict view used below
            multi_experiment_dir = resume_dir
            print(f"[RESUME] Reanudando sesión: {multi_experiment_dir}")
            print(
                f"[RESUME] schema_version={resumed_session.schema_version} "
                f"trials totales en sesión original: {resumed_session.summary.get('n')}"
            )
            print()
        else:
            experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if SAVE_RESULTS:
                multi_experiment_dir = (
                    Path("mapping_experiments")
                    / f"mapping_mapeo_multiples_electrodo_{experiment_timestamp}"
                )
                multi_experiment_dir.mkdir(parents=True, exist_ok=True)
                print(f"Carpeta de experimento: {multi_experiment_dir}\n")
            else:
                multi_experiment_dir = None
                print("⚠️  Modo sin guardado: los datos NO se guardarán en disco\n")

        # ============================================
        # PRE-VALIDACIÓN: Construir recursos por electrodo
        # ============================================
        completed_electrodes = []
        exp_by_electrode = {}
        stim_by_electrode = {}
        position_by_electrode = {}
        current_by_electrode = {}
        valid_electrode_indices = []

        for electrode_num, electrode_index in enumerate(electrode_indices, 1):
            if electrode_index < 0 or electrode_index >= len(mapper.active_electrodes):
                print(f"✗ ERROR: Índice de electrodo fuera de rango: {electrode_index}")
                continue
            if not mapper.active_electrodes[electrode_index]:
                print(f"✗ ERROR: El electrodo {electrode_index} no está activo")
                continue
            try:
                phosphene_position = mapper.get_phosphene_position(electrode_index)
            except ValueError:
                print(
                    f"[SKIP] Electrodo {electrode_index}: posición no disponible en el CSV"
                )
                continue

            current_uA = _select_current_uA(
                STIMULATION_CURRENTS_UA,
                electrode_index=electrode_index,
                order_index=electrode_num - 1,
                total_electrodes=total_electrodes,
                default_current_uA=stim_config.get("default_current_uA", 90.0),
            )

            try:
                electrode_info = mapper.get_electrode_info(electrode_index)
            except Exception:
                electrode_info = {"index": int(electrode_index)}

            mapping_experiment = PhospheneMappingExperiment(
                params=params,
                screen=screen,
                clock=clock,
                eye_tracker=eye_tracker,
                anchor_screen=anchor_screen,
                drawing_tablet=response_capture,
                webcam_viewer=webcam_viewer,
                gaze_trace=gaze_trace,
                display_info=display_metadata,
                apriltag_overlay=_APRILTAG_OVERLAY,
                debug_overlay=_MAPPING_DEBUG_OVERLAY,
                input_mode=input_mode,
                mapping_method=mapping_method,
                timing_config={
                    "prestimulation_ms": PRESTIMULATION_MS,
                    "stimulation_ms": STIMULATION_MS,
                    "poststimulation_ms": POSTSTIMULATION_MS,
                    "interstimulation_ms": INTERSTIMULATION_MS,
                },
                electrode_index=electrode_index,
                electrode_info=electrode_info,
                num_repetitions=num_repetitions,
                experiment_name=f"mapeo_electrodo_{electrode_index}",
                experiment_dir=multi_experiment_dir,
            )
            stimulation_screen = StimulationScreen(
                params,
                eye_tracker,
                phosphene_position=phosphene_position,
                current_uA=current_uA,
                pulse_width_us=PULSE_WIDTH_US,
                frequency_hz=FREQUENCY_HZ,
            )
            stimulation_screen.dynaphos_mapper = mapper
            stimulation_screen.active_electrode_index = electrode_index

            exp_by_electrode[electrode_index] = mapping_experiment
            stim_by_electrode[electrode_index] = stimulation_screen
            position_by_electrode[electrode_index] = phosphene_position
            current_by_electrode[electrode_index] = current_uA
            valid_electrode_indices.append(electrode_index)

        if not valid_electrode_indices:
            print("✗ ERROR: Ningún electrodo válido para mapear.")
            cleanup_and_exit(eye_tracker, webcam_viewer)
            return

        # ============================================
        # CONSTRUIR TRIAL LIST (randomizado, con catch & practice)
        # ============================================
        import secrets as _secrets
        import random as _rng_mod

        if resumed_session_meta is not None:
            tsc = resumed_session.trial_sequence_config
            realized_seed = int(tsc.random_seed)
            catch_rate = float(tsc.catch_trial_rate)
            do_randomize = bool(tsc.randomize)
            no_repeat = bool(tsc.no_immediate_repeat)
            num_practice = int(tsc.num_practice_trials)
            isi_jitter_ms = float(tsc.isi_jitter_ms)
            print(
                f"[RESUME] Usando seed original = {realized_seed} (orden idéntico al de la sesión original)"
            )
        else:
            seed_cfg = mapping_config.get("random_seed")
            realized_seed = (
                int(seed_cfg) if seed_cfg is not None else _secrets.randbits(32)
            )
            catch_rate = float(mapping_config.get("catch_trial_rate", 0.0))
            do_randomize = bool(mapping_config.get("randomize", True))
            no_repeat = bool(mapping_config.get("no_immediate_repeat", True))
            num_practice = int(mapping_config.get("num_practice_trials", 0))
            isi_jitter_ms = float(mapping_config.get("isi_jitter_ms", 0.0))

        trials = build_trial_list(
            valid_electrode_indices,
            num_repetitions,
            seed=realized_seed,
            catch_trial_rate=catch_rate,
            no_immediate_repeat=no_repeat,
            randomize=do_randomize,
            num_practice_trials=num_practice,
        )
        ts = trial_summary(trials)
        print("=" * 70)
        print(f"TRIAL SEQUENCE: {ts}  seed={realized_seed}")
        print(
            f"  randomize={do_randomize}  catch_rate={catch_rate}  practice={num_practice}"
        )
        print(f"  isi_jitter_ms={isi_jitter_ms}")
        print("=" * 70)
        print()

        if SAVE_RESULTS and multi_experiment_dir is not None:
            session_record = SessionMetadata(
                session_started=datetime.now().isoformat(),
                valid_electrode_indices=valid_electrode_indices,
                num_repetitions=num_repetitions,
                mapping_method=mapping_method,
                coords_csv=Path(coords_csv_path).name if coords_csv_path else "",
                trial_sequence_config=TrialSequenceConfig(
                    randomize=do_randomize,
                    random_seed=realized_seed,
                    catch_trial_rate=catch_rate,
                    no_immediate_repeat=no_repeat,
                    num_practice_trials=num_practice,
                    isi_jitter_ms=isi_jitter_ms,
                ),
                summary=ts,
                trial_order=[t.to_dict() for t in trials],
            )
            with open(
                multi_experiment_dir / "session_metadata.json", "w", encoding="utf-8"
            ) as f:
                json.dump(session_record.to_dict(), f, indent=2, ensure_ascii=False)
            print(
                f"✓ Trial order guardado: {multi_experiment_dir / 'session_metadata.json'}"
            )
            print()

        # ============================================
        # EJECUTAR TRIAL LIST (interleaved, jittered ISI)
        # ============================================
        # Resume: descubrir qué trial_idx ya tienen artefactos guardados, para
        # saltarlos. Leemos cada electrode_<idx>/metadata.json y extraemos los
        # trial_idx ya completados.
        completed_trial_idx: set[int] = set()
        if resumed_session_meta is not None and multi_experiment_dir is not None:
            for d in multi_experiment_dir.glob("electrode_*"):
                meta_file = d / "metadata.json"
                if not meta_file.exists():
                    continue
                try:
                    with open(meta_file, encoding="utf-8") as f:
                        emeta = json.load(f)
                    for rep in emeta.get("repetitions", []):
                        tid = rep.get("trial_idx")
                        if isinstance(tid, int):
                            completed_trial_idx.add(tid)
                except Exception as e:
                    print(f"[RESUME] ⚠ No se pudo leer {meta_file}: {e}")
            print(f"[RESUME] Saltando {len(completed_trial_idx)} trials ya completados")

        isi_rng = _rng_mod.Random(realized_seed + 1)
        user_cancelled = False
        # Mini-dashboard: tasa de éxito y catch-response acumuladas.
        dash_real_total = 0
        dash_real_ok = 0
        dash_catch_total = 0
        dash_catch_response = 0

        for trial_pos, trial in enumerate(trials):
            if trial.trial_idx in completed_trial_idx:
                continue
            if trial.is_catch:
                # Stand-in: cualquier electrodo válido sirve (no se muestra fosfeno)
                stand_in = valid_electrode_indices[0]
                exp = exp_by_electrode[stand_in]
                stim = stim_by_electrode[stand_in]
                ph_pos = position_by_electrode[stand_in]
                cur = 0.0
                rep_for_call = trial_pos + 1  # solo label
            else:
                exp = exp_by_electrode[trial.electrode_index]
                stim = stim_by_electrode[trial.electrode_index]
                ph_pos = position_by_electrode[trial.electrode_index]
                cur = current_by_electrode[trial.electrode_index]
                rep_for_call = trial.rep_num if trial.rep_num > 0 else (trial_pos + 1)

            rep_metadata = exp.run_single_repetition(
                repetition_number=rep_for_call,
                stimulation_screen=stim,
                phosphene_position=ph_pos,
                current_uA=cur,
                pulse_width_us=PULSE_WIDTH_US,
                frequency_hz=FREQUENCY_HZ,
                run_prestim_func=run_prestimulation,
                run_stim_func=run_stimulation,
                run_poststim_func=run_poststimulation,
                run_interstim_func=run_interstimulation,
                check_quit_func=check_quit_events,
                drawing_tablet_reset_func=drawing_tablet_reset,
                FPS=FPS,
                trial_idx=trial.trial_idx,
                is_catch=trial.is_catch,
                is_practice=trial.is_practice,
                run_interstim_after=False,
            )

            if rep_metadata is None:
                print("\n[INFO] Experimento cancelado por el usuario")
                user_cancelled = True
                break

            # Mini-dashboard (experimenter visibility, §4.2):
            # imprimir un summary 1-línea cada trial.
            rep_status = (rep_metadata.get("response_status") or "ok").lower()
            if trial.is_catch:
                dash_catch_total += 1
                if rep_status == "ok":
                    dash_catch_response += 1
            elif not trial.is_practice:
                dash_real_total += 1
                if rep_status == "ok":
                    dash_real_ok += 1
            real_rate = (
                f"{dash_real_ok}/{dash_real_total} ({100.0*dash_real_ok/max(1,dash_real_total):.0f}%)"
                if dash_real_total
                else "0/0"
            )
            catch_rate_str = (
                f"{dash_catch_response}/{dash_catch_total} ({100.0*dash_catch_response/max(1,dash_catch_total):.0f}%)"
                if dash_catch_total
                else "0/0"
            )
            print(
                f"      [DASH] trial {trial_pos+1}/{len(trials)} "
                f"real_ok={real_rate}  catch_response={catch_rate_str}  "
                f"status={rep_status}"
            )

            # Inter-trial break con jitter, salvo en el último trial
            if trial_pos < len(trials) - 1:
                next_trial = trials[trial_pos + 1]
                next_electrode = (
                    next_trial.electrode_index or valid_electrode_indices[0]
                )
                next_exp = exp_by_electrode[next_electrode]
                jitter = (
                    isi_rng.uniform(-isi_jitter_ms, isi_jitter_ms)
                    if isi_jitter_ms > 0
                    else 0.0
                )
                duration_ms = max(0.0, INTERSTIMULATION_MS + jitter)
                if not next_exp._run_interstimulation_mapping(
                    trial_pos + 1, len(trials), duration_ms=duration_ms
                ):
                    print("\n[INFO] Cancelado durante interstim")
                    user_cancelled = True
                    break

        if user_cancelled:
            for electrode_index in valid_electrode_indices:
                if SAVE_RESULTS:
                    exp_by_electrode[electrode_index].finalize()
            cleanup_and_exit(eye_tracker, webcam_viewer)
            return

        # Finalizar todos los experimentos (un metadata.json por electrodo)
        for electrode_index in valid_electrode_indices:
            if SAVE_RESULTS:
                exp_by_electrode[electrode_index].finalize()
            completed_electrodes.append(electrode_index)
        print()

        # ════════════════════════════════════════
        # ANÁLISIS CONSOLIDADO DE TODOS LOS ELECTRODOS
        # ════════════════════════════════════════
        print("\n" + "=" * 70)
        print("ANÁLISIS CONSOLIDADO")
        print("=" * 70 + "\n")

        # Ya no necesitamos eye tracker ni webcam viewer
        if eye_tracker:
            eye_tracker.release()
        if webcam_viewer:
            webcam_viewer.release()

        consolidated_analysis_failed = False
        if SAVE_RESULTS:
            try:
                from scripts.multi_electrode_analyzer import MultiElectrodeAnalyzer

                # Crear analizador consolidado
                multi_analyzer = MultiElectrodeAnalyzer(multi_experiment_dir)

                # Analizar todos los electrodos
                consolidated_results = multi_analyzer.analyze_all_electrodes()

                if consolidated_results:
                    # Generar visualización consolidada
                    multi_analyzer.visualize_consolidated_map(consolidated_results)

                    # Generar reporte
                    multi_analyzer.create_summary_report(consolidated_results)

                    print("\n" + "=" * 70)
                    print("MAPEO CONSOLIDADO COMPLETADO")
                    print("=" * 70)
                    print(f"\n📁 Resultados guardados en: {multi_experiment_dir}")
                    print(f"   └─ consolidated_analysis/ (análisis integrado)")

                else:
                    print("⚠ No se pudieron generar resultados consolidados")
                    consolidated_analysis_failed = True

            except Exception as e:
                print(f"✗ ERROR en análisis consolidado: {e}")
                consolidated_analysis_failed = True
                import traceback

                traceback.print_exc()
        else:
            print("\n⚠️  Modo sin guardado: análisis consolidado omitido")

        # ════════════════════════════════════════
        # PANTALLA DE FINALIZACIÓN
        # ════════════════════════════════════════
        print("\n" + "=" * 70)
        print("EXPERIMENTO COMPLETADO - Esperando confirmación del usuario...")
        print("=" * 70 + "\n")

        show_experiment_completion_screen(
            screen=screen,
            clock=clock,
            screen_width=SCREEN_WIDTH,
            screen_height=SCREEN_HEIGHT,
        )

        # Mensaje final
        if SAVE_RESULTS:
            print("\n📁 Resultados guardados en:")
            print(f"   {multi_experiment_dir}")
        else:
            print("\n⚠️  Experimento completado sin guardar datos en disco")
        print()
        print("Presiona cualquier tecla para salir...")
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                    waiting = False
            clock.tick(10)

        pygame.quit()
        if consolidated_analysis_failed:
            print("\n[ERROR] Programa finalizado con errores en el análisis consolidado")
            sys.exit(1)
        print("\n[INFO] Programa finalizado")
        return

    # ============================================
    # MODO STANDARD: TODOS LOS ELECTRODOS UNA VEZ
    # ============================================

    # Crear carpeta del experimento con timestamp único
    experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path("logs") / f"experiment_{experiment_timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    print(f"       ✓ Carpeta de experimento: {experiment_dir}")

    # Metadata del experimento completo
    active_electrode_indices = np.where(mapper.active_electrodes)[0].tolist()
    active_electrodes_info = []
    for electrode_index in active_electrode_indices:
        try:
            active_electrodes_info.append(mapper.get_electrode_info(electrode_index))
        except Exception:
            active_electrodes_info.append({"index": int(electrode_index)})

    experiment_metadata = {
        "experiment_id": experiment_timestamp,
        "start_time": datetime.now().isoformat(),
        "num_phosphenes": NUM_PHOSPHENES,  # Número de fosfenos activados (de Dynaphos)
        "phosphene_positions": PHOSPHENE_POSITIONS,  # Posiciones calculadas por Dynaphos
        "display": display_metadata,
        "input_mode": input_mode,
        "timing": {
            "prestimulation_ms": PRESTIMULATION_MS,
            "stimulation_ms": STIMULATION_MS,
            "poststimulation_ms": POSTSTIMULATION_MS,
            "interstimulation_ms": INTERSTIMULATION_MS,
        },
        "stimulation_parameters": {
            "pulse_width_us": PULSE_WIDTH_US,
            "frequency_hz": FREQUENCY_HZ,
            "currents_uA": STIMULATION_CURRENTS_STANDARD_UA,
        },
        "electrode_indices": active_electrode_indices,  # Índices de electrodos activos
        "electrodes": active_electrodes_info,  # Incluye coords en córtex/visual por electrodo
        "array_type": array_type,  # Tipo de array usado
        "phosphenes": [],  # Se llenará con datos de cada punto
    }

    print()
    print("=" * 70)
    print("INICIANDO EXPERIMENTO STANDARD")
    print(f"Número de fosfenos: {NUM_PHOSPHENES}")
    print("=" * 70)
    print()

    # ============================================
    # BUCLE PRINCIPAL: 5 PUNTOS BRILLANTES
    # ============================================
    for phosphene_index in range(NUM_PHOSPHENES):
        # IMPORTANTE:
        # - phosphene_number es 0-based (se alinea con índices de arrays como currents)
        # - phosphene_display_number es 1-based (solo para mostrar al usuario)
        phosphene_number = phosphene_index
        phosphene_display_number = phosphene_number + 1
        phosphene_position = PHOSPHENE_POSITIONS[phosphene_index]
        electrode_index = active_electrode_indices[phosphene_index]

        # Obtener corriente para este electrodo (MODO STANDARD)
        # Regla de coherencia:
        # - Si el vector tiene longitud == total_electrodes: corriente por índice de electrodo.
        # - Si no: corriente por orden de presentación (phosphene_index).
        current_uA = _select_current_uA(
            STIMULATION_CURRENTS_STANDARD_UA,
            electrode_index=electrode_index,
            order_index=phosphene_index,
            total_electrodes=total_electrodes,
            default_current_uA=stim_config.get("default_current_uA", 90.0),
        )

        print("=" * 70)
        print(f"PUNTO BRILLANTE {phosphene_display_number}/{NUM_PHOSPHENES}")
        print(f"Posición: {phosphene_position}")
        print(f"Corriente: {current_uA} µA")
        print("=" * 70)
        print()

        # Crear stimulation_screen con la posición específica y parámetros de estimulación
        stimulation_screen = StimulationScreen(
            params,
            eye_tracker,
            phosphene_position=phosphene_position,
            current_uA=current_uA,
            pulse_width_us=PULSE_WIDTH_US,
            frequency_hz=FREQUENCY_HZ,
        )
        stimulation_screen.dynaphos_mapper = mapper
        stimulation_screen.active_electrode_index = electrode_index

        # Metadata de este punto específico
        try:
            electrode_info = mapper.get_electrode_info(electrode_index)
        except Exception:
            electrode_info = {"index": int(electrode_index)}

        phosphene_metadata = {
            "phosphene_number": phosphene_number,
            "position": phosphene_position,
            "electrode_index": electrode_index,
            "electrode_info": electrode_info,
            "stimulation_parameters": {
                "current_uA": current_uA,
                "pulse_width_us": PULSE_WIDTH_US,
                "frequency_hz": FREQUENCY_HZ,
            },
            "start_time": datetime.now().isoformat(),
            "events": {},
            "fixation_losses": 0,
            "gaze_tracking": {
                "prestim": [],
                "stim": [],
                "poststim": [],
            },
        }

        # ============================================
        # ESTADOS 1-3: PRESTIM → STIM → POSTSTIM
        # Si se pierde la fijación en stim o poststim, se reintenta
        # desde prestimulation. Solo ESC/QUIT aborta el experimento.
        # ============================================
        trial_attempt = 0
        phase_completed = False

        while not phase_completed:
            trial_attempt += 1
            if trial_attempt > 1:
                print(
                    f"\n      [RETRY] Intento #{trial_attempt} para punto {phosphene_display_number}"
                )

            # ESTADO 1: PRESTIMULATION
            gaze_trace.clear()
            success = run_prestimulation(
                screen,
                clock,
                anchor_screen,
                eye_tracker,
                phosphene_metadata,
                webcam_viewer,
                gaze_trace,
            )
            if not success:
                # Only False on ESC/QUIT/timeout
                cleanup_and_exit(eye_tracker, webcam_viewer)
                return

            # ESTADO 2: STIMULATION
            success = run_stimulation(
                screen,
                clock,
                stimulation_screen,
                eye_tracker,
                phosphene_metadata,
                webcam_viewer,
                gaze_trace,
            )
            if success is None:
                # None = fixation lost → retry from prestim
                print(f"      [RETRY] Volviendo a prestimulation...")
                continue
            if not success:
                # False = ESC/QUIT
                cleanup_and_exit(eye_tracker, webcam_viewer)
                return

            # ESTADO 3: POSTSTIMULATION
            success = run_poststimulation(
                screen,
                clock,
                anchor_screen,
                eye_tracker,
                phosphene_metadata,
                webcam_viewer,
                gaze_trace,
            )
            if success is None:
                # None = fixation lost → retry from prestim
                print(f"      [RETRY] Volviendo a prestimulation...")
                continue
            if not success:
                # False = ESC/QUIT
                cleanup_and_exit(eye_tracker, webcam_viewer)
                return

            # All 3 phases passed
            phase_completed = True

        phosphene_metadata["trial_attempts"] = trial_attempt

        # ============================================
        # ESTADO 4: DRAWING
        # ============================================
        print(f"[4/4] DRAWING: Dibuja el punto {phosphene_display_number}")
        phosphene_metadata["drawing_start"] = datetime.now().isoformat()

        # Resetear pantalla de respuesta (DrawingTablet o SaccadeScreen)
        drawing_tablet_reset(response_capture)

        drawing_completed = False

        while not drawing_completed:
            events = pygame.event.get()

            finished = response_capture.update(screen, events)

            if finished:
                print(
                    f"      ✓ {phosphene_display_number} completado "
                    f"({response_capture.mode}, status={response_capture.last_status})"
                )
                drawing_completed = True

            # Comprobar ESC
            for event in events:
                if event.type == pygame.QUIT:
                    cleanup_and_exit(eye_tracker, webcam_viewer)
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    cleanup_and_exit(eye_tracker, webcam_viewer)
                    return

            _display_flip(screen)
            clock.tick(FPS)  # Cómo de fluido se ejecuta el programa

        phosphene_metadata["drawing_end"] = datetime.now().isoformat()

        # ============================================
        # GUARDADO
        # ============================================
        print(f"      [GUARDANDO] Punto {phosphene_display_number}...")

        response_result = response_capture.save_result(
            experiment_dir,
            drawing_filename=f"drawing_{phosphene_display_number}.png",
            saccade_filename=f"saccade_samples_{phosphene_display_number}.json",
        )
        apply_response_metadata(phosphene_metadata, response_result)
        print(f"        ✓ Respuesta: {response_result.response_file}")

        phosphene_metadata["end_time"] = datetime.now().isoformat()

        # Añadir metadata de este punto al experimento
        experiment_metadata["phosphenes"].append(phosphene_metadata)
        print(f"        ✓ Metadata guardada")

        # ============================================
        # INTERSTIMULATION (solo si NO es el último punto)
        # ============================================
        if phosphene_display_number < NUM_PHOSPHENES:
            print()
            print(
                f"      [BREAK] Descanso antes del punto {phosphene_display_number + 1}..."
            )

            phosphene_metadata["interstim_start"] = datetime.now().isoformat()

            success = run_interstimulation(
                screen,
                clock,
                phosphene_display_number,
                NUM_PHOSPHENES,
                webcam_viewer,
            )
            if not success:
                cleanup_and_exit(eye_tracker, webcam_viewer)
                return

            phosphene_metadata["interstim_end"] = datetime.now().isoformat()
        else:
            print()
            print("      [FIN] Último punto completado - No hay break")

        print()

    # Ya no necesitamos eye tracker ni webcam viewer
    _close_response_screen(response_capture)
    if eye_tracker:
        eye_tracker.release()
    if webcam_viewer:
        webcam_viewer.release()

    # ============================================
    # GUARDAR METADATA DEL EXPERIMENTO COMPLETO
    # ============================================
    print("=" * 70)
    print("GUARDANDO METADATA DEL EXPERIMENTO")
    print("=" * 70)

    experiment_metadata["end_time"] = datetime.now().isoformat()

    # Guardar como JSON
    json_filename = experiment_dir / "metadata.json"
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(experiment_metadata, f, indent=2, ensure_ascii=False)
    print(f"✓ Metadata JSON: {json_filename.name}")

    # Guardar también como TXT legible
    txt_filename = experiment_dir / "metadata.txt"
    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("EXPERIMENTO DE PRÓTESIS CORTICAL VISUAL\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"ID Experimento: {experiment_metadata['experiment_id']}\n")
        f.write(f"Inicio: {experiment_metadata['start_time']}\n")
        f.write(f"Fin: {experiment_metadata['end_time']}\n")
        f.write(f"Número de puntos: {experiment_metadata['num_phosphenes']}\n\n")

        f.write("TIEMPOS:\n")
        f.write(f"  - Prestimulation: {PRESTIMULATION_MS}ms\n")
        f.write(f"  - Stimulation: {STIMULATION_MS}ms\n")
        f.write(f"  - Poststimulation: {POSTSTIMULATION_MS}ms\n")
        f.write(f"  - Interstimulation: {INTERSTIMULATION_MS}ms\n\n")

        f.write("=" * 70 + "\n")
        f.write("DETALLES DE CADA PUNTO BRILLANTE\n")
        f.write("=" * 70 + "\n\n")

        for phos in experiment_metadata["phosphenes"]:
            f.write(f"PUNTO {phos['phosphene_number']}:\n")
            f.write(f"  Posición: {phos['position']}\n")
            f.write(f"  Inicio: {phos['start_time']}\n")
            f.write(f"  Fin: {phos['end_time']}\n")
            f.write(f"  Pérdidas de fijación: {phos['fixation_losses']}\n")
            write_response_summary(f, phos)

            if "events" in phos and "prestim_start" in phos["events"]:
                f.write(f"  Prestim inicio: {phos['events']['prestim_start']}\n")
            if "events" in phos and "stim_start" in phos["events"]:
                f.write(f"  Stim inicio: {phos['events']['stim_start']}\n")
            if "events" in phos and "poststim_start" in phos["events"]:
                f.write(f"  Poststim inicio: {phos['events']['poststim_start']}\n")

            if "interstim_start" in phos:
                f.write(f"  Interstim inicio: {phos['interstim_start']}\n")
                f.write(f"  Interstim fin: {phos['interstim_end']}\n")

            f.write("\n")

    print(f"✓ Metadata TXT: {txt_filename.name}")

    print()
    print("=" * 70)
    print("EXPERIMENTO COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    print()
    print(f"📁 Carpeta: {experiment_dir}")
    print(f"📷 Dibujos: {NUM_PHOSPHENES} archivos PNG")
    print(f"📄 Metadata: JSON + TXT")
    print()

    # ============================================
    # ANÁLISIS AUTOMÁTICO MODO STANDARD
    # ============================================
    print("\n" + "=" * 70)
    print("ANÁLISIS AUTOMÁTICO DE RESULTADOS (STANDARD)")
    print("=" * 70)

    try:
        from scripts.standard_analyzer import StandardExperimentAnalyzer

        analyzer = StandardExperimentAnalyzer(experiment_dir)
        analyzer.analyze_all_electrodes()
    except Exception as e:
        print(f" Error en análisis standard: {e}")
        import traceback

        traceback.print_exc()

    # Mostrar pantalla de finalización
    show_experiment_completion_screen(
        screen=screen,
        clock=clock,
        screen_width=SCREEN_WIDTH,
        screen_height=SCREEN_HEIGHT,
    )

    pygame.quit()
    print("\n[INFO] Programa finalizado")


# ============================================
# FUNCIONES PARA CADA ESTADO
# ============================================


def show_electrode_transition_screen(
    screen, clock, current_electrode, next_electrode, screen_width, screen_height
):
    """
    Muestra una pantalla de transición entre electrodos

    Args:
        screen: Superficie de pygame
        clock: Reloj de pygame
        current_electrode: Índice del electrodo actual
        next_electrode: Índice del siguiente electrodo
        screen_width: Ancho de la pantalla
        screen_height: Alto de la pantalla
    """
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    LIGHT_BLUE = (100, 200, 255)

    font_large = pygame.font.Font(None, 96)
    font_medium = pygame.font.Font(None, 64)
    font_small = pygame.font.Font(None, 48)

    waiting = True
    while waiting:
        # Limpiar pantalla
        screen.fill(BLACK)

        # Título
        title_text = "Electrode Transition"
        title_surface = font_large.render(title_text, True, LIGHT_BLUE)
        title_rect = title_surface.get_rect(
            center=(screen_width // 2, screen_height // 4)
        )
        screen.blit(title_surface, title_rect)

        # Información de transición
        transition_text = f"Completed: Electrode {current_electrode:03d}"
        transition_surface = font_medium.render(transition_text, True, WHITE)
        transition_rect = transition_surface.get_rect(
            center=(screen_width // 2, screen_height // 2 - 60)
        )
        screen.blit(transition_surface, transition_rect)

        # Siguiente
        next_text = f"Next: Electrode {next_electrode:03d}"
        next_surface = font_medium.render(next_text, True, LIGHT_BLUE)
        next_rect = next_surface.get_rect(
            center=(screen_width // 2, screen_height // 2 + 20)
        )
        screen.blit(next_surface, next_rect)

        # Instrucción
        instr_text = "Press SPACE to continue with next electrode"
        instr_surface = font_small.render(instr_text, True, WHITE)
        instr_rect = instr_surface.get_rect(
            center=(screen_width // 2, screen_height - 120)
        )
        screen.blit(instr_surface, instr_rect)

        # Instrucción de escape
        escape_text = "Press ESC to exit"
        escape_surface = font_small.render(escape_text, True, (200, 100, 100))
        escape_rect = escape_surface.get_rect(
            center=(screen_width // 2, screen_height - 50)
        )
        screen.blit(escape_surface, escape_rect)

        # Eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False
                return False  # Salir
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    waiting = False
                    return True  # Continuar al siguiente
                if event.key == pygame.K_ESCAPE:
                    waiting = False
                    return False  # Salir

        _display_flip(screen)
        clock.tick(30)


def show_experiment_completion_screen(screen, clock, screen_width, screen_height):
    """
    Muestra la pantalla final de completación del experimento

    Args:
        screen: Superficie de pygame
        clock: Reloj de pygame
        screen_width: Ancho de la pantalla
        screen_height: Alto de la pantalla
    """
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GREEN = (100, 255, 100)

    font_large = pygame.font.Font(None, 124)
    font_medium = pygame.font.Font(None, 64)
    font_small = pygame.font.Font(None, 48)

    waiting = True
    while waiting:
        # Limpiar pantalla
        screen.fill(BLACK)

        # Título principal
        title_text = "COMPLETADO"
        title_surface = font_large.render(title_text, True, GREEN)
        title_rect = title_surface.get_rect(
            center=(screen_width // 2, screen_height // 3)
        )
        screen.blit(title_surface, title_rect)

        # Mensaje
        msg_text = "El mapeo de fosfenos ha finalizado correctamente"
        msg_surface = font_medium.render(msg_text, True, WHITE)
        msg_rect = msg_surface.get_rect(
            center=(screen_width // 2, screen_height // 2 - 40)
        )
        screen.blit(msg_surface, msg_rect)

        # Instrucción
        instr_text = "Presiona ENTER para cerrar"
        instr_surface = font_small.render(instr_text, True, WHITE)
        instr_rect = instr_surface.get_rect(
            center=(screen_width // 2, screen_height - 120)
        )
        screen.blit(instr_surface, instr_rect)

        # Eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False
                return True  # Cerrar
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                    waiting = False
                    return True  # Cerrar
                if event.key == pygame.K_ESCAPE:
                    waiting = False
                    return True  # Cerrar

        _display_flip(screen)
        clock.tick(30)


def run_prestimulation(
    screen,
    clock,
    anchor_screen,
    eye_tracker,
    phosphene_metadata,
    webcam_viewer=None,
    gaze_trace=None,
):
    """
    Ejecuta la fase de prestimulation

    Returns:
        bool: True si completó exitosamente, False si hubo error
    """
    # Fixation gate (verificado §1.3): el stim NO se dispara hasta acumular
    # PRESTIMULATION_MS de fijación CONTINUA. Si la mirada se pierde,
    # `looking_start_time` se resetea a None y el contador empieza de cero.
    # MAX_FIXATION_WAIT_MS es solo un presupuesto total para evitar trials
    # eternos cuando el participante no puede fijar — no un fallback que
    # deje pasar el stim sin fijación.
    print("[1/4] PRESTIMULATION: Esperando fijación...")

    # Nuevo trial: limpiar el marcador del trial anterior (la rejilla de debug
    # permanece; el marcador reaparece al disparar el siguiente fosfeno).
    if _MAPPING_DEBUG_OVERLAY is not None:
        _MAPPING_DEBUG_OVERLAY.clear()
    print(
        f"      (target=center, tolerance={anchor_screen.tolerance_radius}px, needed={PRESTIMULATION_MS}ms)"
    )

    # Edge-trigger the fixation tick afresh for this trial
    if hasattr(anchor_screen, "reset_fixation_edge"):
        anchor_screen.reset_fixation_edge()

    looking_start_time = None
    wait_start_time = time.time()
    frame_count = 0

    while True:
        frame_count += 1
        # Actualizar visualizador de webcam
        if webcam_viewer is not None:
            if not webcam_viewer.update():
                print("      ⚠ Ventana de webcam cerrada")

        # Timeout
        elapsed_wait = (time.time() - wait_start_time) * 1000
        if elapsed_wait > MAX_FIXATION_WAIT_MS:
            print(f"      ✗ TIMEOUT: No miró en {MAX_FIXATION_WAIT_MS}ms")
            return False

        # Detectar mirada
        eye_frame = eye_tracker.get_frame() if eye_tracker else None
        is_looking = anchor_screen.update(screen, eye_frame)

        # Update and draw gaze trace
        if gaze_trace and eye_tracker:
            gaze_trace.update(eye_tracker.last_raw_gaze)
            gaze_trace.draw(screen)

        # Registrar gaze coordinates durante prestimulation
        if (
            eye_tracker
            and hasattr(eye_tracker, "last_raw_gaze")
            and eye_tracker.last_raw_gaze
        ):
            elapsed_ms = (time.time() - wait_start_time) * 1000
            gaze_x, gaze_y = eye_tracker.last_raw_gaze
            phosphene_metadata["gaze_tracking"]["prestim"].append(
                {
                    "time_ms": int(elapsed_ms),
                    "gaze_x": int(gaze_x),
                    "gaze_y": int(gaze_y),
                }
            )

        if is_looking:
            if looking_start_time is None:
                looking_start_time = time.time()
                phosphene_metadata["events"][
                    "prestim_start"
                ] = datetime.now().isoformat()

            elapsed_ms = (time.time() - looking_start_time) * 1000
            if elapsed_ms >= PRESTIMULATION_MS:
                print(
                    f"      ✓ Prestimulation completado ({elapsed_ms:.0f}ms, {frame_count} frames)"
                )
                return True
        else:
            if looking_start_time is not None:
                phosphene_metadata["fixation_losses"] += 1
                print(
                    f"      [PRESTIM] Fixation lost at frame {frame_count} (loss #{phosphene_metadata['fixation_losses']})"
                )
            looking_start_time = None

        if check_quit_events():
            return False

        _display_flip(screen)
        clock.tick(FPS)  # Cómo de fluido se ejecuta el programa


def run_stimulation(
    screen,
    clock,
    stimulation_screen,
    eye_tracker,
    phosphene_metadata,
    webcam_viewer=None,
    gaze_trace=None,
):
    """
    Ejecuta la fase de stimulation

    Returns:
        True: completed successfully
        None: fixation lost (retryable)
        False: user quit (ESC/close)
    """
    print("[2/4] STIMULATION: Mostrando punto brillante...")
    print(
        f"      (tolerance={stimulation_screen.tolerance_radius}px, duration={STIMULATION_MS}ms)"
    )

    # ⭐ ACTIVAR el punto brillante (solo aquí, después de prestimulation exitosa)
    stimulation_screen.set_show_phosphene(True)
    # Debug: fijar el marcador en la posición del fosfeno (persiste hasta el
    # próximo prestim, cubriendo poststim y la fase de respuesta).
    _set_mapping_debug_marker(stimulation_screen)

    stim_start_time = time.time()
    phosphene_metadata["events"]["stim_start"] = datetime.now().isoformat()
    stim_frame = 0

    while True:
        stim_frame += 1
        # Actualizar visualizador de webcam
        if webcam_viewer is not None:
            if not webcam_viewer.update():
                print("      ⚠ Ventana de webcam cerrada")

        eye_frame = eye_tracker.get_frame() if eye_tracker else None
        is_looking = stimulation_screen.update(screen, eye_frame)

        # Update and draw gaze trace
        if gaze_trace and eye_tracker:
            gaze_trace.update(eye_tracker.last_raw_gaze)
            gaze_trace.draw(screen)

        # Registrar gaze coordinates durante stimulation
        if (
            eye_tracker
            and hasattr(eye_tracker, "last_raw_gaze")
            and eye_tracker.last_raw_gaze
        ):
            elapsed_ms = (time.time() - stim_start_time) * 1000
            gaze_x, gaze_y = eye_tracker.last_raw_gaze
            phosphene_metadata["gaze_tracking"]["stim"].append(
                {
                    "time_ms": int(elapsed_ms),
                    "gaze_x": int(gaze_x),
                    "gaze_y": int(gaze_y),
                }
            )

        if not is_looking and eye_tracker is not None:
            elapsed_ms = (time.time() - stim_start_time) * 1000
            gaze_info = ""
            if hasattr(eye_tracker, "last_raw_gaze") and eye_tracker.last_raw_gaze:
                rx, ry = eye_tracker.last_raw_gaze
                gaze_info = f" gaze=({rx:.0f},{ry:.0f})"
            print(
                f"      ✗ FIXATION LOST at frame {stim_frame} ({elapsed_ms:.0f}ms into stim){gaze_info}"
            )
            phosphene_metadata["fixation_losses"] += 1
            # ⭐ DESACTIVAR el punto antes de reintentar
            stimulation_screen.set_show_phosphene(False)
            return None  # retryable

        elapsed_ms = (time.time() - stim_start_time) * 1000
        if elapsed_ms >= STIMULATION_MS:
            print(f"      ✓ Stimulation completado ({elapsed_ms:.0f}ms)")
            # ⭐ DESACTIVAR el punto al terminar
            stimulation_screen.set_show_phosphene(False)
            return True

        if check_quit_events():
            stimulation_screen.set_show_phosphene(False)
            return False

        _display_flip(screen)
        clock.tick(FPS)  # Cómo de fluido se ejecuta el programa


def run_poststimulation(
    screen,
    clock,
    anchor_screen,
    eye_tracker,
    phosphene_metadata,
    webcam_viewer=None,
    gaze_trace=None,
):
    """
    Ejecuta la fase de poststimulation

    Returns:
        True: completed successfully
        None: fixation lost (retryable)
        False: user quit (ESC/close)
    """
    print("[3/4] POSTSTIMULATION: Verificando fijación...")
    print(
        f"      (tolerance={anchor_screen.tolerance_radius}px, duration={POSTSTIMULATION_MS}ms)"
    )

    post_start_time = time.time()
    phosphene_metadata["events"]["poststim_start"] = datetime.now().isoformat()
    post_frame = 0

    while True:
        post_frame += 1
        # Actualizar visualizador de webcam
        if webcam_viewer is not None:
            if not webcam_viewer.update():
                print("      ⚠ Ventana de webcam cerrada")

        eye_frame = eye_tracker.get_frame() if eye_tracker else None
        is_looking = anchor_screen.update(screen, eye_frame)

        # Update and draw gaze trace
        if gaze_trace and eye_tracker:
            gaze_trace.update(eye_tracker.last_raw_gaze)
            gaze_trace.draw(screen)

        # Registrar gaze coordinates durante poststimulation
        if (
            eye_tracker
            and hasattr(eye_tracker, "last_raw_gaze")
            and eye_tracker.last_raw_gaze
        ):
            elapsed_ms = (time.time() - post_start_time) * 1000
            gaze_x, gaze_y = eye_tracker.last_raw_gaze
            phosphene_metadata["gaze_tracking"]["poststim"].append(
                {
                    "time_ms": int(elapsed_ms),
                    "gaze_x": int(gaze_x),
                    "gaze_y": int(gaze_y),
                }
            )

        if not is_looking and eye_tracker is not None:
            elapsed_ms = (time.time() - post_start_time) * 1000
            gaze_info = ""
            if hasattr(eye_tracker, "last_raw_gaze") and eye_tracker.last_raw_gaze:
                rx, ry = eye_tracker.last_raw_gaze
                gaze_info = f" gaze=({rx:.0f},{ry:.0f})"
            print(
                f"      ✗ FIXATION LOST at frame {post_frame} ({elapsed_ms:.0f}ms into poststim){gaze_info}"
            )
            phosphene_metadata["fixation_losses"] += 1
            return None  # retryable

        elapsed_ms = (time.time() - post_start_time) * 1000
        if elapsed_ms >= POSTSTIMULATION_MS:
            print(f"      ✓ Poststimulation completado ({elapsed_ms:.0f}ms)")
            return True

        if check_quit_events():
            return False

        _display_flip(screen)
        clock.tick(FPS)  # Cómo de fluido se ejecuta el programa


def run_interstimulation(
    screen, clock, current_point, total_points, webcam_viewer=None
):
    """
    Ejecuta la fase de interstimulation (break entre puntos)
    La persona puede mirar donde quiera

    Args:
        screen: Superficie de pygame
        clock: Reloj de pygame
        current_point: Número del punto que acaba de terminar
        total_points: Número total de puntos
        webcam_viewer: Visualizador de webcam (opcional)

    Returns:
        bool: True si completó exitosamente, False si usuario quiere salir
    """
    print(f"      → Duración: {INTERSTIMULATION_MS}ms")
    print(f"      → Puede mirar donde quiera, parpadear, relajar los ojos")
    print(f"      → (Presiona ESPACIO para saltar el descanso)")

    inter_start_time = time.time()
    # Texto de ISI desactivado por defecto (§6.3 del rigor plan): un contador
    # visible entrena al participante a anticipar el siguiente estímulo.
    # Solo se renderiza si debug.show_interstim_text está activo en params.yaml,
    # y entonces se muestra pequeño y abajo, fuera del centro.
    show_text = bool(DEBUG_SHOW_INTERSTIM_TEXT)
    font_small = pygame.font.Font(None, 22) if show_text else None

    while True:
        elapsed_ms = (time.time() - inter_start_time) * 1000

        # Actualizar visualizador de webcam
        if webcam_viewer is not None:
            if not webcam_viewer.update():
                print("      ⚠ Ventana de webcam cerrada")

        screen.fill((0, 0, 0))

        if show_text and font_small is not None:
            remaining_s = max(0.0, (INTERSTIMULATION_MS - elapsed_ms) / 1000)
            dbg = f"[debug] ISI {current_point}/{total_points} · {remaining_s:.1f}s"
            dbg_surface = font_small.render(dbg, True, (120, 120, 120))
            dbg_rect = dbg_surface.get_rect(
                center=(screen.get_width() // 2, screen.get_height() - 24)
            )
            screen.blit(dbg_surface, dbg_rect)

        # Comprobar tiempo transcurrido
        if elapsed_ms >= INTERSTIMULATION_MS:
            print(f"      ✓ Break completado ({elapsed_ms:.0f}ms)")
            return True

        # Eventos del usuario
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    # Permitir saltar el descanso con ESPACIO
                    print(f"      ⏩ Break saltado por el usuario")
                    return True

        _display_flip(screen)
        clock.tick(FPS)  # Cómo de fluido se ejecuta el programa


def drawing_tablet_reset(tablet):
    """
    Resetea la pantalla de respuesta (DrawingTablet o SaccadeScreen) para un
    nuevo trial. Usa la API pública .reset() para mantener la independencia
    de la implementación concreta.
    """
    tablet.reset()


# ============================================
# FUNCIONES AUXILIARES
# ============================================


def check_quit_events():
    """Verifica si el usuario quiere salir"""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return True
    return False


def cleanup_and_exit(eye_tracker, webcam_viewer=None, response_screen=None):
    """Libera recursos y cierra el programa"""
    print("\n[CLEANUP] Liberando recursos...")
    _close_response_screen(response_screen)
    if eye_tracker:
        eye_tracker.release()
    if webcam_viewer:
        webcam_viewer.release()
    pygame.quit()
    sys.exit(0)


# ============================================
# PUNTO DE ENTRADA
# ============================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Programa interrumpido (Ctrl+C)")
        _close_response_screen()
        pygame.quit()
        sys.exit(0)
    except Exception as e:
        print(f"\n\n[ERROR] Error inesperado: {e}")
        import traceback

        traceback.print_exc()
        _close_response_screen()
        pygame.quit()
        sys.exit(1)
