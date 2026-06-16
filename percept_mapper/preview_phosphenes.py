"""
Preview de Fosfenos - Visualiza el mapeo de electrodos antes del experimento

Muestra blobs gaussianos en la posición de cada electrodo, anillos de
iso-excentricidad y ejes X/Y, para verificar la disposición del array.

Modos de entrada (excluyentes):
  --csv <ruta>   carga coordenadas directamente desde un CSV de phosLab
  (ninguno)      carga desde la configuración YAML (comportamiento original)

Uso:
    python preview_phosphenes.py
    python preview_phosphenes.py --csv config/synthetic_4ecc_4el.csv
    python preview_phosphenes.py --csv config/mi.csv --sigma 0.5

Controles:
    ESC / Q   Salir
    S         Guardar captura PNG
    N         Mostrar/ocultar números de electrodo
    G         Mostrar/ocultar anillos de iso-excentricidad
    B         Alternar entre blob gaussiano y punto simple
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import pygame
import numpy as np
import yaml

from scripts.dynaphos_adapter import DynaphosMapper, load_active_electrodes_config


class _PreviewMapper(DynaphosMapper):
    """DynaphosMapper con _get_valid_electrode_indices corregido para preview."""

    def _get_valid_electrode_indices(self):
        px = self.electrode_positions_visual_px
        n = min(self.num_electrodes, len(px))
        return [
            int(i) for i in range(n) if np.isfinite(px[i][0]) and np.isfinite(px[i][1])
        ]


# ---------------------------------------------------------------------------
# Geometría de pantalla
# ---------------------------------------------------------------------------


def _compute_vf_scope(cfg_screen: dict, actual_width: int, actual_height: int) -> float:
    raw = cfg_screen.get("vf_scope_deg", "auto")
    if str(raw).strip().lower() != "auto":
        return float(raw)
    diag_in = float(cfg_screen.get("screen_diagonal_inches", 15.33))
    dist_cm = float(cfg_screen.get("dist_to_screen_cm", 60))
    diag_px = math.sqrt(actual_width**2 + actual_height**2)
    ppi = diag_px / diag_in
    half_h_cm = (actual_height / ppi) * 2.54 / 2.0
    return math.degrees(math.atan(half_h_cm / dist_cm))


# ---------------------------------------------------------------------------
# Carga desde CSV
# ---------------------------------------------------------------------------


def load_from_csv(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def csv_to_pixel_positions(
    rows: list[dict],
    center: tuple[int, int],
    px_per_deg: float,
) -> list[tuple[int, int]]:
    positions = []
    for r in rows:
        cx = center[0] + float(r["x_deg"]) * px_per_deg
        cy = center[1] - float(r["y_deg"]) * px_per_deg
        positions.append((int(round(cx)), int(round(cy))))
    return positions


# ---------------------------------------------------------------------------
# Renderizado de blob gaussiano  ← COLORES ACTUALIZADOS A FUCSIA
# ---------------------------------------------------------------------------


def make_gaussian_overlay(
    width: int,
    height: int,
    positions_px: list[tuple[int, int]],
    sigma_px: float,
) -> pygame.Surface:
    """Genera una Surface RGBA con la suma de gaussianos en fucsia."""
    arr = np.zeros((height, width), dtype=np.float32)
    r = int(sigma_px * 4)
    for cx, cy in positions_px:
        x0, x1 = max(0, cx - r), min(width, cx + r + 1)
        y0, y1 = max(0, cy - r), min(height, cy + r + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        xs = np.arange(x0, x1)
        ys = np.arange(y0, y1)
        XX, YY = np.meshgrid(xs, ys)
        g = np.exp(-((XX - cx) ** 2 + (YY - cy) ** 2) / (2 * sigma_px**2))
        arr[y0:y1, x0:x1] += g

    if arr.max() > 0:
        arr /= arr.max()

    # Colormap amarillo muy claro: blanco en periferia → amarillo pálido en el centro
    R = np.clip(arr * 255, 0, 255).astype(np.uint8)
    G = np.clip(arr * 245, 0, 255).astype(np.uint8)
    B = np.clip(arr * 120, 0, 255).astype(np.uint8)

    rgb = np.stack([R, G, B], axis=-1)
    surf = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
    surf = surf.convert_alpha()

    # Alpha con umbral: los bordes (arr<0.12) son completamente transparentes
    # para evitar el halo oscuro por mezcla de negro con el fondo blanco
    alpha_norm = np.clip((arr - 0.12) / 0.88, 0.0, 1.0)
    alpha = np.clip(alpha_norm * 255, 0, 255).astype(np.uint8)
    alpha_view = pygame.surfarray.pixels_alpha(surf)
    alpha_view[:] = alpha.T
    del alpha_view
    return surf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Preview de fosfenos con blobs gaussianos"
    )
    parser.add_argument(
        "--csv", default=None, help="Ruta a un CSV de phosLab (omitir → usa YAML)"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.35,
        help="Sigma del blob gaussiano en grados (default: 0.35)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("PREVIEW DE FOSFENOS - MAPEO RETINOTÓPICO")
    print("=" * 70)

    pygame.init()
    screen = pygame.display.set_mode((1536, 864), 0)
    pygame.display.set_caption("Preview de Fosfenos")
    clock = pygame.time.Clock()

    actual_width = screen.get_width()
    actual_height = screen.get_height()
    center = (actual_width // 2, actual_height // 2)

    config_path = (Path(__file__).resolve().parent / "config" / "params.yaml").resolve()
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg_screen = cfg.get("screen") or {}

    vf_scope_deg = _compute_vf_scope(cfg_screen, actual_width, actual_height)
    px_per_deg = min(actual_width, actual_height) / (2.0 * vf_scope_deg)

    print(
        f"[GEOM] Pantalla: {actual_width}×{actual_height}  |  vf_scope: ±{vf_scope_deg:.2f}°  |  {px_per_deg:.1f} px/°"
    )

    csv_rows = None
    active_indices = []
    phosphene_positions = []

    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            sys.exit(f"[ERROR] CSV no encontrado: {csv_path}")
        csv_rows = load_from_csv(csv_path)
        phosphene_positions = csv_to_pixel_positions(csv_rows, center, px_per_deg)
        active_indices = [int(r["electrode_index"]) for r in csv_rows]
        print(f"[CSV]  {csv_path.name}  |  {len(csv_rows)} electrodos")
    else:
        print("[YAML] Cargando configuración de Dynaphos...")
        electrode_config = load_active_electrodes_config()

        coords_source = electrode_config.get("coordinate_source", "dynaphos_yaml")
        coords_csv_path_str = electrode_config.get("coords_csv_path", "")

        if coords_source == "phoslab_csv" and coords_csv_path_str:
            csv_path = Path(coords_csv_path_str)
            if not csv_path.exists():
                sys.exit(f"[ERROR] CSV no encontrado: {csv_path}")
            csv_rows = load_from_csv(csv_path)
            phosphene_positions = csv_to_pixel_positions(csv_rows, center, px_per_deg)
            active_indices = [int(r["electrode_index"]) for r in csv_rows]
            print(f"[YAML→CSV] {csv_path.name}  |  {len(csv_rows)} electrodos")
        else:
            array_type = electrode_config["array_type"]
            coord_file = electrode_config["coordinate_files"][array_type]

            mapper = _PreviewMapper(
                electrode_coords_file=coord_file,
                screen_width=actual_width,
                screen_height=actual_height,
                vf_scope_deg=vf_scope_deg,
            )
            mapper.configure_electrodes_from_selection(
                electrode_config["electrode_selection"]
            )
            phosphene_positions = mapper.get_active_phosphene_positions()
            active_indices = list(np.where(mapper.active_electrodes)[0])
            print(f"[YAML] {len(phosphene_positions)} electrodos activos")

    sigma_px = args.sigma * px_per_deg

    gauss_surf = make_gaussian_overlay(
        actual_width, actual_height, phosphene_positions, sigma_px
    )

    max_ring_ecc = vf_scope_deg
    ring_step = 2.0 if vf_scope_deg <= 10 else 5.0
    ring_eccs = [
        e for e in np.arange(ring_step, max_ring_ecc + ring_step * 0.5, ring_step)
    ]

    print(
        f"\n{'IDX':<5}  {'Pos (px)':<22}  {'x_deg':>7}  {'y_deg':>7}  {'ecc_deg':>8}  {'pol_deg':>8}"
    )
    print("-" * 70)
    for i, pos in enumerate(phosphene_positions):
        idx = active_indices[i]
        ox = (pos[0] - center[0]) / px_per_deg
        oy = -(pos[1] - center[1]) / px_per_deg
        ecc = math.sqrt(ox**2 + oy**2)
        pol = math.degrees(math.atan2(oy, ox)) % 360
        pos_str = f"({pos[0]}, {pos[1]})"
        print(
            f"{idx:<5}  {pos_str:<22}  {ox:>7.3f}  {oy:>7.3f}  {ecc:>8.3f}  {pol:>8.1f}"
        )
    print()

    # ------------------------------------------------------------------
    # Paleta de colores  ← ÚNICOS CAMBIOS RESPECTO AL ORIGINAL
    # ------------------------------------------------------------------
    WHITE      = (255, 255, 255)
    BLACK      = (  0,   0,   0)  # negro (texto)
    RED        = (110,  80, 190)  # lila medio-oscuro (punto de fijación)
    GRAY       = (  0,   0,   0)  # negro (etiquetas meridianos)
    LIGHT_GRAY = (  0,   0,   0)  # negro (HUD)

    BG_COLOR   = (255, 255, 255)  # fondo blanco
    AXIS_COLOR = ( 90,  60, 175)  # lila oscuro — ejes
    RING_COLOR = (145, 115, 210)  # lila medio — anillos
    RING_LABEL = (  0,   0,   0)  # negro — etiquetas anillos
    DOT_COLOR  = (255, 245, 120)  # amarillo muy claro — fosfenos
    # ------------------------------------------------------------------

    font_large = pygame.font.Font(None, 40)
    font_small = pygame.font.Font(None, 28)
    font_tiny = pygame.font.Font(None, 22)

    show_numbers = True
    show_rings = True
    use_gaussian = True

    print(
        "CONTROLES:  ESC/Q=salir  S=captura  N=números  G=anillos  B=modo blob/punto\n"
    )

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_s:
                    ts = pygame.time.get_ticks()
                    fname = f"phosphene_preview_{ts}.png"
                    pygame.image.save(screen, fname)
                    print(f"[OK] Captura guardada: {fname}")
                elif event.key == pygame.K_n:
                    show_numbers = not show_numbers
                elif event.key == pygame.K_g:
                    show_rings = not show_rings
                elif event.key == pygame.K_b:
                    use_gaussian = not use_gaussian
                    print(f"[MODO] {'Gaussiano' if use_gaussian else 'Punto simple'}")

        # ---- Fondo blanco ----
        screen.fill(BG_COLOR)

        # ---- Ejes X / Y en azul clarito ----
        pygame.draw.line(
            screen, AXIS_COLOR, (0, center[1]), (actual_width, center[1]), 1
        )
        pygame.draw.line(
            screen, AXIS_COLOR, (center[0], 0), (center[0], actual_height), 1
        )

        # ---- Anillos de iso-excentricidad en amarillo ----
        if show_rings:
            for ecc in ring_eccs:
                r_px = int(ecc * px_per_deg)
                pygame.draw.circle(screen, RING_COLOR, center, r_px, 1)
                lx = center[0] + r_px + 4
                ly = center[1] - 16
                label = font_tiny.render(f"{ecc:.0f}°", True, RING_LABEL)
                screen.blit(label, (lx, ly))

        # ---- Blobs gaussianos (fucsia) o puntos simples ----
        if use_gaussian:
            screen.blit(gauss_surf, (0, 0))
        else:
            for pos in phosphene_positions:
                pygame.draw.circle(screen, DOT_COLOR, pos, 8, 0)
                pygame.draw.circle(screen, BLACK, pos, 8, 1)

        # ---- Punto de fijación central ----
        pygame.draw.circle(screen, RED, center, 6, 2)

        # ---- Números de electrodo ----
        if show_numbers:
            for i, pos in enumerate(phosphene_positions):
                idx = active_indices[i]
                lbl = font_tiny.render(str(idx), True, BLACK)
                screen.blit(lbl, (pos[0] + 10, pos[1] - 10))

        # ---- Etiquetas de meridianos ----
        for txt, px, py in [
            ("RHM", actual_width - 50, center[1] - 18),
            ("LHM", 10, center[1] - 18),
            ("UVM", center[0] + 6, 8),
            ("LVM", center[0] + 6, actual_height - 22),
        ]:
            screen.blit(font_tiny.render(txt, True, GRAY), (px, py))

        # ---- HUD ----
        hud_lines = [
            f"{'CSV:' if args.csv else 'YAML:'} {Path(args.csv).name if args.csv else 'config'}  "
            f"|  {len(phosphene_positions)} electrodos  |  ±{vf_scope_deg:.1f}°  |  {px_per_deg:.1f} px/°",
            f"N=números  G=anillos  B={'gauss→punto' if use_gaussian else 'punto→gauss'}  S=captura  ESC=salir",
        ]
        for i, line in enumerate(hud_lines):
            screen.blit(font_tiny.render(line, True, LIGHT_GRAY), (14, 14 + i * 20))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    print("[INFO] Preview cerrado")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrumpido (Ctrl+C)")
        pygame.quit()
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback

        traceback.print_exc()
        pygame.quit()
        sys.exit(1)
