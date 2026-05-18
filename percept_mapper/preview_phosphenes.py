"""
Preview de Fosfenos - Visualiza el mapeo de electrodos antes del experimento

Muestra en pantalla dónde aparecerán los puntos brillantes según la configuración
de Dynaphos, para verificar que el mapeo retinotópico es correcto.

Uso:
    python preview_phosphenes.py
"""

import pygame
import sys
import numpy as np
import yaml
from pathlib import Path
from scripts.dynaphos_adapter import (
    DynaphosMapper,
    load_active_electrodes_config,
)

# Configuración
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
FULLSCREEN = False  # Cambiar a True para fullscreen


def main():
    """
    Muestra preview del mapeo de fosfenos
    """
    print("=" * 70)
    print("PREVIEW DE FOSFENOS - MAPEO RETINOTÓPICO")
    print("=" * 70)
    print("\nVisualizando posiciones de electrodos activos...\n")

    # ============================================
    # INICIALIZACIÓN
    # ============================================
    pygame.init()

    flags = pygame.FULLSCREEN if FULLSCREEN else 0
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), flags)
    pygame.display.set_caption("Preview de Fosfenos")

    clock = pygame.time.Clock()

    # Tamaño real de pantalla
    actual_width = screen.get_width()
    actual_height = screen.get_height()
    center = (actual_width // 2, actual_height // 2)

    print(f"[INIT] Pantalla: {actual_width}x{actual_height}")

    # ============================================
    # CARGAR MAPEO DE DYNAPHOS
    # ============================================
    print("[INIT] Cargando configuración de Dynaphos...")

    # Leer vf_scope_deg (convenio único) desde params.yaml
    config_path = (Path(__file__).resolve().parent / "config" / "params.yaml").resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    vf_scope_deg = float((cfg.get("screen") or {}).get("vf_scope_deg", 15))

    electrode_config = load_active_electrodes_config()
    array_type = electrode_config["array_type"]
    coord_file = electrode_config["coordinate_files"][array_type]

    print(f"       Array de electrodos: {array_type}")

    # Crear mapper
    mapper = DynaphosMapper(
        electrode_coords_file=coord_file,
        screen_width=actual_width,
        screen_height=actual_height,
        vf_scope_deg=vf_scope_deg,
    )

    # Activar electrodos según configuración
    mapper.configure_electrodes_from_selection(electrode_config["electrode_selection"])

    # Obtener posiciones
    phosphene_positions = mapper.get_active_phosphene_positions()
    num_phosphenes = len(phosphene_positions)

    print(f"\n✓ Total de electrodos: {mapper.num_electrodes}")
    print(f"✓ Electrodos activos: {num_phosphenes}")
    print(f"✓ Posiciones calculadas\n")

    # ============================================
    # CALCULAR ESTADÍSTICAS
    # ============================================
    active_indices = np.where(mapper.active_electrodes)[0]

    stats = []
    for idx in active_indices:
        info = mapper.get_electrode_info(idx)
        stats.append(
            {
                "index": idx,
                "position": info["visual_position_px"],
                "eccentricity": info["eccentricity_deg"],
                "visual_deg": info["visual_position_deg"],
            }
        )

    print("=" * 70)
    print("ESTADÍSTICAS DE FOSFENOS")
    print("=" * 70)
    print(
        f"{'ID':<5} {'Posición (px)':<20} {'Grados visuales':<20} {'Excentricidad':<15}"
    )
    print("-" * 70)

    for s in stats:
        pos_str = f"({int(s['position'][0])}, {int(s['position'][1])})"
        deg_str = f"({s['visual_deg'][0]:.1f}°, {s['visual_deg'][1]:.1f}°)"
        ecc_str = f"{s['eccentricity']:.2f}°"
        print(f"{s['index']:<5} {pos_str:<20} {deg_str:<20} {ecc_str:<15}")

    print("=" * 70)
    print("\nCONTROLES:")
    print("  ESC o Q  - Salir")
    print("  S        - Guardar imagen")
    print("  N        - Mostrar/ocultar números de electrodo")
    print("  G        - Mostrar/ocultar grid de referencia")
    print("\nVisualizando...\n")

    # ============================================
    # CONFIGURACIÓN VISUAL
    # ============================================
    # Colores
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (100, 100, 255)
    GRAY = (80, 80, 80)
    LIGHT_GRAY = (150, 150, 150)

    # Tamaños
    PHOSPHENE_RADIUS = 20
    ANCHOR_RADIUS = 40
    GRID_SPACING = 100

    # Estados
    show_numbers = True
    show_grid = True

    # Fuentes
    font_large = pygame.font.Font(None, 36)
    font_small = pygame.font.Font(None, 24)
    font_tiny = pygame.font.Font(None, 18)

    # ============================================
    # BUCLE PRINCIPAL
    # ============================================
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_s:
                    # Guardar screenshot
                    timestamp = pygame.time.get_ticks()
                    filename = f"phosphene_preview_{timestamp}.png"
                    pygame.image.save(screen, filename)
                    print(f"✓ Imagen guardada: {filename}")
                elif event.key == pygame.K_n:
                    show_numbers = not show_numbers
                    print(f"Números: {'ON' if show_numbers else 'OFF'}")
                elif event.key == pygame.K_g:
                    show_grid = not show_grid
                    print(f"Grid: {'ON' if show_grid else 'OFF'}")

        # ============================================
        # DIBUJO
        # ============================================
        screen.fill(BLACK)

        # 1. Grid de referencia (opcional)
        if show_grid:
            # Líneas verticales
            for x in range(0, actual_width, GRID_SPACING):
                pygame.draw.line(screen, GRAY, (x, 0), (x, actual_height), 1)
            # Líneas horizontales
            for y in range(0, actual_height, GRID_SPACING):
                pygame.draw.line(screen, GRAY, (0, y), (actual_width, y), 1)

            # Ejes centrales en color diferente
            pygame.draw.line(
                screen, LIGHT_GRAY, (center[0], 0), (center[0], actual_height), 2
            )
            pygame.draw.line(
                screen, LIGHT_GRAY, (0, center[1]), (actual_width, center[1]), 2
            )

        # 2. Círculo de anclaje (centro)
        pygame.draw.circle(screen, RED, center, ANCHOR_RADIUS, 3)

        # Etiqueta del centro
        center_text = font_small.render("CENTRO (ANCLAJE)", True, RED)
        center_rect = center_text.get_rect(center=(center[0], center[1] - 60))
        screen.blit(center_text, center_rect)

        # 3. Fosfenos
        for idx, pos in enumerate(phosphene_positions):
            # Círculo del fosfeno
            pygame.draw.circle(screen, WHITE, pos, PHOSPHENE_RADIUS, 2)
            pygame.draw.circle(screen, BLUE, pos, PHOSPHENE_RADIUS - 3, 0)  # Relleno

            # Línea al centro
            pygame.draw.line(screen, GREEN, center, pos, 1)

            # Número del electrodo
            if show_numbers:
                electrode_idx = active_indices[idx]
                number_text = font_small.render(str(electrode_idx), True, WHITE)
                number_rect = number_text.get_rect(center=pos)
                screen.blit(number_text, number_rect)

                # Offset desde el centro (para debugging)
                offset_x = pos[0] - center[0]
                offset_y = pos[1] - center[1]
                offset_text = font_tiny.render(
                    f"({offset_x:+d}, {offset_y:+d})", True, LIGHT_GRAY
                )
                offset_rect = offset_text.get_rect(
                    center=(pos[0], pos[1] + PHOSPHENE_RADIUS + 15)
                )
                screen.blit(offset_text, offset_rect)

        # 4. Información en pantalla
        info_y = 20
        info_lines = [
            f"Array: {array_type}",
            f"Electrodos activos: {num_phosphenes}/{mapper.num_electrodes}",
            f"Campo visual: {mapper.fov_x_deg[1] - mapper.fov_x_deg[0]}° × {mapper.fov_y_deg[1] - mapper.fov_y_deg[0]}°",
            f"Escala: {mapper.pixels_per_degree_x:.1f} px/grado",
        ]

        for line in info_lines:
            text = font_small.render(line, True, WHITE)
            screen.blit(text, (20, info_y))
            info_y += 30

        # 5. Controles en la parte inferior
        controls_y = actual_height - 80
        control_lines = [
            "ESC/Q: Salir  |  S: Guardar imagen  |  N: Números  |  G: Grid"
        ]

        for line in control_lines:
            text = font_tiny.render(line, True, LIGHT_GRAY)
            screen.blit(text, (20, controls_y))
            controls_y += 25

        # ============================================
        # ACTUALIZAR
        # ============================================
        pygame.display.flip()
        clock.tick(30)

    # ============================================
    # LIMPIEZA
    # ============================================
    pygame.quit()
    print("\n[INFO] Preview cerrado")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Programa interrumpido (Ctrl+C)")
        pygame.quit()
        sys.exit(0)
    except Exception as e:
        print(f"\n\n[ERROR] Error inesperado: {e}")
        import traceback

        traceback.print_exc()
        pygame.quit()
        sys.exit(1)
