"""Visualizador de resultados guardados (analysis_results.json).

Objetivo: convertir los JSON de análisis a salidas gráficas reproducibles (PNG)
SIN recalcular centroides (útil si ya existe analysis_results.json).

Uso:
  python scripts/analysis_results_visualizer.py <ruta>

Donde <ruta> puede ser:
  - una carpeta de electrodo: .../electrode_002
  - un archivo analysis_results.json
  - una carpeta de experimento con subcarpetas electrode_XXX

Salidas:
  - analysis_plot.png (si no existe o se regenera)
  - analysis_boxplots.png (si no existe o se regenera)

Nota: requiere metadata.json en cada electrodo para parámetros de pantalla y px/deg.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Permite ejecutar como: python scripts/analysis_results_visualizer.py ...
# (añade la raíz del repo al sys.path)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mapping_analyzer import PhospheneMappingAnalyzer


def _is_electrode_dir(path: Path) -> bool:
    return path.is_dir() and path.name.startswith("electrode_")


def _find_electrode_dirs(path: Path) -> list[Path]:
    """Devuelve una lista de carpetas electrode_XXX a partir de una ruta."""
    if path.is_file() and path.name == "analysis_results.json":
        path = path.parent

    if _is_electrode_dir(path):
        return [path]

    if path.is_dir():
        electrode_dirs = sorted(
            [
                d
                for d in path.iterdir()
                if d.is_dir() and d.name.startswith("electrode_")
            ]
        )
        if electrode_dirs:
            return electrode_dirs

    return []


def visualize_electrode_from_json(electrode_dir: Path, overwrite: bool = True) -> bool:
    results_file = electrode_dir / "analysis_results.json"
    if not results_file.exists():
        print(f"✗ No existe analysis_results.json en: {electrode_dir}")
        return False

    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    analyzer = PhospheneMappingAnalyzer(electrode_dir)

    plot_path = electrode_dir / "analysis_plot.png"
    boxplot_path = electrode_dir / "analysis_boxplots.png"

    if overwrite or not plot_path.exists():
        analyzer.visualize_results(results, output_file=plot_path.name)
    else:
        print(f"• Ya existe: {plot_path.name}")

    if overwrite or not boxplot_path.exists():
        analyzer.visualize_boxplots(results, output_file=boxplot_path.name)
    else:
        print(f"• Ya existe: {boxplot_path.name}")

    return True


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Uso: python scripts/analysis_results_visualizer.py <ruta>")
        print("Ejemplos:")
        print(
            "  python scripts/analysis_results_visualizer.py mapping_experiments/.../electrode_002"
        )
        print(
            "  python scripts/analysis_results_visualizer.py mapping_experiments/.../consolidated_analysis"
        )
        return 1

    path = Path(argv[1])
    overwrite = True
    if "--no-overwrite" in argv:
        overwrite = False

    electrode_dirs = _find_electrode_dirs(path)
    if not electrode_dirs:
        print(f"✗ No se encontraron carpetas electrode_XXX desde: {path}")
        return 2

    ok_count = 0
    for electrode_dir in electrode_dirs:
        print("\n" + "=" * 70)
        print(f"Visualizando: {electrode_dir}")
        print("=" * 70)
        try:
            if visualize_electrode_from_json(electrode_dir, overwrite=overwrite):
                ok_count += 1
        except Exception as e:
            print(f"✗ Error en {electrode_dir.name}: {e}")

    print(f"\n✓ Completado: {ok_count}/{len(electrode_dirs)} electrodo(s)")
    return 0 if ok_count == len(electrode_dirs) else 3


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
