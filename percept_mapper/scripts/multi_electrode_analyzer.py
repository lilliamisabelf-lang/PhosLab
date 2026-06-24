"""
Analizador consolidado de múltiples electrodos

Combina los resultados del mapeo de fosfenos de múltiples electrodos
y genera un mapa visual integrando todos ellos.
"""

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

# Desactivar backend interactivo de matplotlib para evitar interferencias con Pygame
import matplotlib

matplotlib.use("Agg")  # Backend no interactivo - solo guarda archivos
import matplotlib.pyplot as plt

from scripts.mapping_analyzer import PhospheneMappingAnalyzer
from scripts.stats import distance_deg


class MultiElectrodeAnalyzer:
    """
    Analiza y consolida resultados de múltiples electrodos mapeados
    """

    def __init__(self, experiment_dir):
        """
        Inicializa el analizador multi-electrodo

        Args:
            experiment_dir: Ruta a la carpeta principal del experimento
                           (e.g., 'mapping_experiments/_multiples_...')
        """
        self.experiment_dir = Path(experiment_dir)

        if not self.experiment_dir.exists():
            raise FileNotFoundError(
                f"No se encontró la carpeta del experimento: {self.experiment_dir}"
            )

        # Crear carpeta de análisis consolidado
        self.consolidated_dir = self.experiment_dir / "consolidated_analysis"
        self.consolidated_dir.mkdir(parents=True, exist_ok=True)

        # Geometría px↔grados. Estos son solo VALORES POR DEFECTO; la geometría
        # real (resolución, centro y px/grado ISOTRÓPICO) se adopta en
        # analyze_all_electrodes desde los metadatos del primer electrodo
        # (ver _adopt_geometry), evitando estos hardcodeados que asumían
        # 1920×1080. Convenio pipeline: vf_scope_deg = semiancho (max ecc).
        self.screen_width = 1920
        self.screen_height = 1080
        self.screen_center = (960, 540)
        self.vf_scope_deg = 15
        # Escala ISOTRÓPICA anclada al lado menor (mismo criterio que el mapper):
        # un grado ocupa los mismos píxeles en X e Y (los píxeles son cuadrados).
        ppd = float(min(self.screen_width, self.screen_height)) / (2.0 * self.vf_scope_deg)
        self.pixels_per_degree_x = ppd
        self.pixels_per_degree_y = ppd
        self.fov_x_deg = [-(self.screen_width / 2.0) / ppd, (self.screen_width / 2.0) / ppd]
        self.fov_y_deg = [-(self.screen_height / 2.0) / ppd, (self.screen_height / 2.0) / ppd]
        self.fov_width_deg = self.fov_x_deg[1] - self.fov_x_deg[0]
        self.fov_height_deg = self.fov_y_deg[1] - self.fov_y_deg[0]
        self._geometry_locked = False

        print(f"[MultiElectrodeAnalyzer] Experimento: {self.experiment_dir.name}")
        print(f"                         Carpeta consolidada: {self.consolidated_dir}")

    def _adopt_geometry(self, analyzer):
        """Adopta la geometría real (resolución, centro, px/grado isotrópico) del
        analizador de un electrodo, leída de sus metadatos de sesión. Se invoca
        una sola vez (primer electrodo válido) para que las distancias
        consolidadas y el mapa usen la escala correcta de la sesión en lugar de
        los valores por defecto hardcodeados."""
        if self._geometry_locked:
            return
        try:
            self.pixels_per_degree_x = float(analyzer.pixels_per_degree_x)
            self.pixels_per_degree_y = float(analyzer.pixels_per_degree_y)
            self.screen_width = int(analyzer.screen_width)
            self.screen_height = int(analyzer.screen_height)
            self.screen_center = tuple(analyzer.screen_center)
        except (AttributeError, TypeError, ValueError):
            return
        self._geometry_locked = True
        print(
            f"                         Geometría adoptada de metadatos: "
            f"{self.pixels_per_degree_x:.2f} px/° (X), "
            f"{self.pixels_per_degree_y:.2f} px/° (Y), "
            f"{self.screen_width}×{self.screen_height}"
        )

    def analyze_all_electrodes(self):
        """
        Analiza todos los electrodos presentes en el experimento

        Returns:
            dict: Diccionario con resultados consolidados de todos los electrodos
        """
        print("\n" + "=" * 70)
        print("ANÁLISIS CONSOLIDADO DE MÚLTIPLES ELECTRODOS")
        print("=" * 70 + "\n")

        # Encontrar todas las carpetas de electrodos
        electrode_dirs = sorted(
            [
                d
                for d in self.experiment_dir.iterdir()
                if d.is_dir() and d.name.startswith("electrode_")
            ]
        )

        if not electrode_dirs:
            print("✗ No se encontraron carpetas de electrodos")
            return None

        print(f"Encontrados {len(electrode_dirs)} electrodo(s)\n")

        consolidated_results = {
            "experiment_name": self.experiment_dir.name,
            "num_electrodes": len(electrode_dirs),
            "electrodes": {},
            "mean_positions": {},
            "no_response_electrodes": [],
        }

        # Analizar cada electrodo
        all_mean_positions = []
        all_electrode_indices = []  # todos los electrodos procesados (para log)
        valid_indices = []          # solo los que aportan posición; alineado con all_mean_positions

        for electrode_dir in electrode_dirs:
            electrode_index = int(electrode_dir.name.split("_")[1])
            all_electrode_indices.append(electrode_index)

            print(f"{'='*70}")
            print(f"Analizando electrodo {electrode_index:03d}...")
            print(f"{'='*70}")

            try:
                # Usar el analizador individual
                analyzer = PhospheneMappingAnalyzer(electrode_dir)
                # Heredar la geometría real de la sesión (px/grado isotrópico,
                # resolución y centro) del primer electrodo analizado.
                self._adopt_geometry(analyzer)
                results = analyzer.analyze_electrode_repetitions()

                # Propagar identidad de implante (si el CSV combina varios)
                # desde metadata.json hasta el resultado consolidado.
                if results:
                    electrode_info = analyzer.metadata.get("electrode_info") or {}
                    results["implant_id"] = electrode_info.get("implant_id")
                    results["implant_local_index"] = electrode_info.get("implant_local_index")

                if results and results.get("mean_position"):
                    # Electrodo con respuesta utilizable
                    consolidated_results["electrodes"][str(electrode_index)] = results

                    # Extraer posición promedio (índice alineado con valid_indices)
                    mean_pos = np.array(
                        [results["mean_position"]["x"], results["mean_position"]["y"]]
                    )
                    all_mean_positions.append(mean_pos)
                    valid_indices.append(electrode_index)

                    print(
                        f"\n✓ Electrodo {electrode_index:03d} analizado correctamente"
                    )

                elif results and results.get("status") == "no_response":
                    # El electrodo no produjo respuesta (todas las repeticiones
                    # vacías). Se registra aparte para que sea VISIBLE en el
                    # reporte, pero NO entra en el mapa ni en las distancias
                    # (no tiene centroide).
                    consolidated_results["no_response_electrodes"].append(
                        electrode_index
                    )
                    print(
                        f"\n⚠ Electrodo {electrode_index:03d}: SIN RESPUESTA "
                        f"(no_response) — excluido del mapa y de las distancias"
                    )

                else:
                    print(f"\n✗ No se pudo analizar electrodo {electrode_index:03d}")

            except Exception as e:
                print(f"\n✗ Error analizando electrodo {electrode_index:03d}: {e}")
                import traceback

                traceback.print_exc()

        # Calcular estadísticas consolidadas
        if all_mean_positions:
            all_mean_positions = np.array(all_mean_positions)

            # Estadísticas de distancias entre electrodos
            print("\n" + "=" * 70)
            print("ESTADÍSTICAS CONSOLIDADAS")
            print("=" * 70 + "\n")

            print(f"Electrodos procesados: {all_electrode_indices}")
            print(f"Con posición promedio: {valid_indices}")
            print(f"Posiciones promedio obtenidas: {len(all_mean_positions)}\n")

            # Calcular distancias entre pares de electrodos. Se recorre
            # valid_indices (no all_electrode_indices) porque all_mean_positions
            # solo contiene los electrodos con respuesta: usar el listado
            # completo desalineaba índices y posiciones.
            print("Distancias entre posiciones promedio:")
            for i, idx_i in enumerate(valid_indices):
                for j, idx_j in enumerate(valid_indices):
                    if i < j:
                        pos_i = all_mean_positions[i]
                        pos_j = all_mean_positions[j]
                        distance_px = np.sqrt(np.sum((pos_i - pos_j) ** 2))
                        dx_px = float(pos_i[0] - pos_j[0])
                        dy_px = float(pos_i[1] - pos_j[1])
                        distance_deg = float(
                            distance_deg(
                                np.array([dx_px]),
                                np.array([dy_px]),
                                self.pixels_per_degree_x,
                                self.pixels_per_degree_y,
                            )[0]
                        )
                        print(
                            f"  Electrodo {idx_i} → {idx_j}: {distance_px:.1f} px ({distance_deg:.2f}°)"
                        )

            consolidated_results["mean_positions"] = {
                str(idx): {"x": float(pos[0]), "y": float(pos[1])}
                for idx, pos in zip(valid_indices, all_mean_positions)
            }

        # Guardar resultados consolidados
        results_file = self.consolidated_dir / "consolidated_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(consolidated_results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Resultados consolidados guardados en: {results_file.name}")

        # Export tabla consolidada (todas las repeticiones)
        self.export_consolidated_table(consolidated_results)

        return consolidated_results

    def export_consolidated_table(self, consolidated_results):
        """Exporta un CSV con todas las repeticiones de todos los electrodos."""
        if not consolidated_results or not consolidated_results.get("electrodes"):
            return None

        rows = []
        for electrode_idx_str, results in consolidated_results["electrodes"].items():
            electrode_index = int(electrode_idx_str)
            stim = np.array(
                results.get("stimulation_position", [np.nan, np.nan]), dtype=float
            )
            for rep in results.get("per_repetition_metrics", []) or []:
                rows.append(
                    {
                        "electrode_index": electrode_index,
                        "repetition_number": int(rep["repetition_number"]),
                        "dx_to_stim_px": float(rep["dx_to_stim_px"]),
                        "dy_to_stim_px": float(rep["dy_to_stim_px"]),
                        "distance_to_stim_px": float(rep["distance_to_stim_px"]),
                        "dx_to_stim_deg": float(rep["dx_to_stim_deg"]),
                        "dy_to_stim_deg": float(rep["dy_to_stim_deg"]),
                        "distance_to_stim_deg": float(rep["distance_to_stim_deg"]),
                        "stim_x_px": float(stim[0]),
                        "stim_y_px": float(stim[1]),
                    }
                )

        if not rows:
            return None

        output_csv = self.consolidated_dir / "consolidated_repetitions.csv"
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        print(f"✓ Tabla consolidada guardada en: {output_csv.name}")
        return output_csv

    def visualize_consolidated_map(self, consolidated_results):
        """
        Crea un mapa visual consolidado con todos los electrodos

        Args:
            consolidated_results: Diccionario de resultados consolidados
        """
        if not consolidated_results or not consolidated_results.get("electrodes"):
            print("No hay datos para visualizar")
            return

        print("\n" + "=" * 70)
        print("GENERANDO MAPA CONSOLIDADO")
        print("=" * 70 + "\n")

        # Crear figura
        fig, ax = plt.subplots(figsize=(14, 12))

        # Colores para cada electrodo
        colors = plt.cm.tab10(
            np.linspace(0, 1, len(consolidated_results["electrodes"]))
        )

        electrode_indices = sorted(
            [int(e) for e in consolidated_results["electrodes"].keys()]
        )
        all_positions = []

        # Dibujar cada electrodo
        for color_idx, electrode_idx in enumerate(electrode_indices):
            electrode_idx_str = str(electrode_idx)
            results = consolidated_results["electrodes"][electrode_idx_str]

            centroids = np.array(results["centroids"])
            mean_pos = np.array(
                [results["mean_position"]["x"], results["mean_position"]["y"]]
            )
            stim_pos = np.array(results["stimulation_position"])

            all_positions.append(mean_pos)
            color = colors[color_idx]

            # Dibujar centroides individuales
            ax.scatter(
                centroids[:, 0],
                centroids[:, 1],
                c=[color],
                s=50,
                alpha=0.4,
                edgecolors="gray",
                linewidths=1,
            )

            # Dibujar posición promedio
            ax.scatter(
                mean_pos[0],
                mean_pos[1],
                c=[color],
                s=400,
                marker="o",
                edgecolors="black",
                linewidths=2,
                zorder=10,
                label=f"Electrodo {electrode_idx} (media)",
            )

            # Dibujar posición del estímulo (más pequeña)
            ax.scatter(
                stim_pos[0],
                stim_pos[1],
                c=[color],
                s=100,
                marker="x",
                linewidths=2,
                alpha=0.6,
            )

            # Circles de desviación estándar
            std_x = results["std_position"]["x"]
            std_y = results["std_position"]["y"]
            std_radius = np.sqrt(std_x**2 + std_y**2)
            circle = plt.Circle(
                mean_pos,
                std_radius,
                color=color,
                fill=False,
                linestyle="--",
                linewidth=1.5,
                alpha=0.5,
            )
            ax.add_patch(circle)

        # Calcular límites y hacer zoom automático
        all_positions = np.array(all_positions)

        # Margen dinámico
        all_x = all_positions[:, 0]
        all_y = all_positions[:, 1]

        # Rango de datos + margen
        x_margin = (all_x.max() - all_x.min()) * 0.3
        y_margin = (all_y.max() - all_y.min()) * 0.3

        x_min = max(0, all_x.min() - x_margin)
        x_max = min(self.screen_width, all_x.max() + x_margin)
        y_min = max(0, all_y.min() - y_margin)
        y_max = min(self.screen_height, all_y.max() + y_margin)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)  # Invertir Y

        # Conectar posiciones promedio con líneas
        if len(all_positions) > 1:
            for i in range(len(all_positions)):
                for j in range(i + 1, len(all_positions)):
                    ax.plot(
                        [all_positions[i, 0], all_positions[j, 0]],
                        [all_positions[i, 1], all_positions[j, 1]],
                        "gray",
                        alpha=0.3,
                        linewidth=2,
                        linestyle=":",
                    )

        # Configuración
        ax.set_xlabel("X (píxeles)", fontsize=12)
        ax.set_ylabel("Y (píxeles)", fontsize=12)
        title = (
            f"Mapa Consolidado - {len(electrode_indices)} Electrodos\n"
            f"Electrodos: {electrode_indices}"
        )
        no_resp = sorted(consolidated_results.get("no_response_electrodes", []))
        if no_resp:
            title += f"\nSin respuesta (no mostrados): {no_resp}"
        ax.set_title(
            title,
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        # Guardar figura
        output_path = self.consolidated_dir / "consolidated_map.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"✓ Mapa consolidado guardado en: {output_path.name}")
        return fig

    def create_summary_report(self, consolidated_results):
        """
        Crea un reporte de texto consolidado

        Args:
            consolidated_results: Diccionario de resultados consolidados
        """
        print("\n" + "=" * 70)
        print("GENERANDO REPORTE CONSOLIDADO")
        print("=" * 70 + "\n")

        report_lines = [
            "=" * 70,
            "REPORTE CONSOLIDADO DE MAPEO DE FOSFENOS",
            "=" * 70,
            "",
            f"Experimento: {consolidated_results['experiment_name']}",
            f"Número de electrodos: {consolidated_results['num_electrodes']}",
            f"Con respuesta: {len(consolidated_results.get('electrodes', {}))}  |  "
            f"Sin respuesta: {len(consolidated_results.get('no_response_electrodes', []))}",
            "",
            "-" * 70,
            "RESULTADOS POR ELECTRODO",
            "-" * 70,
            "",
        ]

        no_resp = sorted(consolidated_results.get("no_response_electrodes", []))
        if no_resp:
            report_lines.extend(
                [
                    f"⚠ ELECTRODOS SIN RESPUESTA ({len(no_resp)}): {no_resp}",
                    "  (todas las repeticiones vacías — excluidos del mapa y de las distancias)",
                    "",
                ]
            )

        for electrode_idx_str in sorted(
            consolidated_results["electrodes"].keys(), key=lambda x: int(x)
        ):
            results = consolidated_results["electrodes"][electrode_idx_str]

            report_lines.extend(
                [
                    f"ELECTRODO {electrode_idx_str}",
                    f"{'─' * 50}",
                    f"Repeticiones válidas: {results['num_valid_repetitions']}/{results['num_total_repetitions']}",
                    f"Posición promedio: ({results['mean_position']['x']:.1f}, {results['mean_position']['y']:.1f}) px",
                    f"Desviación estándar: ({results['std_position']['x']:.1f}, {results['std_position']['y']:.1f}) px",
                    f"Distancia media al promedio: {results['mean_distance_from_average']:.1f} px ({results.get('mean_distance_from_average_deg', 0):.2f}°)",
                    f"Distancia máxima al promedio: {results['max_distance_from_average']:.1f} px ({results.get('max_distance_from_average_deg', 0):.2f}°)",
                    f"Distancia media al estímulo: {results.get('distance_mean_to_stimulus_px', float('nan')):.1f} px ({results.get('distance_mean_to_stimulus_deg', float('nan')):.2f}°)",
                    "",
                ]
            )

        # Estadísticas de distancias entre electrodos
        if len(consolidated_results["electrodes"]) > 1:
            report_lines.extend(
                [
                    "-" * 70,
                    "DISTANCIAS ENTRE POSICIONES PROMEDIO",
                    "-" * 70,
                    "",
                ]
            )

            electrode_indices = sorted(
                [int(e) for e in consolidated_results["electrodes"].keys()]
            )

            for i, idx_i in enumerate(electrode_indices):
                for j, idx_j in enumerate(electrode_indices):
                    if i < j:
                        pos_i = consolidated_results["mean_positions"][str(idx_i)]
                        pos_j = consolidated_results["mean_positions"][str(idx_j)]

                        distance_px = np.sqrt(
                            (pos_i["x"] - pos_j["x"]) ** 2
                            + (pos_i["y"] - pos_j["y"]) ** 2
                        )
                        dx_px = float(pos_i["x"] - pos_j["x"])
                        dy_px = float(pos_i["y"] - pos_j["y"])
                        distance_deg = float(
                            distance_deg(
                                np.array([dx_px]),
                                np.array([dy_px]),
                                self.pixels_per_degree_x,
                                self.pixels_per_degree_y,
                            )[0]
                        )

                        report_lines.append(
                            f"Electrodo {idx_i} → {idx_j}: {distance_px:.1f} px ({distance_deg:.2f}°)"
                        )

                report_lines.append("")

        report_lines.extend(
            [
                "=" * 70,
                "FIN DEL REPORTE",
                "=" * 70,
            ]
        )

        # Guardar reporte
        report_text = "\n".join(report_lines)
        report_file = self.consolidated_dir / "consolidated_report.txt"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_text)

        print(report_text)
        print(f"\n✓ Reporte guardado en: {report_file.name}")
