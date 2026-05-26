"""
Analizador de experimentos en modo standard.

Lee el metadata.json generado por el modo standard y produce
la misma estructura de analysis_results.json que PhospheneMappingAnalyzer,
permitiendo que el módulo de aprendizaje trate ambos modos de forma idéntica.
"""

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.mapping_analyzer import (
    PhospheneMappingAnalyzer,
    _robust_sigma_mad,
    _trimmed_mean,
    _tukey_boxplot_stats,
    _distance_deg,
    _ellipse_from_cov,
)
from scripts.response_capture import resolve_response_features


class StandardExperimentAnalyzer:
    """
    Analiza un experimento en modo standard (logs/).

    Para cada electrodo estimulado genera un analysis_results.json
    con exactamente la misma estructura que PhospheneMappingAnalyzer,
    de forma que data_loader.py puede leer ambos modos de forma idéntica.
    """

    def __init__(self, experiment_dir):
        self.experiment_dir = Path(experiment_dir)

        if not self.experiment_dir.exists():
            raise FileNotFoundError(f"No se encontró la carpeta: {self.experiment_dir}")

        metadata_file = self.experiment_dir / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(
                f"No se encontró metadata.json en {self.experiment_dir}"
            )

        with open(metadata_file, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Parámetros de display
        display = self.metadata.get("display", {})
        screen_res = display.get("screen_resolution_px", [1920, 1080])
        self.screen_width = int(screen_res[0])
        self.screen_height = int(screen_res[1])

        screen_center = display.get(
            "screen_center_px", [self.screen_width // 2, self.screen_height // 2]
        )
        self.screen_center = (int(screen_center[0]), int(screen_center[1]))

        px_per_deg_x = display.get("pixels_per_degree_x")
        px_per_deg_y = display.get("pixels_per_degree_y")
        vf_scope_deg = abs(float(display.get("vf_scope_deg", 15.0)))

        self.pixels_per_degree_x = (
            float(px_per_deg_x)
            if px_per_deg_x
            else self.screen_width / (2.0 * vf_scope_deg)
        )
        self.pixels_per_degree_y = (
            float(px_per_deg_y)
            if px_per_deg_y
            else self.screen_height / (2.0 * vf_scope_deg)
        )

        # Carpeta de análisis
        self.analysis_dir = self.experiment_dir / "analysis"
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

        print(f"[StandardExperimentAnalyzer] Experimento: {self.experiment_dir.name}")
        print(
            f"                             Electrodos: {len(self.metadata.get('phosphenes', []))}"
        )

    def _px_to_deg(self, x_px, y_px):
        cx, cy = self.screen_center
        x_deg = (float(x_px) - float(cx)) / float(self.pixels_per_degree_x)
        y_deg = -(float(y_px) - float(cy)) / float(self.pixels_per_degree_y)
        return float(x_deg), float(y_deg)

    def _extract_centroid(self, image_path, background_threshold=10):
        """Mismo método que PhospheneMappingAnalyzer."""
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)
        gray = np.sum(img_array, axis=2)
        drawn_pixels = gray > background_threshold

        if not np.any(drawn_pixels):
            return None

        y_coords, x_coords = np.where(drawn_pixels)
        weights = gray[drawn_pixels]

        centroid_x = float(np.average(x_coords, weights=weights))
        centroid_y = float(np.average(y_coords, weights=weights))

        x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
        y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
        bbox_width = x_max - x_min + 1
        bbox_height = y_max - y_min + 1
        bbox_area = bbox_width * bbox_height
        n_pixels = int(x_coords.size)

        return {
            "centroid": (centroid_x, centroid_y),
            "n_pixels": n_pixels,
            "intensity_sum": float(np.sum(weights)),
            "bbox": {
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "width": bbox_width,
                "height": bbox_height,
                "area": bbox_area,
            },
            "fill_ratio": float(n_pixels / bbox_area) if bbox_area > 0 else 0.0,
        }

    def analyze_all_electrodes(self):
        """
        Analiza todos los electrodos del experimento standard.
        Genera un analysis_results.json por electrodo en analysis/.
        """
        phosphenes = self.metadata.get("phosphenes", [])
        if not phosphenes:
            print("✗ No hay fosfenos en el experimento")
            return []

        results_list = []

        # Catch / practice gating (de §1.2 y §1.4 del rigor plan): los catch
        # trials no llevan estimulación, las prácticas no se guardan como
        # señal. Ninguno entra en el análisis principal. Catch responses se
        # reportan por separado al final.
        catch_total = 0
        catch_with_response = 0

        for phos in phosphenes:
            if phos.get("is_practice"):
                continue
            if phos.get("is_catch"):
                catch_total += 1
                resp = resolve_response_features(
                    phos, self.experiment_dir, self._extract_centroid
                )
                if resp.get("ok"):
                    catch_with_response += 1
                continue

            electrode_index = phos["electrode_index"]
            stim_pos = phos.get("position", [0, 0])
            electrode_info = phos.get("electrode_info", {})
            pred_deg = electrode_info.get("visual_position_deg", [0.0, 0.0])

            print(f"\n[StandardAnalyzer] Electrodo {electrode_index}...")

            response = resolve_response_features(
                phos,
                self.experiment_dir,
                self._extract_centroid,
            )
            if not response["ok"]:
                print(f"  ✗ {response['error']}")
                continue
            response_mode = response["mode"]
            response_file_name = response["source_file"]
            features = response["features"]

            centroid = features["centroid"]
            centroid_deg = self._px_to_deg(centroid[0], centroid[1])
            stim_pos_arr = np.array(stim_pos, dtype=float)
            stim_pos_deg = self._px_to_deg(stim_pos_arr[0], stim_pos_arr[1])

            # Calcular métricas (con una sola repetición)
            centroids = np.array([centroid])
            mean_position = centroids[0]
            std_position = np.array([0.0, 0.0])
            mean_position_deg = centroid_deg
            distances_from_avg = np.array([0.0])

            dx_to_stim_px = centroid[0] - stim_pos_arr[0]
            dy_to_stim_px = centroid[1] - stim_pos_arr[1]
            dist_to_stim_px = float(np.hypot(dx_to_stim_px, dy_to_stim_px))
            dx_to_stim_deg = dx_to_stim_px / self.pixels_per_degree_x
            dy_to_stim_deg = dy_to_stim_px / self.pixels_per_degree_y
            dist_to_stim_deg = float(np.hypot(dx_to_stim_deg, dy_to_stim_deg))

            offset_px = mean_position - stim_pos_arr
            offset_deg_x = float(offset_px[0] / self.pixels_per_degree_x)
            offset_deg_y = float(offset_px[1] / self.pixels_per_degree_y)
            dist_mean_to_stim_px = float(np.hypot(offset_px[0], offset_px[1]))
            dist_mean_to_stim_deg = float(np.hypot(offset_deg_x, offset_deg_y))

            results = {
                "electrode_index": int(electrode_index),
                "experiment_mode": "standard",
                "pred_x_deg": float(pred_deg[0]),
                "pred_y_deg": float(pred_deg[1]),
                "num_total_repetitions": 1,
                "num_valid_repetitions": 1,
                "num_invalid_repetitions": 0,
                "centroids": [list(centroid)],
                "centroids_deg": [list(centroid_deg)],
                "valid_repetitions": [1],
                "per_repetition": [
                    {
                        "repetition_number": 1,
                        "response_mode": response_mode,
                        "response_file": response_file_name,
                        "drawing_file": response_file_name if response_mode == "drawing" else None,
                        "saccade_samples_file": response_file_name if response_mode == "saccade" else None,
                        "centroid": {"x": float(centroid[0]), "y": float(centroid[1])},
                        "centroid_deg": {
                            "x": float(centroid_deg[0]),
                            "y": float(centroid_deg[1]),
                        },
                        "n_pixels": features["n_pixels"],
                        "intensity_sum": features["intensity_sum"],
                        "bbox": features["bbox"],
                        "fill_ratio": features["fill_ratio"],
                    }
                ],
                "mean_position": {
                    "x": float(mean_position[0]),
                    "y": float(mean_position[1]),
                },
                "mean_position_deg": {
                    "x": float(mean_position_deg[0]),
                    "y": float(mean_position_deg[1]),
                },
                "median_position": {"x": float(centroid[0]), "y": float(centroid[1])},
                "median_position_deg": {
                    "x": float(centroid_deg[0]),
                    "y": float(centroid_deg[1]),
                },
                "trimmed_mean_position": {
                    "x": float(centroid[0]),
                    "y": float(centroid[1]),
                },
                "trimmed_mean_position_deg": {
                    "x": float(centroid_deg[0]),
                    "y": float(centroid_deg[1]),
                },
                "std_position": {"x": 0.0, "y": 0.0},
                "robust_sigma_mad": {"x": 0.0, "y": 0.0},
                "mean_distance_from_average": 0.0,
                "max_distance_from_average": 0.0,
                "mean_distance_from_average_deg": 0.0,
                "max_distance_from_average_deg": 0.0,
                "distances_from_average_deg": [0.0],
                "stimulation_position": [
                    float(stim_pos_arr[0]),
                    float(stim_pos_arr[1]),
                ],
                "stimulation_position_deg": [
                    float(stim_pos_deg[0]),
                    float(stim_pos_deg[1]),
                ],
                "per_repetition_metrics": [
                    {
                        "repetition_number": 1,
                        "dx_to_stim_px": float(dx_to_stim_px),
                        "dy_to_stim_px": float(dy_to_stim_px),
                        "distance_to_stim_px": dist_to_stim_px,
                        "dx_to_stim_deg": float(dx_to_stim_deg),
                        "dy_to_stim_deg": float(dy_to_stim_deg),
                        "distance_to_stim_deg": dist_to_stim_deg,
                    }
                ],
                "boxplot_stats": {
                    "dx_to_stim_deg": _tukey_boxplot_stats(np.array([dx_to_stim_deg])),
                    "dy_to_stim_deg": _tukey_boxplot_stats(np.array([dy_to_stim_deg])),
                    "distance_to_stim_deg": _tukey_boxplot_stats(
                        np.array([dist_to_stim_deg])
                    ),
                },
                "offset_mean_to_stimulus_px": {
                    "x": float(offset_px[0]),
                    "y": float(offset_px[1]),
                },
                "offset_mean_to_stimulus_deg": {"x": offset_deg_x, "y": offset_deg_y},
                "distance_mean_to_stimulus_px": dist_mean_to_stim_px,
                "distance_mean_to_stimulus_deg": dist_mean_to_stim_deg,
            }

            # Guardar por electrodo
            electrode_dir = self.analysis_dir / f"electrode_{electrode_index:03d}"
            electrode_dir.mkdir(parents=True, exist_ok=True)

            results_file = electrode_dir / "analysis_results.json"
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            # CSV por repetición (mismo formato que mapping)
            csv_file = electrode_dir / "analysis_repetitions.csv"
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "electrode_index",
                        "repetition_number",
                        "centroid_x_px",
                        "centroid_y_px",
                        "centroid_x_deg",
                        "centroid_y_deg",
                        "n_pixels",
                        "fill_ratio",
                        "dx_to_stim_px",
                        "dy_to_stim_px",
                        "distance_to_stim_px",
                        "dx_to_stim_deg",
                        "dy_to_stim_deg",
                        "distance_to_stim_deg",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "electrode_index": electrode_index,
                        "repetition_number": 1,
                        "centroid_x_px": centroid[0],
                        "centroid_y_px": centroid[1],
                        "centroid_x_deg": centroid_deg[0],
                        "centroid_y_deg": centroid_deg[1],
                        "n_pixels": features["n_pixels"],
                        "fill_ratio": features["fill_ratio"],
                        "dx_to_stim_px": dx_to_stim_px,
                        "dy_to_stim_px": dy_to_stim_px,
                        "distance_to_stim_px": dist_to_stim_px,
                        "dx_to_stim_deg": dx_to_stim_deg,
                        "dy_to_stim_deg": dy_to_stim_deg,
                        "distance_to_stim_deg": dist_to_stim_deg,
                    }
                )

            print(
                f"  ✓ Electrodo {electrode_index}: centroide ({centroid[0]:.1f}, {centroid[1]:.1f}) px, error {dist_mean_to_stim_deg:.2f}°"
            )
            results_list.append(results)

        # Guardar CSV consolidado
        self._export_consolidated_csv(results_list)

        # Reporte agregado de catch trials (tasa de false-positive)
        if catch_total > 0:
            rate = catch_with_response / catch_total
            print(
                f"\n[StandardAnalyzer] Catch trials: {catch_with_response}/{catch_total} "
                f"con respuesta ({100.0 * rate:.1f}%)"
            )
            summary = {
                "n_total": catch_total,
                "n_with_response": catch_with_response,
                "response_rate": rate,
            }
            with open(self.analysis_dir / "catch_trial_stats.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Análisis guardado en: {self.analysis_dir}")
        return results_list

    def _export_consolidated_csv(self, results_list):
        if not results_list:
            return

        csv_file = self.analysis_dir / "consolidated_repetitions.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "electrode_index",
                    "repetition_number",
                    "dx_to_stim_px",
                    "dy_to_stim_px",
                    "distance_to_stim_px",
                    "dx_to_stim_deg",
                    "dy_to_stim_deg",
                    "distance_to_stim_deg",
                    "stim_x_px",
                    "stim_y_px",
                    "pred_x_deg",
                    "pred_y_deg",
                ],
            )
            writer.writeheader()
            for r in results_list:
                for m in r["per_repetition_metrics"]:
                    writer.writerow(
                        {
                            "electrode_index": r["electrode_index"],
                            "repetition_number": m["repetition_number"],
                            "dx_to_stim_px": m["dx_to_stim_px"],
                            "dy_to_stim_px": m["dy_to_stim_px"],
                            "distance_to_stim_px": m["distance_to_stim_px"],
                            "dx_to_stim_deg": m["dx_to_stim_deg"],
                            "dy_to_stim_deg": m["dy_to_stim_deg"],
                            "distance_to_stim_deg": m["distance_to_stim_deg"],
                            "stim_x_px": r["stimulation_position"][0],
                            "stim_y_px": r["stimulation_position"][1],
                            "pred_x_deg": r["pred_x_deg"],
                            "pred_y_deg": r["pred_y_deg"],
                        }
                    )

        print(f"✓ CSV consolidado: {csv_file.name}")
