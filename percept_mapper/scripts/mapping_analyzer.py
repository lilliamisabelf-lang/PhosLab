"""
Analizador de mapeo de fosfenos

Calcula la posición promedio de un fosfeno a partir de múltiples
representaciones dibujadas por el usuario.
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
from matplotlib.patches import Ellipse
from scripts.response_capture import resolve_response_features
from scripts.schemas import ElectrodeAnalysisResult

try:
    from scipy.stats import chi2
except Exception:  # pragma: no cover
    chi2 = None


def _robust_sigma_mad(values: np.ndarray) -> float:
    """Estimador robusto de sigma usando MAD (Normal-consistente)."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return float(1.4826 * mad)


def _trimmed_mean(values: np.ndarray, proportion_to_cut: float = 0.1) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    proportion_to_cut = float(np.clip(proportion_to_cut, 0.0, 0.49))
    sorted_vals = np.sort(values)
    k = int(np.floor(sorted_vals.size * proportion_to_cut))
    if sorted_vals.size - 2 * k <= 0:
        return float(np.mean(sorted_vals))
    return float(np.mean(sorted_vals[k : sorted_vals.size - k]))


def _tukey_boxplot_stats(values: np.ndarray) -> dict:
    """Estadísticos de caja y bigotes (Tukey, 1.5*IQR)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "n": 0,
            "q1": None,
            "median": None,
            "q3": None,
            "iqr": None,
            "whisker_low": None,
            "whisker_high": None,
            "outliers": [],
        }

    q1, median, q3 = np.percentile(values, [25, 50, 75])
    iqr = q3 - q1
    low_fence = q1 - 1.5 * iqr
    high_fence = q3 + 1.5 * iqr

    inliers = values[(values >= low_fence) & (values <= high_fence)]
    if inliers.size == 0:
        whisker_low = float(np.min(values))
        whisker_high = float(np.max(values))
    else:
        whisker_low = float(np.min(inliers))
        whisker_high = float(np.max(inliers))

    outliers = values[(values < whisker_low) | (values > whisker_high)]

    return {
        "n": int(values.size),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "iqr": float(iqr),
        "whisker_low": float(whisker_low),
        "whisker_high": float(whisker_high),
        "outliers": [float(v) for v in outliers.tolist()],
    }


def _distance_deg(
    dx_px: np.ndarray, dy_px: np.ndarray, px_per_deg_x: float, px_per_deg_y: float
) -> np.ndarray:
    """Distancia radial en grados, respetando anisotropía px/deg (X,Y)."""
    dx_deg = dx_px / float(px_per_deg_x)
    dy_deg = dy_px / float(px_per_deg_y)
    return np.hypot(dx_deg, dy_deg)


def _ellipse_from_cov(
    cov: np.ndarray, confidence: float
) -> tuple[float, float, float] | None:
    """Devuelve (width, height, angle_deg) para un Ellipse en coordenadas del plano."""
    if chi2 is None:
        return None
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        return None

    # Regularización mínima por estabilidad numérica
    cov = cov + np.eye(2) * 1e-9

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    scale = float(np.sqrt(chi2.ppf(confidence, df=2)))
    width = 2.0 * scale * float(np.sqrt(max(eigvals[0], 0.0)))
    height = 2.0 * scale * float(np.sqrt(max(eigvals[1], 0.0)))
    angle = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
    return width, height, angle


class PhospheneMappingAnalyzer:
    """
    Analiza los resultados de un experimento de mapeo de fosfenos
    """

    def __init__(self, electrode_dir):
        """
        Inicializa el analizador

        Args:
            electrode_dir: Ruta a la carpeta del electrodo (e.g., 'mapping_experiments/mapping_test_20260310_120000/electrode_001')
        """
        self.electrode_dir = Path(electrode_dir)

        if not self.electrode_dir.exists():
            raise FileNotFoundError(
                f"No se encontró la carpeta del electrodo: {self.electrode_dir}"
            )

        # Cargar metadata
        metadata_file = self.electrode_dir / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(
                f"No se encontró metadata.json en {self.electrode_dir}"
            )

        with open(metadata_file, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.electrode_index = self.metadata["electrode_index"]
        self.num_repetitions = self.metadata["num_repetitions"]

        def _as_float_or_none(value):
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        def _as_int_or_none(value):
            if value is None:
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        # Parámetros de visualización (se leen de metadata.json si existe; fallback compatible)
        display = self.metadata.get("display")
        if not isinstance(display, dict):
            display = {}

        screen_res = display.get("screen_resolution_px")
        if isinstance(screen_res, (list, tuple)) and len(screen_res) == 2:
            self.screen_width = _as_int_or_none(screen_res[0]) or 1920
            self.screen_height = _as_int_or_none(screen_res[1]) or 1080
        else:
            self.screen_width = 1920
            self.screen_height = 1080

        screen_center = display.get("screen_center_px")
        if isinstance(screen_center, (list, tuple)) and len(screen_center) == 2:
            cx = _as_int_or_none(screen_center[0])
            cy = _as_int_or_none(screen_center[1])
            self.screen_center = (
                cx if cx is not None else self.screen_width // 2,
                cy if cy is not None else self.screen_height // 2,
            )
        else:
            self.screen_center = (self.screen_width // 2, self.screen_height // 2)

        # Parámetros de conversión píxeles → grados
        # Prioridad:
        # 1) px/deg guardados en metadata
        # 2) resolución + assumed_fov
        # 3) fallback histórico (30°×30° en 1920×1080)
        px_per_deg_x = _as_float_or_none(display.get("pixels_per_degree_x"))
        px_per_deg_y = _as_float_or_none(display.get("pixels_per_degree_y"))

        # Contrato único (pipeline): vf_scope_deg (semiancho).
        vf_scope_deg = _as_float_or_none(display.get("vf_scope_deg"))

        if vf_scope_deg is not None:
            vf_scope_deg = abs(vf_scope_deg)
        if vf_scope_deg is None:
            vf_scope_deg = 15.0
        vf_scope_deg = abs(vf_scope_deg)

        assumed_fov_x = 2.0 * vf_scope_deg
        assumed_fov_y = 2.0 * vf_scope_deg
        if assumed_fov_x is not None:
            assumed_fov_x = abs(assumed_fov_x)
        if assumed_fov_y is not None:
            assumed_fov_y = abs(assumed_fov_y)
        if assumed_fov_x is None:
            assumed_fov_x = 30.0
        if assumed_fov_y is None:
            assumed_fov_y = 30.0

        if px_per_deg_x is None:
            px_per_deg_x = self.screen_width / assumed_fov_x if assumed_fov_x else 64.0
        if px_per_deg_y is None:
            px_per_deg_y = self.screen_height / assumed_fov_y if assumed_fov_y else 36.0

        self.pixels_per_degree_x = float(px_per_deg_x)
        self.pixels_per_degree_y = float(px_per_deg_y)

        self.fov_x_deg = display.get("fov_x_deg_range")
        self.fov_y_deg = display.get("fov_y_deg_range")
        if not (isinstance(self.fov_x_deg, (list, tuple)) and len(self.fov_x_deg) == 2):
            self.fov_x_deg = [-(assumed_fov_x / 2.0), (assumed_fov_x / 2.0)]
        if not (isinstance(self.fov_y_deg, (list, tuple)) and len(self.fov_y_deg) == 2):
            self.fov_y_deg = [-(assumed_fov_y / 2.0), (assumed_fov_y / 2.0)]

        self.fov_width_deg = float(self.fov_x_deg[1] - self.fov_x_deg[0])
        self.fov_height_deg = float(self.fov_y_deg[1] - self.fov_y_deg[0])

        print(f"[PhospheneMappingAnalyzer] Cargado electrodo {self.electrode_index}")
        print(
            f"                           Repeticiones totales: {self.num_repetitions}"
        )
        print(f"                           Centro pantalla: {self.screen_center}")
        print(
            f"                           Conversión: {self.pixels_per_degree_x:.1f} px/° (X), {self.pixels_per_degree_y:.1f} px/° (Y)"
        )

    def calculate_drawing_centroid(self, image_path, background_threshold=10):
        """
        Calcula el centroide (centro de masa) de un dibujo

        Args:
            image_path: Ruta a la imagen PNG del dibujo
            background_threshold: Umbral para considerar un píxel como "dibujado"
                                 (valores por encima de este umbral se consideran trazos)

        Returns:
            tuple: (centroid_x, centroid_y) o None si no hay píxeles dibujados
        """
        # Cargar imagen
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)

        # Convertir a escala de grises sumando los canales RGB
        # Los trazos son amarillos (255, 255, 0) o blancos, el fondo es negro (0, 0, 0)
        gray = np.sum(img_array, axis=2)  # Suma de R+G+B

        # Encontrar píxeles dibujados (no negros)
        drawn_pixels = gray > background_threshold

        # Verificar que hay píxeles dibujados
        if not np.any(drawn_pixels):
            print(f"      ⚠ No se encontraron trazos en {image_path.name}")
            return None

        # Obtener coordenadas de los píxeles dibujados
        y_coords, x_coords = np.where(drawn_pixels)

        # Calcular centroide (promedio ponderado por intensidad)
        # Usamos los valores de intensidad como pesos
        weights = gray[drawn_pixels]

        centroid_x = np.average(x_coords, weights=weights)
        centroid_y = np.average(y_coords, weights=weights)

        return (centroid_x, centroid_y)

    def _px_to_deg(self, x_px: float, y_px: float) -> tuple[float, float]:
        """Convierte coordenadas de pantalla (px) a grados visuales (deg).

        Convención:
        - Origen (0,0 deg) en el centro de pantalla.
        - X positivo hacia la derecha.
        - Y positivo hacia arriba (por eso se invierte el signo respecto a píxeles).
        """
        cx, cy = self.screen_center
        x_deg = (float(x_px) - float(cx)) / float(self.pixels_per_degree_x)
        y_deg = -(float(y_px) - float(cy)) / float(self.pixels_per_degree_y)
        return (float(x_deg), float(y_deg))

    def extract_drawing_features(self, image_path, background_threshold=10):
        """Extrae centroides y features básicas del trazo para control de calidad."""
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

        x_min = int(np.min(x_coords))
        x_max = int(np.max(x_coords))
        y_min = int(np.min(y_coords))
        y_max = int(np.max(y_coords))

        bbox_width = x_max - x_min + 1
        bbox_height = y_max - y_min + 1
        bbox_area = int(bbox_width * bbox_height)
        n_pixels = int(x_coords.size)

        fill_ratio = float(n_pixels / bbox_area) if bbox_area > 0 else 0.0

        return {
            "centroid": (centroid_x, centroid_y),
            "n_pixels": n_pixels,
            "intensity_sum": float(np.sum(weights)),
            "bbox": {
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "width": int(bbox_width),
                "height": int(bbox_height),
                "area": bbox_area,
            },
            "fill_ratio": fill_ratio,
        }

    def analyze_electrode_repetitions(self):
        """
        Analiza todas las repeticiones de un electrodo y calcula la posición promedio

        Returns:
            dict: Diccionario con los resultados del análisis
        """
        print(f"\n{'='*70}")
        print(f"ANALIZANDO ELECTRODO {self.electrode_index}")
        print(f"{'='*70}\n")

        centroids = []
        valid_repetitions = []
        per_repetition = []
        # Catch trials: contar cuántos produjeron una respuesta y cuántos no.
        # Tasa de false-positives = catch_with_response / catch_total.
        catch_total = 0
        catch_with_response = 0

        # Analizar cada repetición
        for rep_data in self.metadata["repetitions"]:
            # Skip practice trials: nunca debieron persistir aquí pero por si acaso.
            if rep_data.get("is_practice"):
                continue
            # Catch trials: analizar por separado, no entran en el centroide.
            if rep_data.get("is_catch"):
                catch_total += 1
                # Tuvo respuesta válida? (saccade reportó endpoint, o drawing tenía píxeles)
                response = resolve_response_features(
                    rep_data,
                    self.electrode_dir,
                    self.extract_drawing_features,
                )
                if response.get("ok"):
                    catch_with_response += 1
                continue

            rep_number = rep_data["repetition_number"]
            print(f"Repeticion {rep_number}: ", end="")

            response = resolve_response_features(
                rep_data,
                self.electrode_dir,
                self.extract_drawing_features,
            )
            if not response["ok"]:
                print(f"X {response['error']}")
                continue
            response_mode = response["mode"]
            features = response["features"]
            centroid = features["centroid"]
            rep_source_file = response["source_file"]

            centroid_deg = self._px_to_deg(centroid[0], centroid[1])

            centroids.append(centroid)
            valid_repetitions.append(rep_number)
            per_repetition.append(
                {
                    "repetition_number": int(rep_number),
                    "response_mode": response_mode,
                    "response_file": rep_source_file,
                    "drawing_file": rep_source_file if response_mode == "drawing" else None,
                    "saccade_samples_file": rep_source_file if response_mode == "saccade" else None,
                    "centroid": {"x": float(centroid[0]), "y": float(centroid[1])},
                    "centroid_deg": {
                        "x": float(centroid_deg[0]),
                        "y": float(centroid_deg[1]),
                    },
                    "n_pixels": int(features["n_pixels"]),
                    "intensity_sum": float(features["intensity_sum"]),
                    "bbox": features["bbox"],
                    "fill_ratio": float(features["fill_ratio"]),
                }
            )
            print(f"✓ Centroide: ({centroid[0]:.1f}, {centroid[1]:.1f})")

        # Verificar que hay suficientes repeticiones válidas
        if len(centroids) == 0:
            print("\n✗ ERROR: No se encontraron repeticiones válidas")
            return None

        # Calcular estadísticas
        centroids = np.array(centroids)
        mean_position = np.mean(centroids, axis=0)
        std_position = np.std(centroids, axis=0)

        # Convertir centroide agregado a grados
        mean_position_deg = self._px_to_deg(mean_position[0], mean_position[1])
        median_position_deg = self._px_to_deg(
            float(np.median(centroids[:, 0])), float(np.median(centroids[:, 1]))
        )
        trimmed_mean_position_deg = self._px_to_deg(
            _trimmed_mean(centroids[:, 0], 0.1), _trimmed_mean(centroids[:, 1], 0.1)
        )

        # Convertir centroides individuales a grados (para usar directo en aprendizaje)
        centroids_deg = [
            list(self._px_to_deg(float(x), float(y))) for x, y in centroids.tolist()
        ]

        # Calcular distancia de cada centroide a la media
        distances = np.sqrt(np.sum((centroids - mean_position) ** 2, axis=1))
        mean_distance = np.mean(distances)
        max_distance = np.max(distances)

        # Calcular número de repeticiones inválidas
        num_invalid = self.num_repetitions - len(centroids)

        # Distancia en grados visuales (ANISOTRÓPICA: X/Y por separado)
        dx_from_mean_px = centroids[:, 0] - mean_position[0]
        dy_from_mean_px = centroids[:, 1] - mean_position[1]
        distances_from_average_deg = _distance_deg(
            dx_from_mean_px,
            dy_from_mean_px,
            self.pixels_per_degree_x,
            self.pixels_per_degree_y,
        )
        mean_distance_deg = float(np.mean(distances_from_average_deg))
        max_distance_deg = float(np.max(distances_from_average_deg))

        # Comparación con posición real del estímulo
        stim_pos = np.array(self.metadata["repetitions"][0]["position"], dtype=float)
        stim_pos_deg = self._px_to_deg(stim_pos[0], stim_pos[1])
        dx_to_stim_px = centroids[:, 0] - stim_pos[0]
        dy_to_stim_px = centroids[:, 1] - stim_pos[1]
        dist_to_stim_px = np.hypot(dx_to_stim_px, dy_to_stim_px)
        dist_to_stim_deg = _distance_deg(
            dx_to_stim_px,
            dy_to_stim_px,
            self.pixels_per_degree_x,
            self.pixels_per_degree_y,
        )

        # Resultados
        results = {
            "electrode_index": self.electrode_index,
            "num_total_repetitions": self.num_repetitions,
            "num_valid_repetitions": len(centroids),
            "num_invalid_repetitions": num_invalid,
            "centroids": centroids.tolist(),
            "centroids_deg": centroids_deg,
            "valid_repetitions": valid_repetitions,
            "per_repetition": per_repetition,
            "mean_position": {
                "x": float(mean_position[0]),
                "y": float(mean_position[1]),
            },
            "mean_position_deg": {
                "x": float(mean_position_deg[0]),
                "y": float(mean_position_deg[1]),
            },
            "median_position": {
                "x": float(np.median(centroids[:, 0])),
                "y": float(np.median(centroids[:, 1])),
            },
            "median_position_deg": {
                "x": float(median_position_deg[0]),
                "y": float(median_position_deg[1]),
            },
            "trimmed_mean_position": {
                "x": _trimmed_mean(centroids[:, 0], 0.1),
                "y": _trimmed_mean(centroids[:, 1], 0.1),
            },
            "trimmed_mean_position_deg": {
                "x": float(trimmed_mean_position_deg[0]),
                "y": float(trimmed_mean_position_deg[1]),
            },
            "std_position": {
                "x": float(std_position[0]),
                "y": float(std_position[1]),
            },
            "robust_sigma_mad": {
                "x": _robust_sigma_mad(centroids[:, 0]),
                "y": _robust_sigma_mad(centroids[:, 1]),
            },
            "mean_distance_from_average": float(mean_distance),
            "max_distance_from_average": float(max_distance),
            "mean_distance_from_average_deg": float(mean_distance_deg),
            "max_distance_from_average_deg": float(max_distance_deg),
            "distances_from_average_deg": [
                float(v) for v in distances_from_average_deg.tolist()
            ],
            "stimulation_position": stim_pos.tolist(),
            "stimulation_position_deg": [
                float(stim_pos_deg[0]),
                float(stim_pos_deg[1]),
            ],
            "per_repetition_metrics": [
                {
                    "repetition_number": int(rep),
                    "dx_to_stim_px": float(dx_px),
                    "dy_to_stim_px": float(dy_px),
                    "distance_to_stim_px": float(dr_px),
                    "dx_to_stim_deg": float(dx_px / self.pixels_per_degree_x),
                    "dy_to_stim_deg": float(dy_px / self.pixels_per_degree_y),
                    "distance_to_stim_deg": float(dr_deg),
                }
                for rep, dx_px, dy_px, dr_px, dr_deg in zip(
                    valid_repetitions,
                    dx_to_stim_px.tolist(),
                    dy_to_stim_px.tolist(),
                    dist_to_stim_px.tolist(),
                    dist_to_stim_deg.tolist(),
                )
            ],
            "boxplot_stats": {
                "dx_to_stim_deg": _tukey_boxplot_stats(
                    dx_to_stim_px / self.pixels_per_degree_x
                ),
                "dy_to_stim_deg": _tukey_boxplot_stats(
                    dy_to_stim_px / self.pixels_per_degree_y
                ),
                "distance_to_stim_deg": _tukey_boxplot_stats(dist_to_stim_deg),
            },
        }

        # Catch-trial false-positive rate (de §1.2 del plan de rigor):
        # cuántos catch trials produjeron una respuesta — si la tasa es alta,
        # el participante está respondiendo aún sin estímulo.
        results["catch_trial_stats"] = {
            "n_total": catch_total,
            "n_with_response": catch_with_response,
            "response_rate": (
                float(catch_with_response) / float(catch_total) if catch_total > 0 else None
            ),
        }

        # 95% confidence ellipse del centroide (de §2.3). Si tenemos al menos
        # 2 puntos válidos, la covarianza 2D se reduce a (major, minor, angle).
        ellipse_params = None
        if centroids.shape[0] >= 2:
            cov = np.cov(centroids.T, ddof=1)
            ellipse_params = _ellipse_from_cov(cov, confidence=0.95)
        if ellipse_params is not None:
            major, minor, angle_deg = ellipse_params
            results["confidence_ellipse_95_px"] = {
                "major_axis": float(major),
                "minor_axis": float(minor),
                "angle_deg": float(angle_deg),
            }
            # Mismo elipse pero en grados (asumiendo isotropía aproximada)
            results["confidence_ellipse_95_deg"] = {
                "major_axis": float(major / max(self.pixels_per_degree_x, 1e-9)),
                "minor_axis": float(minor / max(self.pixels_per_degree_y, 1e-9)),
                "angle_deg": float(angle_deg),
            }

        # Within-electrode reliability (sustituto de test-retest, §2.4):
        # SEM por eje = std / sqrt(n). Cuanto más bajo, más reproducible.
        n = max(1, centroids.shape[0])
        sem_x = float(std_position[0]) / (n ** 0.5)
        sem_y = float(std_position[1]) / (n ** 0.5)
        results["within_electrode_reliability"] = {
            "n_valid": int(centroids.shape[0]),
            "sem_x_px": sem_x,
            "sem_y_px": sem_y,
            "sem_x_deg": sem_x / max(self.pixels_per_degree_x, 1e-9),
            "sem_y_deg": sem_y / max(self.pixels_per_degree_y, 1e-9),
            "noise_floor_radius_deg": (
                (sem_x ** 2 + sem_y ** 2) ** 0.5
                / max((self.pixels_per_degree_x + self.pixels_per_degree_y) / 2.0, 1e-9)
            ),
        }

        # Imprimir resumen
        print(f"\n{'='*70}")
        print("RESULTADOS DEL ANÁLISIS")
        print(f"{'='*70}\n")
        print(f"Repeticiones válidas: {len(centroids)}/{self.num_repetitions}")
        print(f"Repeticiones inválidas: {num_invalid}")
        print(f"Posición promedio: ({mean_position[0]:.1f}, {mean_position[1]:.1f}) px")
        print(f"Desviación estándar: ({std_position[0]:.1f}, {std_position[1]:.1f}) px")
        print(
            f"Distancia media al promedio: {mean_distance:.1f} px ({mean_distance_deg:.2f}°)"
        )
        print(
            f"Distancia máxima al promedio: {max_distance:.1f} px ({max_distance_deg:.2f}°)"
        )

        offset = mean_position - stim_pos
        distance_to_stim = float(np.hypot(offset[0], offset[1]))
        distance_to_stim_deg = float(
            np.hypot(
                offset[0] / self.pixels_per_degree_x,
                offset[1] / self.pixels_per_degree_y,
            )
        )

        results["offset_mean_to_stimulus_px"] = {
            "x": float(offset[0]),
            "y": float(offset[1]),
        }
        results["offset_mean_to_stimulus_deg"] = {
            "x": float(offset[0] / self.pixels_per_degree_x),
            "y": float(offset[1] / self.pixels_per_degree_y),
        }
        results["distance_mean_to_stimulus_px"] = float(distance_to_stim)
        results["distance_mean_to_stimulus_deg"] = float(distance_to_stim_deg)

        print(f"\nPosición del estímulo: ({stim_pos[0]}, {stim_pos[1]}) px")
        print(
            f"Offset desde estímulo: ({offset[0]:.1f}, {offset[1]:.1f}) px ({offset[0]/self.pixels_per_degree_x:.2f}°, {offset[1]/self.pixels_per_degree_y:.2f}°)"
        )
        print(
            f"Distancia al estímulo: {distance_to_stim:.1f} px ({distance_to_stim_deg:.2f}°)"
        )

        # Guardar resultados
        results_file = self.electrode_dir / "analysis_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Resultados guardados en: {results_file.name}")

        # Exportar tabla por repetición (reproducible / fácil de meter en TFG)
        table_file = self.electrode_dir / "analysis_repetitions.csv"
        with open(table_file, "w", newline="", encoding="utf-8") as f:
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
            rep_features_by_number = {r["repetition_number"]: r for r in per_repetition}
            for m in results["per_repetition_metrics"]:
                rep_number = m["repetition_number"]
                rep_feat = rep_features_by_number.get(rep_number, {})
                centroid = rep_feat.get("centroid", {})
                centroid_deg = rep_feat.get("centroid_deg", {})
                writer.writerow(
                    {
                        "electrode_index": int(self.electrode_index),
                        "repetition_number": int(rep_number),
                        "centroid_x_px": float(centroid.get("x", float("nan"))),
                        "centroid_y_px": float(centroid.get("y", float("nan"))),
                        "centroid_x_deg": float(centroid_deg.get("x", float("nan"))),
                        "centroid_y_deg": float(centroid_deg.get("y", float("nan"))),
                        "n_pixels": int(rep_feat.get("n_pixels", 0)),
                        "fill_ratio": float(rep_feat.get("fill_ratio", 0.0)),
                        "dx_to_stim_px": float(m["dx_to_stim_px"]),
                        "dy_to_stim_px": float(m["dy_to_stim_px"]),
                        "distance_to_stim_px": float(m["distance_to_stim_px"]),
                        "dx_to_stim_deg": float(m["dx_to_stim_deg"]),
                        "dy_to_stim_deg": float(m["dy_to_stim_deg"]),
                        "distance_to_stim_deg": float(m["distance_to_stim_deg"]),
                    }
                )
        print(f"✓ Tabla por repetición guardada en: {table_file.name}")

        # Validate at the boundary: routing through ElectrodeAnalysisResult
        # stamps schema_version, normalises known fields, and preserves the
        # large amount of derived structure (boxplot_stats, ellipses, etc.)
        # via the extras escape hatch. Callers still get a dict — they don't
        # need to know about the dataclass yet.
        return ElectrodeAnalysisResult.from_dict(results).to_dict()

    def visualize_results(self, results, output_file="analysis_plot.png"):
        """
        Crea una visualización de los resultados del análisis

        Args:
            results: Diccionario de resultados del análisis
            output_file: Nombre del archivo de salida
        """
        if results is None:
            print("No hay resultados para visualizar")
            return

        # Crear figura
        fig, ax = plt.subplots(figsize=(10, 10))

        # Obtener datos
        centroids = np.array(results["centroids"])
        mean_pos = np.array(
            [results["mean_position"]["x"], results["mean_position"]["y"]]
        )
        stim_pos = np.array(results["stimulation_position"])
        screen_center = np.array(self.screen_center)  # (960, 540)

        # Configurar límites del gráfico con ZOOM AUTOMÁTICO
        # Calcular rango de datos
        std_x = results["std_position"]["x"]
        std_y = results["std_position"]["y"]

        # Margen dinámico basado en desviación estándar (3σ = 99.7% de los datos)
        margin = max(std_x, std_y) * 3.5  # 3.5 sigmas para ver bien el círculo

        # Rango del zoom
        x_min = mean_pos[0] - margin
        x_max = mean_pos[0] + margin
        y_min = mean_pos[1] - margin
        y_max = mean_pos[1] + margin

        # Asegurar que no sale de la pantalla
        x_min = max(0, x_min)
        x_max = min(self.screen_width, x_max)
        y_min = max(0, y_min)
        y_max = min(self.screen_height, y_max)

        # Aplicar zoom
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)  # Invertir Y para que crezca hacia abajo

        print(
            f"[PhospheneMappingAnalyzer] Auto-zoom: X:[{x_min:.0f}, {x_max:.0f}], Y:[{y_min:.0f}, {y_max:.0f}]"
        )
        print(f"                           Margen: {margin:.1f} px")

        # Dibujar centroides individuales
        ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            c="yellow",
            s=100,
            alpha=0.6,
            label="Dibujos individuales",
            edgecolors="orange",
            linewidths=2,
        )

        # Dibujar posición promedio
        ax.scatter(
            mean_pos[0],
            mean_pos[1],
            c="red",
            s=300,
            marker="X",
            label="Posición promedio",
            edgecolors="darkred",
            linewidths=2,
            zorder=10,
        )

        # Dibujar posición del estímulo
        ax.scatter(
            stim_pos[0],
            stim_pos[1],
            c="cyan",
            s=300,
            marker="*",
            label="Posición del estímulo",
            edgecolors="blue",
            linewidths=2,
            zorder=9,
        )

        # Líneas desde cada centroide a la media
        for centroid in centroids:
            ax.plot(
                [centroid[0], mean_pos[0]],
                [centroid[1], mean_pos[1]],
                "gray",
                alpha=0.3,
                linewidth=1,
            )

        # Círculo de desviación estándar
        std_x = results["std_position"]["x"]
        std_y = results["std_position"]["y"]
        std_radius = np.sqrt(std_x**2 + std_y**2)
        circle = plt.Circle(
            mean_pos, std_radius, color="red", fill=False, linestyle="--", linewidth=2
        )
        ax.add_patch(circle)

        # Elipse de dispersión (95%) y elipse de confianza de la media (95%)
        if centroids.shape[0] >= 2:
            cov = np.cov(centroids.T, ddof=1)
            params = _ellipse_from_cov(cov, confidence=0.95)
            if params is not None:
                width, height, angle = params
                ellipse = Ellipse(
                    xy=mean_pos,
                    width=width,
                    height=height,
                    angle=angle,
                    fill=False,
                    edgecolor="red",
                    linestyle="-",
                    linewidth=2,
                    alpha=0.8,
                    label="Elipse 95% (dispersión)",
                )
                ax.add_patch(ellipse)

                cov_mean = cov / float(centroids.shape[0])
                params_mean = _ellipse_from_cov(cov_mean, confidence=0.95)
                if params_mean is not None:
                    w_m, h_m, a_m = params_mean
                    ellipse_mean = Ellipse(
                        xy=mean_pos,
                        width=w_m,
                        height=h_m,
                        angle=a_m,
                        fill=False,
                        edgecolor="darkred",
                        linestyle=":",
                        linewidth=2,
                        alpha=0.9,
                        label="Elipse 95% (IC media)",
                    )
                    ax.add_patch(ellipse_mean)

        # Configuración
        ax.set_xlabel("X (píxeles)", fontsize=12)
        ax.set_ylabel("Y (píxeles)", fontsize=12)
        ax.set_title(
            f"Análisis de Mapeo - Electrodo {results['electrode_index']}\n"
            f"{results['num_valid_repetitions']} rep. válidas | {results['num_invalid_repetitions']} inválidas",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        # Calcular offset desde el estímulo
        offset = mean_pos - stim_pos
        pixels_per_degree_avg = (
            self.pixels_per_degree_x + self.pixels_per_degree_y
        ) / 2
        offset_deg_x = offset[0] / self.pixels_per_degree_x
        offset_deg_y = offset[1] / self.pixels_per_degree_y

        # Añadir texto con estadísticas
        stats_text = (
            f"Posición media: ({mean_pos[0]:.1f}, {mean_pos[1]:.1f}) px\n"
            f"Posición estímulo: ({stim_pos[0]:.1f}, {stim_pos[1]:.1f}) px\n"
            f"Desv. Est.: ({std_x:.1f}, {std_y:.1f}) px\n"
            f"Dist. media: {results['mean_distance_from_average']:.1f} px ({results['mean_distance_from_average_deg']:.2f}°)\n"
            f"Dist. máx.: {results['max_distance_from_average']:.1f} px ({results['max_distance_from_average_deg']:.2f}°)\n"
            f"Offset desde estímulo: ({offset[0]:.1f}, {offset[1]:.1f}) px\n"
            f"                       ({offset_deg_x:.2f}°, {offset_deg_y:.2f}°)"
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        # Guardar figura
        output_path = self.electrode_dir / output_file
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        # No cerrar automáticamente - dejar que el llamador lo maneje
        # plt.close()

        print(f"✓ Visualización guardada en: {output_file}")

        return fig  # Devolver la figura para poder mostrarla

    def visualize_boxplots(self, results, output_file="analysis_boxplots.png"):
        """Genera cajas y bigotes de error (por repetición) respecto al estímulo."""
        if results is None:
            return None

        metrics = results.get("per_repetition_metrics") or []
        if len(metrics) == 0:
            return None

        dx_deg = np.array([m["dx_to_stim_deg"] for m in metrics], dtype=float)
        dy_deg = np.array([m["dy_to_stim_deg"] for m in metrics], dtype=float)
        dr_deg = np.array([m["distance_to_stim_deg"] for m in metrics], dtype=float)

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))

        axes[0].boxplot(dx_deg, vert=True, showfliers=True)
        axes[0].set_title("Error X (°)")
        axes[0].set_ylabel("Grados visuales")
        axes[0].grid(True, alpha=0.3)

        axes[1].boxplot(dy_deg, vert=True, showfliers=True)
        axes[1].set_title("Error Y (°)")
        axes[1].grid(True, alpha=0.3)

        axes[2].boxplot(dr_deg, vert=True, showfliers=True)
        axes[2].set_title("Error radial (°)")
        axes[2].grid(True, alpha=0.3)

        fig.suptitle(
            f"Cajas y bigotes - Electrodo {results['electrode_index']}\n"
            f"n={results['num_valid_repetitions']} repeticiones válidas",
            fontsize=12,
            fontweight="bold",
        )

        output_path = self.electrode_dir / output_file
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"✓ Boxplots guardados en: {output_file}")
        return fig

    def show_results(self, results):
        """
        Muestra la visualización de resultados en pantalla de forma interactiva

        Args:
            results: Diccionario de resultados del análisis
        """
        if results is None:
            print("No hay resultados para mostrar")
            return

        # Crear la figura (sin guardarla automáticamente)
        fig, ax = plt.subplots(figsize=(12, 10))

        # Obtener datos
        centroids = np.array(results["centroids"])
        mean_pos = np.array(
            [results["mean_position"]["x"], results["mean_position"]["y"]]
        )
        stim_pos = np.array(results["stimulation_position"])

        # Configurar límites del gráfico para que el centro esté en (960, 540)
        ax.set_xlim(0, self.screen_width)
        ax.set_ylim(self.screen_height, 0)  # Invertir Y para que crezca hacia abajo

        # Dibujar centroides individuales
        ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            c="yellow",
            s=150,
            alpha=0.6,
            label="Dibujos individuales",
            edgecolors="orange",
            linewidths=2,
        )

        # Dibujar posición promedio
        ax.scatter(
            mean_pos[0],
            mean_pos[1],
            c="red",
            s=400,
            marker="X",
            label="Posición promedio",
            edgecolors="darkred",
            linewidths=3,
            zorder=10,
        )

        # Dibujar posición del estímulo
        ax.scatter(
            stim_pos[0],
            stim_pos[1],
            c="cyan",
            s=400,
            marker="*",
            label="Posición del estímulo",
            edgecolors="blue",
            linewidths=3,
            zorder=9,
        )

        # Líneas desde cada centroide a la media
        for centroid in centroids:
            ax.plot(
                [centroid[0], mean_pos[0]],
                [centroid[1], mean_pos[1]],
                "gray",
                alpha=0.3,
                linewidth=1,
            )

        # Círculo de desviación estándar
        std_x = results["std_position"]["x"]
        std_y = results["std_position"]["y"]
        std_radius = np.sqrt(std_x**2 + std_y**2)
        circle = plt.Circle(
            mean_pos, std_radius, color="red", fill=False, linestyle="--", linewidth=2
        )
        ax.add_patch(circle)

        # Configuración
        ax.set_xlabel("X (píxeles)", fontsize=14)
        ax.set_ylabel("Y (píxeles)", fontsize=14)
        ax.set_title(
            f"Análisis de Mapeo - Electrodo {results['electrode_index']}\n"
            f"{results['num_valid_repetitions']} rep. válidas | {results.get('num_invalid_repetitions', 0)} inválidas",
            fontsize=16,
            fontweight="bold",
        )
        ax.legend(loc="upper right", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        # Calcular offset desde el estímulo
        offset = mean_pos - stim_pos
        offset_deg_x = offset[0] / self.pixels_per_degree_x
        offset_deg_y = offset[1] / self.pixels_per_degree_y

        # Añadir texto con estadísticas
        stats_text = (
            f"Posición media: ({mean_pos[0]:.1f}, {mean_pos[1]:.1f}) px\n"
            f"Posición estímulo: ({stim_pos[0]:.1f}, {stim_pos[1]:.1f}) px\n"
            f"Desv. Est.: ({std_x:.1f}, {std_y:.1f}) px\n"
            f"Dist. media: {results['mean_distance_from_average']:.1f} px ({results.get('mean_distance_from_average_deg', 0):.2f}°)\n"
            f"Dist. máx.: {results['max_distance_from_average']:.1f} px ({results.get('max_distance_from_average_deg', 0):.2f}°)\n"
            f"Offset: ({offset[0]:.1f}, {offset[1]:.1f}) px ({offset_deg_x:.2f}°, {offset_deg_y:.2f}°)"
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.tight_layout()

        # Guardar figura sin mostrarla (para no interferir con Pygame)
        output_path = self.electrode_dir / "analysis_plot.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\n✓ Gráfico guardado en: analysis_plot.png")
        print("  (Abre visualmente después del experimento si lo necesitas)")

        # Cerrar figura para liberar memoria
        plt.close(fig)


def analyze_electrode(electrode_dir, visualize=True):
    """
    Función de conveniencia para analizar un electrodo

    Args:
        electrode_dir: Ruta a la carpeta del electrodo
        visualize: Si True, crea una visualización de los resultados

    Returns:
        dict: Resultados del análisis
    """
    analyzer = PhospheneMappingAnalyzer(electrode_dir)
    results = analyzer.analyze_electrode_repetitions()

    if results and visualize:
        analyzer.visualize_results(results)
        analyzer.visualize_boxplots(results)

    return results


if __name__ == "__main__":
    """
    Script standalone para analizar experimentos de mapeo
    Uso: python scripts/mapping_analyzer.py <ruta_carpeta_electrodo>
    """
    import sys

    if len(sys.argv) < 2:
        print("Uso: python scripts/mapping_analyzer.py <ruta_carpeta_electrodo>")
        print(
            "\nEjemplo: python scripts/mapping_analyzer.py mapping_experiments/mapping_test_20260310_120000/electrode_001"
        )
        sys.exit(1)

    electrode_dir = sys.argv[1]
    results = analyze_electrode(electrode_dir, visualize=True)

    if results:
        print("\n✓ Análisis completado exitosamente")
    else:
        print("\n✗ Error en el análisis")
        sys.exit(1)
