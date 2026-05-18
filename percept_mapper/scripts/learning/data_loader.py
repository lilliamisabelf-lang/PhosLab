"""
data_loader.py

Lee experimentos de mapping_experiments/ y logs/ y construye
un dataset unificado para el módulo de aprendizaje.

Cada fila del dataset representa una observación:
    pred_x_deg, pred_y_deg  → predicción del modelo (phosLab)
    obs_x_deg, obs_y_deg    → observación del usuario (centroide)
    error_x_deg             → obs_x - pred_x
    error_y_deg             → obs_y - pred_y
    electrode_index
    implant_id
    experiment_mode         → 'mapping' o 'standard'
    experiment_id
    current_uA              → corriente de estimulación
    eccentricity_deg        → excentricidad del electrodo
"""

import json
from pathlib import Path
import numpy as np


class PhospheneDataLoader:
    """
    Carga y unifica datos de experimentos de mapeo (mapping y standard).
    """

    def __init__(
        self,
        mapping_dir="mapping_experiments",
        logs_dir="logs",
        mapping_experiments=None,
        logs_experiments=None,
    ):
        self.mapping_dir = Path(mapping_dir)
        self.logs_dir = Path(logs_dir)
        self.mapping_experiments = (
            None if mapping_experiments is None else set(mapping_experiments)
        )
        self.logs_experiments = (
            None if logs_experiments is None else set(logs_experiments)
        )
        self.dataset = []
        self._seen = set()

    def load_all(self):
        """Carga todos los experimentos disponibles."""
        self.dataset = []
        self._seen = set()
        n_mapping = self._load_mapping_experiments()
        n_standard = self._load_standard_experiments()
        print(f"\n[DataLoader] Dataset cargado:")
        print(f"             Mapping:  {n_mapping} observaciones")
        print(f"             Standard: {n_standard} observaciones")
        print(f"             Total:    {len(self.dataset)} observaciones")
        return self.dataset

    def _row_key(self, row: dict) -> tuple:
        return (
            row.get("experiment_mode"),
            row.get("experiment_id"),
            (
                int(row.get("electrode_index"))
                if row.get("electrode_index") is not None
                else None
            ),
            (
                int(row.get("repetition_index"))
                if row.get("repetition_index") is not None
                else None
            ),
            round(float(row.get("pred_x_deg", 0.0)), 6),
            round(float(row.get("pred_y_deg", 0.0)), 6),
            round(float(row.get("obs_x_deg", 0.0)), 6),
            round(float(row.get("obs_y_deg", 0.0)), 6),
        )

    def _append_rows(self, rows: list[dict]) -> int:
        added = 0
        for row in rows:
            key = self._row_key(row)
            if key in self._seen:
                continue
            self._seen.add(key)
            self.dataset.append(row)
            added += 1
        return added

    def _load_mapping_experiments(self):
        """Carga experimentos de mapping_experiments/."""
        if not self.mapping_dir.exists():
            return 0

        count = 0
        for exp_dir in sorted(self.mapping_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            if (
                self.mapping_experiments
                and exp_dir.name not in self.mapping_experiments
            ):
                continue

            # Buscar carpetas de electrodos
            for electrode_dir in sorted(exp_dir.iterdir()):
                if not electrode_dir.is_dir():
                    continue
                if not electrode_dir.name.startswith("electrode_"):
                    continue

                results_file = electrode_dir / "analysis_results.json"
                metadata_file = electrode_dir / "metadata.json"

                if not results_file.exists() or not metadata_file.exists():
                    continue

                try:
                    with open(results_file, "r", encoding="utf-8") as f:
                        results = json.load(f)
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        metadata = json.load(f)

                    rows = self._extract_mapping_rows(results, metadata, exp_dir.name)
                    count += self._append_rows(rows)

                except Exception as e:
                    print(f"  WARN: Error leyendo {electrode_dir}: {e}")

        return count

    def _extract_mapping_rows(self, results, metadata, experiment_id):
        """Extrae filas del dataset desde un electrodo de mapping."""
        rows = []

        electrode_index = results.get("electrode_index")
        electrode_info = metadata.get("electrode_info", {})
        pred_deg = electrode_info.get("visual_position_deg", [None, None])
        eccentricity = electrode_info.get("eccentricity_deg", None)
        implant_id = electrode_info.get("implant_id", "unknown")
        current_uA = metadata.get("stimulation_parameters", {}).get("current_uA", None)

        if pred_deg[0] is None or pred_deg[1] is None:
            return rows

        # Una fila por centroide individual
        centroids_deg = results.get("centroids_deg", [])
        for i, centroid_deg in enumerate(centroids_deg):
            obs_x = centroid_deg[0]
            obs_y = centroid_deg[1]
            rows.append(
                {
                    "electrode_index": electrode_index,
                    "implant_id": implant_id,
                    "experiment_mode": "mapping",
                    "experiment_id": experiment_id,
                    "pred_x_deg": float(pred_deg[0]),
                    "pred_y_deg": float(pred_deg[1]),
                    "obs_x_deg": float(obs_x),
                    "obs_y_deg": float(obs_y),
                    "error_x_deg": float(obs_x) - float(pred_deg[0]),
                    "error_y_deg": float(obs_y) - float(pred_deg[1]),
                    "error_radial_deg": float(
                        np.hypot(
                            float(obs_x) - float(pred_deg[0]),
                            float(obs_y) - float(pred_deg[1]),
                        )
                    ),
                    "current_uA": float(current_uA) if current_uA is not None else None,
                    "eccentricity_deg": float(eccentricity) if eccentricity else None,
                    "repetition_index": i,
                }
            )

        return rows

    def _load_standard_experiments(self):
        """Carga experimentos de logs/ (modo standard)."""
        if not self.logs_dir.exists():
            return 0

        count = 0
        for exp_dir in sorted(self.logs_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            if self.logs_experiments and exp_dir.name not in self.logs_experiments:
                continue

            analysis_dir = exp_dir / "analysis"
            if not analysis_dir.exists():
                continue

            for electrode_dir in sorted(analysis_dir.iterdir()):
                if not electrode_dir.is_dir():
                    continue
                if not electrode_dir.name.startswith("electrode_"):
                    continue

                results_file = electrode_dir / "analysis_results.json"
                if not results_file.exists():
                    continue

                try:
                    with open(results_file, "r", encoding="utf-8") as f:
                        results = json.load(f)

                    rows = self._extract_standard_rows(results, exp_dir.name)
                    count += self._append_rows(rows)

                except Exception as e:
                    print(f"  WARN: Error leyendo {electrode_dir}: {e}")

        return count

    def _extract_standard_rows(self, results, experiment_id):
        """Extrae filas del dataset desde un electrodo de standard."""
        rows = []

        electrode_index = results.get("electrode_index")
        pred_x = results.get("pred_x_deg")
        pred_y = results.get("pred_y_deg")

        if pred_x is None or pred_y is None:
            return rows

        centroids_deg = results.get("centroids_deg", [])
        for i, centroid_deg in enumerate(centroids_deg):
            obs_x = centroid_deg[0]
            obs_y = centroid_deg[1]
            rows.append(
                {
                    "electrode_index": electrode_index,
                    "implant_id": "unknown",
                    "experiment_mode": "standard",
                    "experiment_id": experiment_id,
                    "pred_x_deg": float(pred_x),
                    "pred_y_deg": float(pred_y),
                    "obs_x_deg": float(obs_x),
                    "obs_y_deg": float(obs_y),
                    "error_x_deg": float(obs_x) - float(pred_x),
                    "error_y_deg": float(obs_y) - float(pred_y),
                    "error_radial_deg": float(
                        np.hypot(
                            float(obs_x) - float(pred_x), float(obs_y) - float(pred_y)
                        )
                    ),
                    "current_uA": None,
                    "eccentricity_deg": None,
                    "repetition_index": i,
                }
            )

        return rows

    def get_arrays(self):
        """
        Devuelve arrays numpy para usar directamente en los modelos.

        Returns:
            pred:  (N, 2) predicciones [x_deg, y_deg]
            obs:   (N, 2) observaciones [x_deg, y_deg]
            error: (N, 2) errores [error_x, error_y]
        """
        if not self.dataset:
            raise ValueError("Dataset vacío. Llama a load_all() primero.")

        pred = np.array([[r["pred_x_deg"], r["pred_y_deg"]] for r in self.dataset])
        obs = np.array([[r["obs_x_deg"], r["obs_y_deg"]] for r in self.dataset])
        error = obs - pred

        return pred, obs, error

    def summary(self):
        """Imprime un resumen del dataset."""
        if not self.dataset:
            print("Dataset vacío.")
            return

        n = len(self.dataset)
        errors_x = np.array([r["error_x_deg"] for r in self.dataset])
        errors_y = np.array([r["error_y_deg"] for r in self.dataset])
        errors_r = np.array([r["error_radial_deg"] for r in self.dataset])

        electrodes = set(r["electrode_index"] for r in self.dataset)
        experiments = set(r["experiment_id"] for r in self.dataset)

        print(f"\n{'='*60}")
        print(f"RESUMEN DEL DATASET")
        print(f"{'='*60}")
        print(f"Observaciones totales: {n}")
        print(f"Electrodos únicos:     {len(electrodes)}")
        print(f"Experimentos únicos:   {len(experiments)}")
        print(
            f"\nError X — media: {np.mean(errors_x):.3f}°  std: {np.std(errors_x):.3f}°"
        )
        print(
            f"Error Y — media: {np.mean(errors_y):.3f}°  std: {np.std(errors_y):.3f}°"
        )
        print(
            f"Error radial — media: {np.mean(errors_r):.3f}°  max: {np.max(errors_r):.3f}°"
        )
        print(f"{'='*60}\n")

    def get_dataset_summary(self) -> dict:
        summary = {
            "total_observations": len(self.dataset),
            "mapping_observations": 0,
            "standard_observations": 0,
            "experiments": {},
            "electrodes": {},
        }

        for row in self.dataset:
            mode = row.get("experiment_mode", "unknown")
            exp_id = row.get("experiment_id", "unknown")
            electrode = row.get("electrode_index")

            if mode == "mapping":
                summary["mapping_observations"] += 1
            elif mode == "standard":
                summary["standard_observations"] += 1

            exp_entry = summary["experiments"].setdefault(
                exp_id,
                {
                    "mode": mode,
                    "observations": 0,
                    "electrodes": {},
                },
            )
            exp_entry["observations"] += 1
            if electrode is not None:
                exp_entry["electrodes"].setdefault(str(electrode), 0)
                exp_entry["electrodes"][str(electrode)] += 1

            if electrode is not None:
                summary["electrodes"].setdefault(str(electrode), 0)
                summary["electrodes"][str(electrode)] += 1

        return summary
