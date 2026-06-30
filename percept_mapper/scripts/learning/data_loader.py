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
        input_mode=None,
    ):
        self.mapping_dir = Path(mapping_dir)
        self.logs_dir = Path(logs_dir)
        self.mapping_experiments = (
            None if mapping_experiments is None else set(mapping_experiments)
        )
        self.logs_experiments = (
            None if logs_experiments is None else set(logs_experiments)
        )
        # Filtro por modo de entrada de la sesión (pupil/gaze/mouse).
        # None o 'all' => sin filtro.
        self.input_mode = None if input_mode in (None, "all") else str(input_mode)
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

            # Sesión PAREADA (mapping_method: paired): un único 'pairs/' con
            # metadata.json en vez de carpetas por electrodo. Se reconstruye la
            # posición absoluta por electrodo con MDS y se emiten filas pred→obs.
            pairs_dir = exp_dir / "pairs"
            if (pairs_dir / "metadata.json").exists():
                try:
                    count += self._append_rows(
                        self._extract_paired_rows(pairs_dir, exp_dir.name)
                    )
                except Exception as e:
                    print(f"  WARN: Error leyendo {pairs_dir}: {e}")
                continue

            # Buscar carpetas de electrodos. Dos esquemas de nombrado (ver
            # scripts/phosphene_mapping.py:_electrode_dir_name):
            #   - un solo implante  -> electrode_001 (histórico)
            #   - varios implantes  -> impA_electrode50 (implante + índice local)
            # El electrode_index real se lee del contenido del JSON, no del
            # nombre, así que basta con dejar entrar ambos.
            for electrode_dir in sorted(exp_dir.iterdir()):
                if not electrode_dir.is_dir():
                    continue
                name = electrode_dir.name
                is_electrode_dir = name.startswith("electrode_") or "_electrode" in name
                if not is_electrode_dir:
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

        # Filtro por modo de entrada de la sesión.
        if self.input_mode and metadata.get("input_mode") != self.input_mode:
            return rows

        # Electrodos sin respuesta (status no_response) no aportan centroides.
        if results.get("status") == "no_response":
            return rows

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

    def _extract_paired_rows(self, pairs_dir, experiment_id):
        """Reconstruye una posición observada ABSOLUTA por electrodo desde una
        sesión pareada y emite filas pred→obs (mismo esquema que mapping).

        Cómo: el dato pareado es relativo (cada ensayo da Δ(A→B) = pos_B − pos_A
        en grados, de los dos extremos trazados). Con MDS sobre |Δ| recuperamos
        la geometría hasta una similitud, y la llevamos al marco absoluto en
        grados alineándola por Procrustes sobre las posiciones PREDICHAS (atlas)
        de los electrodos — las mismas `visual_position_deg` que en el modo
        absoluto. La salida `obs` es esa reconstrucción alineada; `pred` es la
        predicha; `error = obs − pred`. Reutiliza scripts/relative_map.py y la
        carga de scripts/analysis/build_relative_map.py:PairedSession.
        """
        # Imports diferidos: solo se necesitan para sesiones pareadas y arrastran
        # scipy/PairedSession; mantenerlos aquí no penaliza el caso absoluto.
        import sys as _sys
        from pathlib import Path as _Path

        _root = _Path(__file__).resolve().parents[2]  # percept_mapper/
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        from scripts.analysis.build_relative_map import PairedSession
        from scripts.relative_map import embed_mds, align_procrustes

        sess = PairedSession(_Path(pairs_dir))

        # Filtro por modo de entrada (a nivel de sesión, igual que el resto).
        if self.input_mode and sess.meta.get("input_mode") != self.input_mode:
            return []

        if not sess.edges or sess.n_nodes < 1:
            return []

        # Posiciones PREDICHAS (atlas) e info por electrode_index, leídas de los
        # electrode_info de cada par. Sirven de objetivo de alineación y de `pred`.
        pred_deg = {}
        ecc_deg = {}
        implant_by_e = {}
        for t in sess.meta.get("trials", []):
            if t.get("is_practice"):
                continue
            for key in ("electrode_info_a", "electrode_info_b"):
                info = t.get(key) or {}
                e = info.get("index")
                vp = info.get("visual_position_deg")
                if e is None or not vp or vp[0] is None or vp[1] is None:
                    continue
                e = int(e)
                if e not in pred_deg:
                    pred_deg[e] = (float(vp[0]), float(vp[1]))
                    ecc_deg[e] = info.get("eccentricity_deg")
                    implant_by_e[e] = info.get("implant_id", "unknown")

        # Anclas de alineación: todos los electrodos con predicción, por NODO.
        # Procrustes ajusta la similitud que mejor lleva la geometría MDS sobre
        # las posiciones predichas (≥3 anclas no colineales para fijar rotación).
        anchors = {
            node: pred_deg[e]
            for node, e in enumerate(sess.node_to_electrode)
            if e in pred_deg
        }
        if len(anchors) < 3:
            print(
                f"  WARN: sesión pareada {experiment_id}: solo {len(anchors)} "
                "electrodos con predicción; MDS necesita ≥3 anclas para orientar. "
                "Se omite."
            )
            return []

        mds_est, mds_info = embed_mds(
            sess.edges, sess.distances_obs, sess.n_nodes, method="smacof",
            n_init=4, seed=0,
        )
        try:
            obs_coords, _ = align_procrustes(
                mds_est, anchors, allow_scale=True, allow_reflection=True
            )
        except ValueError:
            return []

        rows = []
        for node, e in enumerate(sess.node_to_electrode):
            if e not in pred_deg:
                continue
            obs = obs_coords[node]
            if obs is None or np.any(np.isnan(obs)):
                continue
            px, py = pred_deg[e]
            ox, oy = float(obs[0]), float(obs[1])
            ecc = ecc_deg.get(e)
            rows.append(
                {
                    "electrode_index": e,
                    "implant_id": implant_by_e.get(e, "unknown"),
                    "experiment_mode": "mapping",
                    "experiment_id": experiment_id,
                    "pred_x_deg": px,
                    "pred_y_deg": py,
                    "obs_x_deg": ox,
                    "obs_y_deg": oy,
                    "error_x_deg": ox - px,
                    "error_y_deg": oy - py,
                    "error_radial_deg": float(np.hypot(ox - px, oy - py)),
                    "current_uA": None,
                    "eccentricity_deg": float(ecc) if ecc else None,
                    "repetition_index": 0,
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

            # Filtro por modo de entrada: el modo se guarda a nivel de sesión.
            if self.input_mode:
                exp_meta_file = exp_dir / "metadata.json"
                exp_input_mode = None
                if exp_meta_file.exists():
                    try:
                        with open(exp_meta_file, "r", encoding="utf-8") as f:
                            exp_input_mode = json.load(f).get("input_mode")
                    except Exception:
                        exp_input_mode = None
                if exp_input_mode != self.input_mode:
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
