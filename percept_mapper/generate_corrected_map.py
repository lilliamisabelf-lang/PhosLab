"""
generate_corrected_map.py

Genera el mapa de fosfenos corregido aplicando los modelos entrenados
(bayesiano y/o neuronal) a las coordenadas del CSV de phosLab.

Uso:
    python generate_corrected_map.py --csv config/trial1.csv
    python generate_corrected_map.py --csv config/trial1.csv --model both
    python generate_corrected_map.py --csv config/trial1.csv --model bayesian

Salida en learning_results/:
    corrected_map_bayesian.csv
    corrected_map_neural.csv
    corrected_map_both.csv
    corrected_map_summary.json
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

LEARNING_DIR = Path("learning_results")
BAYESIAN_MODEL_PATH = LEARNING_DIR / "bayesian_model.json"
NEURAL_MODEL_PATH = LEARNING_DIR / "neural_model.pt"

CSV_COLUMNS = [
    "source_app",
    "dataset",
    "prf_source",
    "implant_id",
    "electrode_index",
    "x_deg",
    "y_deg",
    "polar_deg",
    "ecc_deg",
    "x_deg_corrected",
    "y_deg_corrected",
    "polar_deg_corrected",
    "ecc_deg_corrected",
    "correction_model",
]


def load_csv(csv_path: str) -> list[dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def xy_to_polar_ecc(x_deg: float, y_deg: float) -> tuple[float, float]:
    """Convierte (x, y) en grados a (polar_deg, ecc_deg)."""
    ecc = float(np.sqrt(x_deg**2 + y_deg**2))
    polar = float(np.degrees(np.arctan2(y_deg, x_deg))) % 360.0
    return polar, ecc


def apply_bayesian(rows: list[dict]) -> list[dict] | None:
    if not BAYESIAN_MODEL_PATH.exists():
        print(f"[WARN] Modelo bayesiano no encontrado: {BAYESIAN_MODEL_PATH}")
        return None

    from scripts.learning.bayesian_model import BayesianPhospheneCorrector

    model = BayesianPhospheneCorrector()
    model.load(str(BAYESIAN_MODEL_PATH))

    pred = np.array(
        [[float(r["x_deg"]), float(r["y_deg"])] for r in rows], dtype=np.float32
    )
    corrected, _ = model.correct_array(pred)

    result = []
    for i, row in enumerate(rows):
        cx, cy = float(corrected[i, 0]), float(corrected[i, 1])
        pol, ecc = xy_to_polar_ecc(cx, cy)
        result.append(
            {
                **row,
                "x_deg_corrected": f"{cx:.6f}",
                "y_deg_corrected": f"{cy:.6f}",
                "polar_deg_corrected": f"{pol:.6f}",
                "ecc_deg_corrected": f"{ecc:.6f}",
                "correction_model": "bayesian",
            }
        )
    print(f"[OK] Bayesiano aplicado a {len(result)} electrodos")
    return result


def apply_neural(rows: list[dict]) -> list[dict] | None:
    if not NEURAL_MODEL_PATH.exists():
        print(f"[WARN] Modelo neural no encontrado: {NEURAL_MODEL_PATH}")
        return None

    try:
        from scripts.learning.neural_model import NeuralPhospheneCorrector

        model = NeuralPhospheneCorrector()
        model.load(str(NEURAL_MODEL_PATH))
    except Exception as e:
        print(f"[WARN] No se pudo cargar el modelo neural: {e}")
        return None

    pred = np.array(
        [[float(r["x_deg"]), float(r["y_deg"])] for r in rows], dtype=np.float32
    )
    corrected = model.correct_array(pred)

    result = []
    for i, row in enumerate(rows):
        cx, cy = float(corrected[i, 0]), float(corrected[i, 1])
        pol, ecc = xy_to_polar_ecc(cx, cy)
        result.append(
            {
                **row,
                "x_deg_corrected": f"{cx:.6f}",
                "y_deg_corrected": f"{cy:.6f}",
                "polar_deg_corrected": f"{pol:.6f}",
                "ecc_deg_corrected": f"{ecc:.6f}",
                "correction_model": "neural",
            }
        )
    print(f"[OK] Neural aplicado a {len(result)} electrodos")
    return result


def save_corrected_csv(rows: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})
    print(f"[OK] CSV corregido guardado: {output_path}")


def generate_summary(
    rows_original: list[dict],
    rows_bayes: list[dict] | None,
    rows_neural: list[dict] | None,
    csv_name: str,
) -> dict:
    summary = {
        "source_csv": csv_name,
        "n_electrodes": len(rows_original),
        "implant_ids": list({r["implant_id"] for r in rows_original}),
        "models_applied": [],
        "original": {
            "x_mean": float(np.mean([float(r["x_deg"]) for r in rows_original])),
            "y_mean": float(np.mean([float(r["y_deg"]) for r in rows_original])),
            "ecc_mean": float(np.mean([float(r["ecc_deg"]) for r in rows_original])),
        },
    }

    if rows_bayes:
        summary["models_applied"].append("bayesian")
        cx = [float(r["x_deg_corrected"]) for r in rows_bayes]
        cy = [float(r["y_deg_corrected"]) for r in rows_bayes]
        ox = [float(r["x_deg"]) for r in rows_original]
        oy = [float(r["y_deg"]) for r in rows_original]
        diffs = np.sqrt(
            (np.array(cx) - np.array(ox)) ** 2 + (np.array(cy) - np.array(oy)) ** 2
        )
        summary["bayesian"] = {
            "x_mean_corrected": float(np.mean(cx)),
            "y_mean_corrected": float(np.mean(cy)),
            "mean_shift_deg": float(np.mean(diffs)),
            "max_shift_deg": float(np.max(diffs)),
        }

    if rows_neural:
        summary["models_applied"].append("neural")
        cx = [float(r["x_deg_corrected"]) for r in rows_neural]
        cy = [float(r["y_deg_corrected"]) for r in rows_neural]
        ox = [float(r["x_deg"]) for r in rows_original]
        oy = [float(r["y_deg"]) for r in rows_original]
        diffs = np.sqrt(
            (np.array(cx) - np.array(ox)) ** 2 + (np.array(cy) - np.array(oy)) ** 2
        )
        summary["neural"] = {
            "x_mean_corrected": float(np.mean(cx)),
            "y_mean_corrected": float(np.mean(cy)),
            "mean_shift_deg": float(np.mean(diffs)),
            "max_shift_deg": float(np.max(diffs)),
        }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Genera mapa de fosfenos corregido a partir de CSV de phosLab"
    )
    parser.add_argument("--csv", required=True, help="Ruta al CSV de phosLab")
    parser.add_argument(
        "--model", choices=["bayesian", "neural", "both"], default="both"
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[ERROR] CSV no encontrado: {csv_path}")
        return 1

    print("=" * 60)
    print("GENERANDO MAPA DE FOSFENOS CORREGIDO")
    print("=" * 60)
    print(f"CSV: {csv_path.name}")
    print(f"Modelo: {args.model}")
    print()

    rows = load_csv(str(csv_path))
    if not rows:
        print("[ERROR] CSV vacío o sin datos válidos")
        return 1

    print(f"Electrodos cargados: {len(rows)}")
    implant_ids = list({r["implant_id"] for r in rows})
    print(f"Implant IDs: {implant_ids}")
    print()

    rows_bayes = None
    rows_neural = None

    if args.model in ("bayesian", "both"):
        rows_bayes = apply_bayesian(rows)
        if rows_bayes:
            save_corrected_csv(rows_bayes, LEARNING_DIR / "corrected_map_bayesian.csv")

    if args.model in ("neural", "both"):
        rows_neural = apply_neural(rows)
        if rows_neural:
            save_corrected_csv(rows_neural, LEARNING_DIR / "corrected_map_neural.csv")

    # CSV combinado (bayesiano tiene prioridad, neural como alternativa)
    if rows_bayes and rows_neural:
        save_corrected_csv(rows_bayes, LEARNING_DIR / "corrected_map_both_bayesian.csv")
        save_corrected_csv(rows_neural, LEARNING_DIR / "corrected_map_both_neural.csv")

    summary = generate_summary(rows, rows_bayes, rows_neural, csv_path.name)
    summary_path = LEARNING_DIR / "corrected_map_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[OK] Resumen guardado: {summary_path}")

    print()
    print("=" * 60)
    print("MAPA CORREGIDO GENERADO")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
