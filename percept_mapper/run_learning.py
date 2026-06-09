"""
run_learning.py

Script de entrada para el módulo de aprendizaje.

Uso:
    python run_learning.py --model bayesian
    python run_learning.py --model neural
    python run_learning.py --model both
"""

import argparse
import json
from pathlib import Path
import yaml
import numpy as np

from scripts.learning.data_loader import PhospheneDataLoader
from scripts.learning.bayesian_model import BayesianPhospheneCorrector
from scripts.learning.model_evaluator import ModelEvaluator
from scripts.learning.cross_validation import k_fold_cv, bayesian_corrector

try:
    from scripts.learning.neural_model import NeuralPhospheneCorrector

    _NEURAL_AVAILABLE = True
    _NEURAL_IMPORT_ERROR = None
except Exception as exc:
    NeuralPhospheneCorrector = None
    _NEURAL_AVAILABLE = False
    _NEURAL_IMPORT_ERROR = exc


def _list_experiments(path: Path) -> list[str]:
    if not path.exists():
        return []
    return sorted([d.name for d in path.iterdir() if d.is_dir()])


def _get_latest_experiment(path: Path):
    if not path.exists():
        return None
    dirs = [d for d in path.iterdir() if d.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda d: d.stat().st_mtime)


def _scope_filters(scope: str, mapping_names: list[str], logs_names: list[str]):
    mapping_filter: list[str] = []
    logs_filter: list[str] = []

    if scope == "all":
        mapping_filter = list(mapping_names)
        logs_filter = list(logs_names)
    elif scope == "latest_mapping":
        mapping_filter = [mapping_names[-1]] if mapping_names else []
        logs_filter = []
    elif scope == "latest_standard":
        mapping_filter = []
        logs_filter = [logs_names[-1]] if logs_names else []
    elif scope == "latest_any":
        latest_mapping = mapping_names[-1] if mapping_names else None
        latest_logs = logs_names[-1] if logs_names else None
        if latest_mapping and latest_logs:
            mapping_filter = [latest_mapping]
            logs_filter = []
        elif latest_mapping:
            mapping_filter = [latest_mapping]
        elif latest_logs:
            logs_filter = [latest_logs]

    return mapping_filter, logs_filter


def _resolve_test_experiment(
    args,
    mapping_path: Path,
    logs_path: Path,
    mapping_names: list[str],
    logs_names: list[str],
):
    if args.test_mode == "none":
        return None, None

    if args.test_mode == "select":
        if not args.test_experiment:
            return None, None
        source = args.test_source
        if source == "any":
            if args.test_experiment in mapping_names:
                source = "mapping"
            elif args.test_experiment in logs_names:
                source = "standard"
            else:
                return None, None
        return source, args.test_experiment

    source = args.test_source
    if source == "mapping":
        latest = _get_latest_experiment(mapping_path)
        return ("mapping", latest.name) if latest else (None, None)
    if source == "standard":
        latest = _get_latest_experiment(logs_path)
        return ("standard", latest.name) if latest else (None, None)

    latest_mapping = _get_latest_experiment(mapping_path)
    latest_logs = _get_latest_experiment(logs_path)
    if latest_mapping and latest_logs:
        if latest_mapping.stat().st_mtime >= latest_logs.stat().st_mtime:
            return "mapping", latest_mapping.name
        return "standard", latest_logs.name
    if latest_mapping:
        return "mapping", latest_mapping.name
    if latest_logs:
        return "standard", latest_logs.name
    return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Módulo de aprendizaje para corrección de fosfenos"
    )
    parser.add_argument(
        "--model",
        choices=["bayesian", "neural", "both"],
        default="both",
        help="Modelo a usar (default: both)",
    )
    parser.add_argument(
        "--config",
        default="config/params.yaml",
        help="Ruta al params.yaml (default: config/params.yaml)",
    )
    parser.add_argument(
        "--scope",
        choices=["all", "latest_mapping", "latest_standard", "latest_any"],
        default="all",
        help="Conjunto de experimentos a usar (default: all)",
    )
    parser.add_argument(
        "--input-mode",
        choices=["all", "pupil", "gaze", "mouse"],
        default="all",
        help="Filtra observaciones por modo de entrada de la sesión (default: all)",
    )
    parser.add_argument(
        "--test-mode",
        choices=["none", "last", "select"],
        default="none",
        help="Modo de test (default: none)",
    )
    parser.add_argument(
        "--test-source",
        choices=["any", "mapping", "standard"],
        default="any",
        help="Origen del test (default: any)",
    )
    parser.add_argument(
        "--test-experiment",
        default="",
        help="Nombre del experimento de test (solo con test-mode=select)",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="K-fold CV para el modelo bayesiano. 0 desactiva CV (default: 5).",
    )
    parser.add_argument(
        "--cv-bootstrap",
        type=int,
        default=2000,
        help="Iteraciones de bootstrap para p-value (default: 2000)",
    )
    args = parser.parse_args()

    # Cargar configuración
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    learning_cfg = config.get("learning", {})
    mapping_dir = learning_cfg.get("mapping_dir", "mapping_experiments")
    logs_dir = learning_cfg.get("logs_dir", "logs")
    output_dir = learning_cfg.get("output_dir", "learning_results")

    bayes_cfg = learning_cfg.get("bayesian", {})
    neural_cfg = learning_cfg.get("neural", {})

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MÓDULO DE APRENDIZAJE — CORRECCIÓN DE FOSFENOS")
    print("=" * 60)
    print(f"Modelo seleccionado: {args.model}")
    print(f"Scope dataset (train): {args.scope}")
    print(f"Modo de entrada (datos): {args.input_mode}")
    print(f"Test mode: {args.test_mode}")

    mapping_path = Path(mapping_dir)
    logs_path = Path(logs_dir)
    mapping_names = _list_experiments(mapping_path)
    logs_names = _list_experiments(logs_path)

    mapping_filter, logs_filter = _scope_filters(args.scope, mapping_names, logs_names)

    test_source, test_name = _resolve_test_experiment(
        args, mapping_path, logs_path, mapping_names, logs_names
    )
    if test_source and test_name:
        print(f"Test experiment: {test_source}:{test_name}")
    elif args.test_mode != "none":
        print("WARN: No se pudo resolver experimento de test.")

    if test_source == "mapping" and test_name:
        mapping_filter = [n for n in mapping_filter if n != test_name]
    elif test_source == "standard" and test_name:
        logs_filter = [n for n in logs_filter if n != test_name]

    # Cargar datos
    loader = PhospheneDataLoader(
        mapping_dir=mapping_dir,
        logs_dir=logs_dir,
        mapping_experiments=mapping_filter,
        logs_experiments=logs_filter,
        input_mode=args.input_mode,
    )
    dataset = loader.load_all()

    if not dataset:
        print("\nERROR: No se encontraron datos de experimentos.")
        print("  Asegúrate de tener experimentos en mapping_experiments/ o logs/")
        return

    loader.summary()
    train_summary = loader.get_dataset_summary()
    pred, obs, error = loader.get_arrays()

    test_pred = None
    test_obs = None
    test_error = None
    test_summary = None
    if test_source and test_name:
        test_mapping_filter = [test_name] if test_source == "mapping" else []
        test_logs_filter = [test_name] if test_source == "standard" else []
        test_loader = PhospheneDataLoader(
            mapping_dir=mapping_dir,
            logs_dir=logs_dir,
            mapping_experiments=test_mapping_filter,
            logs_experiments=test_logs_filter,
            input_mode=args.input_mode,
        )
        test_dataset = test_loader.load_all()
        if test_dataset:
            test_summary = test_loader.get_dataset_summary()
            test_pred, test_obs, test_error = test_loader.get_arrays()
        else:
            print("WARN: El dataset de test esta vacio.")

    evaluator = ModelEvaluator(output_dir=output_dir)

    corrected_bayes = None
    corrected_neural = None
    corrected_bayes_train = None
    corrected_neural_train = None
    bayes_model = None
    neural_model = None

    # ── BAYESIANO ──────────────────────────────────────────
    if args.model in ("bayesian", "both"):
        print("\n" + "=" * 60)
        print("MODELO BAYESIANO")
        print("=" * 60)

        bayes_model = BayesianPhospheneCorrector(
            prior_mean=bayes_cfg.get("prior_mean", 0.0),
            prior_std=bayes_cfg.get("prior_std", 5.0),
            noise_std=bayes_cfg.get("noise_std", 0.5),
        )
        bayes_model.fit(error[:, 0], error[:, 1])
        corrected_bayes_train, _ = bayes_model.correct_array(pred)

        bayes_model.save(output_path / "bayesian_model.json")

    # ── RED NEURONAL ───────────────────────────────────────
    if args.model in ("neural", "both"):
        if not _NEURAL_AVAILABLE:
            print("\n" + "=" * 60)
            print("RED NEURONAL")
            print("=" * 60)
            print("ERROR: Modelo neural no disponible en este entorno.")
            print(f"  Causa: {_NEURAL_IMPORT_ERROR}")
            if args.model == "neural":
                return
            print("  Continuando solo con el modelo bayesiano.")
        else:
            print("\n" + "=" * 60)
            print("RED NEURONAL")
            print("=" * 60)

            neural_model = NeuralPhospheneCorrector(
                hidden_size=neural_cfg.get("hidden_size", 32),
                learning_rate=neural_cfg.get("learning_rate", 0.01),
                num_epochs=neural_cfg.get("num_epochs", 500),
            )
            trained = neural_model.fit(
                pred,
                obs,
                train_split=neural_cfg.get("train_split", 0.8),
            )

            if trained:
                corrected_neural_train = neural_model.correct_array(pred)
                neural_model.save(output_path / "neural_model.pt")

                if neural_model.train_losses:
                    evaluator.plot_neural_training(
                        neural_model.train_losses,
                        neural_model.val_losses,
                    )

    # ── EVALUACIÓN ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUACIÓN")
    print("=" * 60)

    metrics_train = evaluator.evaluate(
        pred,
        obs,
        corrected_bayes=corrected_bayes_train,
        corrected_neural=corrected_neural_train,
    )

    eval_pred = test_pred if test_pred is not None else pred
    eval_obs = test_obs if test_obs is not None else obs
    corrected_bayes = None
    corrected_neural = None

    if bayes_model:
        corrected_bayes, _ = bayes_model.correct_array(eval_pred)
    if neural_model and neural_model.is_trained:
        corrected_neural = neural_model.correct_array(eval_pred)

    metrics_eval = evaluator.evaluate(
        eval_pred,
        eval_obs,
        corrected_bayes=corrected_bayes,
        corrected_neural=corrected_neural,
    )

    print("\nTRAIN")
    evaluator.print_summary(metrics_train)
    if test_pred is not None:
        print("\nTEST")
    evaluator.print_summary(metrics_eval)

    evaluator.plot_visual_field(
        eval_pred,
        eval_obs,
        corrected_bayes=corrected_bayes,
        corrected_neural=corrected_neural,
    )
    evaluator.plot_error_comparison(metrics_eval)

    # ── CROSS-VALIDATION (k-fold) ─────────────────────────
    # Reemplaza el single-holdout `train_split: 0.8`. Cada fold deja fuera
    # 1/K del dataset, ajusta el corrector bayesiano en el resto, y mide
    # error en held-out. El paired bootstrap sobre los errores por-trial
    # da un CI 95% y un p-value de "corregido < no corregido".
    cv_payload = None
    if args.cv and args.cv > 1 and pred.shape[0] >= args.cv:
        print("\n" + "=" * 60)
        print(f"CROSS-VALIDATION ({args.cv}-fold, bayesiano)")
        print("=" * 60)
        cv_result = k_fold_cv(
            pred, obs,
            fit_correct_fn=bayesian_corrector(
                prior_mean=bayes_cfg.get("prior_mean", 0.0),
                prior_std=bayes_cfg.get("prior_std", 5.0),
                noise_std=bayes_cfg.get("noise_std", 0.5),
            ),
            k=int(args.cv),
            seed=int(bayes_cfg.get("cv_seed", 0)),
            bootstrap_iter=int(args.cv_bootstrap),
        )
        print(
            f"  MSE uncorrected (mean across folds): {cv_result.mean_mse_uncorrected:.3f}"
            f" ± {cv_result.std_mse_uncorrected:.3f}"
        )
        print(
            f"  MSE corrected   (mean across folds): {cv_result.mean_mse_corrected:.3f}"
            f" ± {cv_result.std_mse_corrected:.3f}"
        )
        print(
            f"  Improvement: {cv_result.improvement_abs:.3f} ({cv_result.improvement_pct:.1f}%)"
        )
        print(
            f"  Bootstrap p={cv_result.bootstrap_p_value:.4f}  "
            f"95% CI=({cv_result.bootstrap_ci_low:.3f}, {cv_result.bootstrap_ci_high:.3f})"
        )
        if cv_result.bootstrap_p_value is not None and cv_result.bootstrap_p_value < 0.05:
            print("   Mejora estadísticamente significativa al 5%.")
        else:
            print("   Mejora NO estadísticamente significativa al 5%.")
        cv_payload = cv_result.to_dict()
    elif args.cv:
        print(f"\nWARN: dataset (N={pred.shape[0]}) demasiado pequeño para {args.cv}-fold CV; CV omitido.")

    # Guardar métricas
    metrics_file = output_path / "evaluation_metrics.json"
    metrics_payload = {
        "train": metrics_train,
        "test": metrics_eval,
        "cross_validation": cv_payload,
    }
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)
    print(f"\nOK: Métricas guardadas en: {metrics_file}")

    summary_file = output_path / "dataset_summary.json"
    dataset_summary = {
        "scope": args.scope,
        "test_mode": args.test_mode,
        "test_source": test_source,
        "test_experiment": test_name,
        "train": train_summary,
        "test": test_summary,
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(dataset_summary, f, indent=2, ensure_ascii=False)
    print(f"OK: Resumen de dataset guardado en: {summary_file}")

    print("\n" + "=" * 60)
    print(f"OK: Resultados en: {output_path}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
