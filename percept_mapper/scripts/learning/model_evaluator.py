"""
model_evaluator.py

Compara las predicciones originales, corregidas por Bayes
y corregidas por red neuronal. Genera visualizaciones y métricas.
"""

from pathlib import Path
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ModelEvaluator:
    """
    Evalúa y compara modelos de corrección de fosfenos.
    """

    def __init__(self, output_dir="learning_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _radial_error(self, pred, obs):
        """Error radial en grados."""
        return np.sqrt(np.sum((pred - obs) ** 2, axis=1))

    def evaluate(self, pred, obs, corrected_bayes=None, corrected_neural=None):
        """
        Calcula métricas de error para cada modelo.

        Args:
            pred:             (N, 2) predicciones originales
            obs:              (N, 2) observaciones
            corrected_bayes:  (N, 2) predicciones corregidas por Bayes (opcional)
            corrected_neural: (N, 2) predicciones corregidas por red neuronal (opcional)

        Returns:
            dict con métricas por modelo
        """
        pred = np.array(pred)
        obs = np.array(obs)

        results = {}

        # Modelo original (sin corrección)
        err_orig = self._radial_error(pred, obs)
        results["original"] = {
            "mean_error_deg": float(np.mean(err_orig)),
            "std_error_deg": float(np.std(err_orig)),
            "median_error_deg": float(np.median(err_orig)),
            "max_error_deg": float(np.max(err_orig)),
            "mean_error_x_deg": float(np.mean(np.abs(pred[:, 0] - obs[:, 0]))),
            "mean_error_y_deg": float(np.mean(np.abs(pred[:, 1] - obs[:, 1]))),
        }

        if corrected_bayes is not None:
            corrected_bayes = np.array(corrected_bayes)
            err_bayes = self._radial_error(corrected_bayes, obs)
            results["bayesian"] = {
                "mean_error_deg": float(np.mean(err_bayes)),
                "std_error_deg": float(np.std(err_bayes)),
                "median_error_deg": float(np.median(err_bayes)),
                "max_error_deg": float(np.max(err_bayes)),
                "mean_error_x_deg": float(
                    np.mean(np.abs(corrected_bayes[:, 0] - obs[:, 0]))
                ),
                "mean_error_y_deg": float(
                    np.mean(np.abs(corrected_bayes[:, 1] - obs[:, 1]))
                ),
                "improvement_pct": float(
                    100 * (np.mean(err_orig) - np.mean(err_bayes)) / np.mean(err_orig)
                ),
            }

        if corrected_neural is not None:
            corrected_neural = np.array(corrected_neural)
            err_neural = self._radial_error(corrected_neural, obs)
            results["neural"] = {
                "mean_error_deg": float(np.mean(err_neural)),
                "std_error_deg": float(np.std(err_neural)),
                "median_error_deg": float(np.median(err_neural)),
                "max_error_deg": float(np.max(err_neural)),
                "mean_error_x_deg": float(
                    np.mean(np.abs(corrected_neural[:, 0] - obs[:, 0]))
                ),
                "mean_error_y_deg": float(
                    np.mean(np.abs(corrected_neural[:, 1] - obs[:, 1]))
                ),
                "improvement_pct": float(
                    100 * (np.mean(err_orig) - np.mean(err_neural)) / np.mean(err_orig)
                ),
            }

        return results

    def print_summary(self, metrics):
        """Imprime resumen de métricas."""
        print(f"\n{'='*60}")
        print(f"EVALUACIÓN DE MODELOS")
        print(f"{'='*60}")

        for model_name, m in metrics.items():
            print(f"\n{model_name.upper()}")
            print(f"  Error radial medio:   {m['mean_error_deg']:.4f}°")
            print(f"  Error radial mediana: {m['median_error_deg']:.4f}°")
            print(f"  Error radial std:     {m['std_error_deg']:.4f}°")
            print(f"  Error radial máx:     {m['max_error_deg']:.4f}°")
            print(f"  Error medio |X|:      {m['mean_error_x_deg']:.4f}°")
            print(f"  Error medio |Y|:      {m['mean_error_y_deg']:.4f}°")
            if "improvement_pct" in m:
                print(f"  Mejora vs original:   {m['improvement_pct']:.1f}%")

        print(f"{'='*60}\n")

    def plot_visual_field(
        self,
        pred,
        obs,
        corrected_bayes=None,
        corrected_neural=None,
        output_file="visual_field_comparison.png",
    ):
        """
        Visualiza predicciones y observaciones en el campo visual.
        """
        pred = np.array(pred)
        obs = np.array(obs)

        n_panels = 1
        if corrected_bayes is not None:
            n_panels += 1
        if corrected_neural is not None:
            n_panels += 1

        fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 7))
        if n_panels == 1:
            axes = [axes]

        panel = 0

        def _draw_panel(ax, pred_pts, obs_pts, title):
            for p, o in zip(pred_pts, obs_pts):
                ax.annotate(
                    "",
                    xy=o,
                    xytext=p,
                    arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5),
                )
            ax.scatter(
                pred_pts[:, 0],
                pred_pts[:, 1],
                c="blue",
                s=80,
                zorder=5,
                label="Predicción",
                marker="o",
            )
            ax.scatter(
                obs_pts[:, 0],
                obs_pts[:, 1],
                c="red",
                s=80,
                zorder=5,
                label="Observación",
                marker="x",
            )
            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
            ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
            ax.set_xlabel("X (grados visuales)")
            ax.set_ylabel("Y (grados visuales)")
            ax.set_title(title)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_aspect("equal")

        _draw_panel(axes[panel], pred, obs, "Sin corrección (original)")
        panel += 1

        if corrected_bayes is not None:
            _draw_panel(
                axes[panel], np.array(corrected_bayes), obs, "Corrección Bayesiana"
            )
            panel += 1

        if corrected_neural is not None:
            _draw_panel(
                axes[panel], np.array(corrected_neural), obs, "Corrección Red Neuronal"
            )

        plt.suptitle(
            "Comparación predicho vs observado en campo visual",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        output_path = self.output_dir / output_file
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Evaluator] OK: Gráfica guardada: {output_path}")

    def plot_error_comparison(self, metrics, output_file="error_comparison.png"):
        """
        Gráfica de barras comparando errores entre modelos.
        """
        model_names = list(metrics.keys())
        mean_errors = [metrics[m]["mean_error_deg"] for m in model_names]
        std_errors = [metrics[m]["std_error_deg"] for m in model_names]

        colors = {"original": "#e74c3c", "bayesian": "#3498db", "neural": "#2ecc71"}
        bar_colors = [colors.get(m, "gray") for m in model_names]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(
            model_names,
            mean_errors,
            yerr=std_errors,
            color=bar_colors,
            capsize=6,
            alpha=0.85,
            edgecolor="black",
        )

        for bar, val in zip(bars, mean_errors):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}°",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        ax.set_ylabel("Error radial medio (grados visuales)", fontsize=12)
        ax.set_title(
            "Comparación de error entre modelos", fontsize=13, fontweight="bold"
        )
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(0, max(mean_errors) * 1.3)

        plt.tight_layout()
        output_path = self.output_dir / output_file
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Evaluator] OK: Gráfica guardada: {output_path}")

    def plot_neural_training(
        self, train_losses, val_losses, output_file="neural_training.png"
    ):
        """Curva de entrenamiento de la red neuronal."""
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(train_losses, label="Train loss", color="#3498db")
        ax.plot(val_losses, label="Val loss", color="#e74c3c", linestyle="--")
        ax.set_xlabel("Época")
        ax.set_ylabel("MSE Loss")
        ax.set_title("Curva de entrenamiento — Red Neuronal")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        output_path = self.output_dir / output_file
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Evaluator] OK: Gráfica guardada: {output_path}")
