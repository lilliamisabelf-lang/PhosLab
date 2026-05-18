"""
bayesian_model.py

Inferencia bayesiana para corrección del mapa de fosfenos.

El modelo asume que el error entre predicción y observación sigue
una distribución gaussiana con media mu (sesgo sistemático) y
varianza sigma² (variabilidad del usuario + ruido del modelo).

Con cada electrodo observado, actualiza la estimación de mu y sigma
usando actualización bayesiana conjugada (prior Normal-InverseGamma).

No requiere scipy — solo numpy.
"""

import json
from pathlib import Path
import numpy as np


class BayesianPhospheneCorrector:
    """
    Modelo bayesiano de corrección del mapa de fosfenos.

    Aprende el sesgo sistemático del modelo (mu_x, mu_y) y su
    incertidumbre, y genera predicciones corregidas.
    """

    def __init__(self, prior_mean=0.0, prior_std=5.0, noise_std=0.5):
        """
        Args:
            prior_mean: Sesgo previo asumido en grados (default: 0 = sin sesgo)
            prior_std:  Incertidumbre del prior en grados (default: 5°)
            noise_std:  Ruido de observación asumido en grados (default: 0.5°)
        """
        # Prior gaussiano para el sesgo (X e Y independientes)
        self.prior_mean = float(prior_mean)
        self.prior_var = float(prior_std) ** 2
        self.noise_var = float(noise_std) ** 2

        # Posterior (se actualiza con cada observación)
        self.posterior_mean_x = self.prior_mean
        self.posterior_var_x = self.prior_var
        self.posterior_mean_y = self.prior_mean
        self.posterior_var_y = self.prior_var

        # Historial de actualizaciones
        self.n_observations = 0
        self.observed_errors_x = []
        self.observed_errors_y = []

    def update(self, error_x_deg, error_y_deg):
        """
        Actualiza el posterior con una nueva observación de error.

        Args:
            error_x_deg: error observado en X (obs_x - pred_x)
            error_y_deg: error observado en Y (obs_y - pred_y)
        """
        self.observed_errors_x.append(float(error_x_deg))
        self.observed_errors_y.append(float(error_y_deg))
        self.n_observations += 1

    # Actualización bayesiana conjugada (Normal con varianza conocida)
    # nueva_var  = 1 / (1/prior_var + 1/noise_var)
    # nueva_mean = nueva_var * (prior_mean/prior_var + obs/noise_var)

        # X
        new_var_x = 1.0 / (1.0 / self.posterior_var_x + 1.0 / self.noise_var)
        self.posterior_mean_x = new_var_x * (
            self.posterior_mean_x / self.posterior_var_x
            + float(error_x_deg) / self.noise_var
        )
        self.posterior_var_x = new_var_x

        # Y
        new_var_y = 1.0 / (1.0 / self.posterior_var_y + 1.0 / self.noise_var)
        self.posterior_mean_y = new_var_y * (
            self.posterior_mean_y / self.posterior_var_y
            + float(error_y_deg) / self.noise_var
        )
        self.posterior_var_y = new_var_y

    def fit(self, errors_x, errors_y):
        """
        Entrena el modelo con arrays de errores.

        Args:
            errors_x: array de errores en X (obs_x - pred_x)
            errors_y: array de errores en Y (obs_y - pred_y)
        """
        # Reset posterior al prior
        self.posterior_mean_x = self.prior_mean
        self.posterior_var_x = self.prior_var
        self.posterior_mean_y = self.prior_mean
        self.posterior_var_y = self.prior_var
        self.n_observations = 0
        self.observed_errors_x = []
        self.observed_errors_y = []

        for ex, ey in zip(errors_x, errors_y):
            self.update(ex, ey)

        print(f"[BayesianModel] Entrenado con {self.n_observations} observaciones")
        print(
            f"               Sesgo estimado X: {self.posterior_mean_x:.4f}° ± {np.sqrt(self.posterior_var_x):.4f}°"
        )
        print(
            f"               Sesgo estimado Y: {self.posterior_mean_y:.4f}° ± {np.sqrt(self.posterior_var_y):.4f}°"
        )

    def correct(self, pred_x_deg, pred_y_deg):
        """
        Corrige una predicción aplicando el sesgo estimado.

        Args:
            pred_x_deg: predicción X en grados
            pred_y_deg: predicción Y en grados

        Returns:
            (corrected_x, corrected_y, uncertainty_x, uncertainty_y)
        """
        corrected_x = float(pred_x_deg) + self.posterior_mean_x
        corrected_y = float(pred_y_deg) + self.posterior_mean_y
        uncertainty_x = float(np.sqrt(self.posterior_var_x + self.noise_var))
        uncertainty_y = float(np.sqrt(self.posterior_var_y + self.noise_var))
        return corrected_x, corrected_y, uncertainty_x, uncertainty_y

    def correct_array(self, pred):
        """
        Corrige un array de predicciones.

        Args:
            pred: (N, 2) array de predicciones

        Returns:
            corrected: (N, 2) array corregido
            uncertainty: (N, 2) array de incertidumbre
        """
        pred = np.array(pred)
        bias = np.array([self.posterior_mean_x, self.posterior_mean_y])
        uncertainty = np.array(
            [
                np.sqrt(self.posterior_var_x + self.noise_var),
                np.sqrt(self.posterior_var_y + self.noise_var),
            ]
        )
        corrected = pred + bias
        return corrected, np.tile(uncertainty, (len(pred), 1))

    def get_params(self):
        """Devuelve los parámetros actuales del modelo."""
        return {
            "n_observations": self.n_observations,
            "posterior_mean_x": self.posterior_mean_x,
            "posterior_mean_y": self.posterior_mean_y,
            "posterior_std_x": float(np.sqrt(self.posterior_var_x)),
            "posterior_std_y": float(np.sqrt(self.posterior_var_y)),
            "noise_std": float(np.sqrt(self.noise_var)),
            "prior_mean": self.prior_mean,
            "prior_std": float(np.sqrt(self.prior_var)),
        }

    def save(self, path):
        """Guarda el modelo en JSON."""
        path = Path(path)
        params = self.get_params()
        params["observed_errors_x"] = self.observed_errors_x
        params["observed_errors_y"] = self.observed_errors_y
        with open(path, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)
        print(f"[BayesianModel] Modelo guardado en: {path}")

    def load(self, path):
        """Carga el modelo desde JSON."""
        with open(path, "r", encoding="utf-8") as f:
            params = json.load(f)
        self.posterior_mean_x = params["posterior_mean_x"]
        self.posterior_mean_y = params["posterior_mean_y"]
        self.posterior_var_x = params["posterior_std_x"] ** 2
        self.posterior_var_y = params["posterior_std_y"] ** 2
        self.n_observations = params["n_observations"]
        self.observed_errors_x = params.get("observed_errors_x", [])
        self.observed_errors_y = params.get("observed_errors_y", [])
        print(f"[BayesianModel] Modelo cargado desde: {path}")
