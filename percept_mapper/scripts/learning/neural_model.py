"""
neural_model.py

Red neuronal para corrección del mapa de fosfenos.

Aprende una función f(pred_x, pred_y) → (obs_x, obs_y) usando
una red MLP de 2 capas ocultas. Usa PyTorch, que ya está instalado.

Recomendado con 20+ electrodos. Con menos datos puede sobreajustar.
"""

import json
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


class PhospheneNet(nn.Module):
    """Red neuronal MLP para corrección de fosfenos."""

    def __init__(self, hidden_size=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, x):
        return self.net(x)


class NeuralPhospheneCorrector:
    """
    Corrector de fosfenos basado en red neuronal.

    Aprende la función de corrección pred → obs directamente.
    """

    def __init__(self, hidden_size=32, learning_rate=0.01, num_epochs=500):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        self.model = PhospheneNet(hidden_size=hidden_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        # Normalización
        self.pred_mean = None
        self.pred_std = None
        self.obs_mean = None
        self.obs_std = None

        self.train_losses = []
        self.val_losses = []
        self.is_trained = False

    def fit(self, pred, obs, train_split=0.8, verbose=True):
        """
        Entrena la red neuronal.

        Args:
            pred: (N, 2) array de predicciones
            obs:  (N, 2) array de observaciones
            train_split: proporción de datos para entrenamiento
            verbose: mostrar progreso
        """
        pred = np.array(pred, dtype=np.float32)
        obs = np.array(obs, dtype=np.float32)

        if len(pred) < 3:
            print("[NeuralModel] WARN: Pocos datos para entrenar una red neuronal.")
            print("              Se necesitan al menos 3 observaciones.")
            print("              Usa el modelo Bayesiano con pocos datos.")
            return False

        # Normalizar
        self.pred_mean = pred.mean(axis=0)
        self.pred_std = pred.std(axis=0) + 1e-8
        self.obs_mean = obs.mean(axis=0)
        self.obs_std = obs.std(axis=0) + 1e-8

        pred_norm = (pred - self.pred_mean) / self.pred_std
        obs_norm = (obs - self.obs_mean) / self.obs_std

        # Split train/val
        n = len(pred_norm)
        n_train = max(1, int(n * train_split))
        idx = np.random.permutation(n)
        train_idx = idx[:n_train]
        val_idx = idx[n_train:] if n_train < n else idx[:1]

        X_train = torch.tensor(pred_norm[train_idx])
        y_train = torch.tensor(obs_norm[train_idx])
        X_val = torch.tensor(pred_norm[val_idx])
        y_val = torch.tensor(obs_norm[val_idx])

        print(f"[NeuralModel] Entrenando: {n_train} train, {len(val_idx)} val")
        print(f"              Épocas: {self.num_epochs}, lr: {self.learning_rate}")

        self.train_losses = []
        self.val_losses = []

        for epoch in range(self.num_epochs):
            # Train
            self.model.train()
            self.optimizer.zero_grad()
            pred_out = self.model(X_train)
            loss = self.loss_fn(pred_out, y_train)
            loss.backward()
            self.optimizer.step()

            # Val
            self.model.eval()
            with torch.no_grad():
                val_out = self.model(X_val)
                val_loss = self.loss_fn(val_out, y_val)

            self.train_losses.append(float(loss))
            self.val_losses.append(float(val_loss))

            if verbose and (epoch + 1) % 100 == 0:
                print(
                    f"  Época {epoch+1}/{self.num_epochs} — "
                    f"train loss: {loss:.6f}, val loss: {val_loss:.6f}"
                )

        self.is_trained = True
        print("[NeuralModel] Entrenamiento completado")
        print(f"              Loss final train: {self.train_losses[-1]:.6f}")
        print(f"              Loss final val:   {self.val_losses[-1]:.6f}")
        return True

    def correct(self, pred_x_deg, pred_y_deg):
        """
        Corrige una predicción.

        Returns:
            (corrected_x, corrected_y)
        """
        if not self.is_trained:
            raise RuntimeError("El modelo no ha sido entrenado. Llama a fit() primero.")

        pred = np.array([[pred_x_deg, pred_y_deg]], dtype=np.float32)
        pred_norm = (pred - self.pred_mean) / self.pred_std
        x = torch.tensor(pred_norm)

        self.model.eval()
        with torch.no_grad():
            out = self.model(x).numpy()

        corrected = out * self.obs_std + self.obs_mean
        return float(corrected[0, 0]), float(corrected[0, 1])

    def correct_array(self, pred):
        """Corrige un array de predicciones."""
        if not self.is_trained:
            raise RuntimeError("El modelo no ha sido entrenado.")

        pred = np.array(pred, dtype=np.float32)
        pred_norm = (pred - self.pred_mean) / self.pred_std
        x = torch.tensor(pred_norm)

        self.model.eval()
        with torch.no_grad():
            out = self.model(x).numpy()

        corrected = out * self.obs_std + self.obs_mean
        return corrected

    def save(self, path):
        """Guarda el modelo."""
        path = Path(path)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "hidden_size": self.hidden_size,
                "pred_mean": self.pred_mean.tolist(),
                "pred_std": self.pred_std.tolist(),
                "obs_mean": self.obs_mean.tolist(),
                "obs_std": self.obs_std.tolist(),
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
            },
            path,
        )
        print(f"[NeuralModel] Modelo guardado en: {path}")

    def load(self, path):
        """Carga el modelo."""
        checkpoint = torch.load(path, map_location="cpu")
        self.hidden_size = checkpoint["hidden_size"]
        self.model = PhospheneNet(hidden_size=self.hidden_size)
        self.model.load_state_dict(checkpoint["model_state"])
        self.pred_mean = np.array(checkpoint["pred_mean"], dtype=np.float32)
        self.pred_std = np.array(checkpoint["pred_std"], dtype=np.float32)
        self.obs_mean = np.array(checkpoint["obs_mean"], dtype=np.float32)
        self.obs_std = np.array(checkpoint["obs_std"], dtype=np.float32)
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.is_trained = True
        print(f"[NeuralModel] Modelo cargado desde: {path}")
