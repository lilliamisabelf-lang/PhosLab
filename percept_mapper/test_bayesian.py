# Crea esto como test_bayesian.py en la raíz
import numpy as np
from scripts.learning.bayesian_model import BayesianPhospheneCorrector

np.random.seed(42)
n = 20
pred = np.random.uniform(-10, 10, (n, 2))

# Sesgo conocido: +3° en X, -2° en Y, más ruido pequeño
TRUE_BIAS_X = 3.0
TRUE_BIAS_Y = -2.0
obs = pred + np.array([TRUE_BIAS_X, TRUE_BIAS_Y]) + np.random.normal(0, 0.3, (n, 2))

errors_x = obs[:, 0] - pred[:, 0]
errors_y = obs[:, 1] - pred[:, 1]

model = BayesianPhospheneCorrector(prior_mean=0.0, prior_std=5.0, noise_std=0.5)
model.fit(errors_x, errors_y)

params = model.get_params()
print(f"Sesgo real X: {TRUE_BIAS_X}°  →  estimado: {params['posterior_mean_x']:.3f}°")
print(f"Sesgo real Y: {TRUE_BIAS_Y}°  →  estimado: {params['posterior_mean_y']:.3f}°")
