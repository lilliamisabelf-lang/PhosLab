# PhosLab — Protocolo de Experimentos (TFG)

Guía de referencia para los 5 experimentos de validación de PhosLab. Cada experimento aísla una variable del sistema y usa el mismo formato de salida, por lo que los datos son reutilizables entre experimentos siempre que los parámetros de control coincidan.

## Scripts de análisis

Todos los scripts de análisis viven en `scripts/analysis/`. Se ejecutan desde el directorio `percept_mapper/`.

| Script | Experimento | Descripción |
|--------|-------------|-------------|
| [`scripts/analysis/plot_error_vs_ecc.py`](scripts/analysis/plot_error_vs_ecc.py) | Exp 1 | Scatter + boxplot de error vs excentricidad en una sola sesión, con r/IC95%/p bootstrap por electrodo |
| [`scripts/analysis/compare_mapmethod.py`](scripts/analysis/compare_mapmethod.py) | Exp 2, 4 | Compara N sesiones con etiquetas libres (dispositivos de entrada, métodos de mapeo) |
| [`scripts/analysis/compare_implants.py`](scripts/analysis/compare_implants.py) | Exp 3 | Compara N implantes: cobertura, error por anillo de excentricidad, r/IC/p y test Mann-Whitney |
| [`scripts/analysis/plot_learning_curve.py`](scripts/analysis/plot_learning_curve.py) | Exp 5 | Curva de convergencia del sesgo estimado por el modelo Bayesiano |
| [`scripts/analysis/stats_utils.py`](scripts/analysis/stats_utils.py) | — | Módulo compartido (no se ejecuta): bootstrap por electrodo, r, IC95%, p-valor, Mann-Whitney |
| [`scripts/analysis/map_plot_utils.py`](scripts/analysis/map_plot_utils.py) | — | Módulo compartido (no se ejecuta): figura de 3 paneles estímulo/percepción/superposición |

> `compare_experiments.py` (Exp2 con categorías fijas mouse/gaze/pupil/wacom) se eliminó por no usarse: la comparación real de Exp2 (WACOM vs Pupil Core vs Pupil Neon) usa `compare_mapmethod.py`, que admite etiquetas libres.

---

## Reutilización de datos

Un experimento con `input_mode: mouse`, `mapping_method: absolute` y el mismo CSV sirve **simultáneamente** para varios experimentos sin volver a ejecutarlo:

| Sesión | Exp 1 | Exp 2 (cond. mouse) | Exp 4 (cond. absolute) |
|--------|-------|---------------------|------------------------|
| **Baseline** (mouse + absolute + `synthetic_4ecc_4el.csv`) | ✓ | ✓ | ✓ |

**Regla:** el baseline que ejecutes primero te da datos para Exp 1, la condición de referencia de Exp 2 y la condición `absolute` de Exp 4. Solo necesitas sesiones adicionales para las otras condiciones. Exp 5 siempre requiere sesión propia (`simulated_display_error: enabled: true`).

---

## Parámetros fijos en todos los experimentos

Salvo que se indique lo contrario:

```yaml
experiment_mode: mapping
response_mode: drawing           # excepto Exp 2 condición saccade
mapping_method: absolute         # excepto Exp 4
input_mode: mouse                # excepto Exp 2 condición pupil
stimulation:
  default_current_uA: 90
phosphene_mapping:
  num_repetitions: 10
  catch_trial_rate: 0.15
  randomize: true
  no_immediate_repeat: true
  num_practice_trials: 2
  isi_jitter_ms: 200
retinotopic_mapping:
  simulated_display_error:
    enabled: false               # solo true en Exp 5
```

---

## Experimento 1 — Error vs Excentricidad

**Objetivo:** verificar que el error de localización del fosfeno aumenta con la excentricidad, validando que PhosLab reproduce la pérdida de acuidad periférica conocida en la literatura.

**Variable independiente:** excentricidad del electrodo (grados).
**Variable dependiente:** error euclidiano entre posición predicha (atlas Benson) y posición observada.

### Configuración de params.yaml

```yaml
input_mode: mouse
mapping_method: absolute
response_mode: drawing
retinotopic_mapping:
  coordinate_source: phoslab_csv
  coords_csv_path: config/synthetic_4ecc_4el.csv
  electrode_selection:
    mode: all
phosphene_mapping:
  num_repetitions: 10
```

### Cómo ejecutar el experimento

```powershell
cd percept_mapper
uv run python main.py
```

### Cómo analizar los resultados

```powershell
cd percept_mapper
uv run python scripts/analysis/plot_error_vs_ecc.py `
    --session mapping_experiments/<nombre_sesion>
```

Produce en `comparison_results/exp1_error_vs_ecc/`:
- `error_vs_ecc_boxplot.png` — boxplots de error por anillo de excentricidad
- `error_vs_ecc_scatter.png` — scatter con línea de regresión

### Resultado esperado

Error creciente con la excentricidad. La caja (IQR) del boxplot más periférico debería ser más alta y más ancha que la del central.

---

## Experimento 2 — Paradigma de respuesta: motor-manual vs oculomotor

**Objetivo:** comparar si la precisión y varianza de la respuesta difieren entre responder dibujando con el ratón (motor-manual) y responder mirando con los ojos (oculomotor).

**Variable independiente:** paradigma de respuesta.
**Variable dependiente:** error medio y varianza intra-electrodo.

En los **boxplots** el resultado esperado es:
- `mouse+drawing` → caja más **estrecha** (el ratón es consistente) pero mediana **desplazada** del cero (la mano traduce la posición visual con un pequeño sesgo sistemático).
- `pupil+saccade` → caja más **ancha** (el eye-tracker tiene ruido ensayo a ensayo) pero mediana **más centrada** en cero (los ojos van directamente a donde viste el fosfeno, sin transformación mano-ojo).

### Condición A — Motor-manual (reutiliza el baseline de Exp 1)

```yaml
input_mode: mouse
response_mode: drawing
mapping_method: absolute
```

### Condición B — Oculomotor

```yaml
input_mode: pupil
response_mode: saccade
saccade:
  capture_duration_ms: 2000
  extraction: idt_first_fixation
pupil:
  one_euro:
    min_cutoff: 0.1
    beta: 0.007
```

> Usa exactamente el mismo CSV, los mismos electrodos y el mismo N que la condición A.

### Cómo ejecutar el experimento

Ejecuta dos sesiones separadas cambiando los parámetros anteriores en `params.yaml` cada vez:

```powershell
cd percept_mapper
uv run python main.py   # sesión mouse+drawing
# cambiar params.yaml
uv run python main.py   # sesión pupil+saccade
```

### Cómo analizar los resultados

```powershell
cd percept_mapper
uv run python scripts/analysis/compare_mapmethod.py `
    --sessions mapping_experiments/<sesion_mouse> `
               mapping_experiments/<sesion_pupil> `
    --labels   "Mouse" "Pupil" `
    --out-dir  comparison_results/exp2_input_mode
```

Produce en `comparison_results/exp2_input_mode/`:
- `error_comparison.png` — boxplots agrupados por excentricidad, una caja por condición
- `map_comparison.png` — overlay de posiciones verdaderas vs medidas para cada condición
- `map_<condición>_split.png` — un archivo por condición (estímulo / percepción media ± std / superposición)
- r de Pearson + IC95% + p-valor (bootstrap por electrodo) por condición, impreso en consola

---

## Experimento 3 — Configuración de implante

**Objetivo:** comparar tres arquitecturas de implantación reales (4× Utah Array, Comb 10×10 y Thread-1024/Neuralink) en términos de cobertura del campo visual y error de predicción del atlas de Benson.

**Variable independiente:** tipo y configuración de implante.
**Variable dependiente:** distribución espacial del error + cobertura del campo visual.

### Los tres tipos disponibles

**4× Utah Array** — cuadrícula plana 10×10, pitch 0.4mm, profundidad 1.5mm. Cuatro arrays colocados en posiciones contiguas en V1 cubren un área amplia del campo visual en cuadrícula uniforme. PhosLab soporta múltiples implantes en `electrodes_by_implant`.

**Comb 10×10 5mm** — 10 shanks en línea recta, separados 0.5mm, 10 contactos por shank, profundidad 5mm. Cubre una banda lineal del campo visual con electrodos a distintas profundidades. El **Comb 32×32 30mm tiene shanks de 30mm** y no es válido para corteza visual (espesor ~3mm) — no usar.

**Thread-1024 (Neuralink)** — 32 hilos flexibles que salen en todas las direcciones desde un hub central de 3.5mm de radio, con 16 contactos por hilo a lo largo de 4mm. La cobertura es circular y la densidad disminuye hacia los hilos más periféricos. A diferencia del Comb (shanks paralelos en línea), el Thread cubre un disco completo.

```
Comb 10×10:          Thread-1024 (Neuralink):
| | | | | | | | | |        * * * * *
| | | | | | | | | |      *     hub    *
| | | | | | | | | |        * * * * *
(banda lineal)             (disco radial)
```

### Configuración (una sesión por configuración)

Genera el CSV de cada configuración desde `implant_explorer` y cambia en `params.yaml`:

```yaml
input_mode: mouse
mapping_method: absolute
response_mode: drawing
phosphene_mapping:
  num_repetitions: 10
retinotopic_mapping:
  coords_csv_path: config/<csv_de_la_configuracion>.csv
```

Para 4× Utah, añade los 4 implants en `electrodes_by_implant`:

```yaml
phosphene_mapping:
  electrodes_by_implant:
    - implant_id: <utah_1>
      electrode_index: [0, 10, 20, 30]
    - implant_id: <utah_2>
      electrode_index: [0, 10, 20, 30]
    - implant_id: <utah_3>
      electrode_index: [0, 10, 20, 30]
    - implant_id: <utah_4>
      electrode_index: [0, 10, 20, 30]
```

### Cómo ejecutar el experimento

```powershell
cd percept_mapper
uv run python main.py   # sesión Utah×4
# cambiar params.yaml
uv run python main.py   # sesión Comb
# cambiar params.yaml
uv run python main.py   # sesión Thread
```

### Cómo analizar los resultados

```powershell
cd percept_mapper
uv run python scripts/analysis/compare_implants.py `
    --sessions mapping_experiments/<sesion_utah> `
               mapping_experiments/<sesion_comb> `
               mapping_experiments/<sesion_thread> `
    --labels   "Utah Array" "Comb 10x10" "Thread-1024" `
    --out-dir  comparison_results/exp3_implants `
    --compare-ring 4
```

Produce en `comparison_results/exp3_implants/`:
- `coverage_map.png` — mapa polar de cobertura del campo visual por implante
- `error_comparison.png` — boxplots side-by-side de error por configuración
- `error_vs_ecc_exp3.png` — error por anillo de excentricidad, con recta de regresión (r) por implante
- `map_<implante>_split.png` — un archivo por implante (estímulo / percepción media ± std / superposición)
- r/IC95%/p por implante y, con `--compare-ring`, test de Mann-Whitney entre pares de implantes en esa excentricidad — todo impreso en consola

---

## Experimento 4 — Método de mapeo

**Objetivo:** comparar si el paradigma de respuesta (`absolute`, `relative`, `forced_adjustment`) afecta al error y la varianza de localización.

**Variable independiente:** `mapping_method`.
**Variable dependiente:** error medio y varianza intra-electrodo.

### Las tres condiciones

```yaml
# Condición A — baseline reutilizable
mapping_method: absolute

# Condición B
mapping_method: relative       # cruz blanca fija en el centro durante la respuesta

# Condición C
mapping_method: forced_adjustment  # punto aleatorio que el participante arrastra
```

Usa el mismo CSV, electrodos, `input_mode: mouse` y N en las tres sesiones.

### Cómo ejecutar el experimento

```powershell
cd percept_mapper
uv run python main.py   # sesión absolute
# cambiar mapping_method en params.yaml
uv run python main.py   # sesión relative
# cambiar mapping_method en params.yaml
uv run python main.py   # sesión forced_adjustment
```

### Cómo analizar los resultados

```powershell
cd percept_mapper
uv run python scripts/analysis/compare_mapmethod.py `
    --sessions mapping_experiments/<sesion_absolute> `
               mapping_experiments/<sesion_relative> `
               mapping_experiments/<sesion_forced> `
    --labels   "Absoluto" "Relativo" "Ajuste forzado" `
    --out-dir  comparison_results/exp4_mapping_method
```

Produce en `comparison_results/exp4_mapping_method/`:
- `error_comparison.png` — boxplots side-by-side de error por método
- `map_comparison.png` — overlay de posiciones medidas por método
- `map_absolute_split.png`, `map_relative_split.png`, `map_forced_adjustment_split.png` — un archivo por método (estímulo / percepción media ± std / superposición)
- r/IC95%/p por método (bootstrap por electrodo), impreso en consola

### Resultado esperado en los boxplots

- `forced_adjustment` → caja más estrecha (ajuste continuo reduce varianza).
- `relative` → mediana más centrada en electrodos centrales (la cruz ayuda a calibrar).
- `absolute` → caja más ancha, mediana más variable (sin anclaje externo).

---

## Experimento 5 — Validación del pipeline de aprendizaje

**Objetivo:** verificar que el modelo Bayesiano y la red neuronal detectan y recuperan un sesgo conocido inyectado artificialmente.

**Variable independiente:** número de ensayos acumulados.
**Variable dependiente:** sesgo estimado por el modelo vs sesgo real inyectado (`bias_deg: [2.0, 1.0]`).

### Configuración de params.yaml

```yaml
input_mode: mouse
mapping_method: absolute
response_mode: drawing
phosphene_mapping:
  num_repetitions: 20
retinotopic_mapping:
  simulated_display_error:
    enabled: true
    bias_deg: [2.0, 1.0]      # 2° derecha, 1° arriba — verdad conocida
    noise_std_deg: 0.0         # sin ruido para validación limpia
    noise_seed: 42
```

### Cómo ejecutar el experimento

```powershell
cd percept_mapper
uv run python main.py
```

### Cómo ejecutar el pipeline de aprendizaje

```powershell
cd percept_mapper
uv run python scripts/learning/bayesian_model.py `
    --mapping_dir mapping_experiments/<nombre_sesion>

uv run python scripts/learning/neural_model.py `
    --mapping_dir mapping_experiments/<nombre_sesion>
```

Para validar con cross-validation (k-fold, requerido antes de cualquier afirmación publicable sobre la mejora del modelo):

```powershell
cd percept_mapper
uv run python scripts/learning/cross_validation.py `
    --mapping_dir mapping_experiments/<nombre_sesion>
```

Reporta error medio ± std en cada fold. El modelo solo es válido si el error en test es consistentemente inferior al error sin corrección.

### Cómo analizar los resultados

```powershell
cd percept_mapper
uv run python scripts/analysis/plot_learning_curve.py `
    --session   mapping_experiments/<nombre_sesion> `
    --bias-true 2.0 1.0 `
    --out-dir   comparison_results/exp5_learning
```

Produce en `comparison_results/exp5_learning/`:
- `learning_curve.png` — sesgo estimado por iteración vs línea de referencia `[2.0, 1.0]`

### Resultado esperado

El sesgo estimado converge hacia `[2.0, 1.0]` con más ensayos. Si no converge, la causa más probable es N insuficiente (`num_repetitions < 15`) o sesgo demasiado pequeño respecto al ruido de respuesta del participante.

---

## Orden de ejecución recomendado

| Orden | Sesión | Experimentos que alimenta | Sesiones nuevas |
|-------|--------|---------------------------|-----------------|
| 1° | Baseline (mouse + absolute + `synthetic_4ecc_4el.csv`) | Exp 1, Exp 2 cond. A, Exp 4 cond. A | 1 |
| 2° | Exp 2 — pupil + saccade | Exp 2 cond. B | 1 |
| 3° | Exp 4 — relative | Exp 4 cond. B | 1 |
| 4° | Exp 4 — forced_adjustment | Exp 4 cond. C | 1 |
| 5° | Exp 3 — generar CSVs en implant_explorer + 3 sesiones | Exp 3 | 3 |
| 6° | Exp 5 — error simulado | Exp 5 | 1 |

**Total: 8 sesiones de captura.**

---

## File reference

| Ruta | Rol |
|------|-----|
| [`config/params.yaml`](config/params.yaml) | Configuración de cada sesión |
| [`config/synthetic_4ecc_4el.csv`](config/synthetic_4ecc_4el.csv) | CSV baseline (4 excentricidades) |
| [`scripts/analysis/plot_error_vs_ecc.py`](scripts/analysis/plot_error_vs_ecc.py) | Error vs excentricidad (Exp 1) |
| [`scripts/analysis/compare_mapmethod.py`](scripts/analysis/compare_mapmethod.py) | Comparación N sesiones genérica (Exp 2, 4) |
| [`scripts/analysis/compare_implants.py`](scripts/analysis/compare_implants.py) | Comparación de implantes (Exp 3) |
| [`scripts/analysis/plot_learning_curve.py`](scripts/analysis/plot_learning_curve.py) | Curva de aprendizaje (Exp 5) |
| [`scripts/analysis/stats_utils.py`](scripts/analysis/stats_utils.py) | Módulo compartido: r, IC95%, p-valor, Mann-Whitney (bootstrap por electrodo) |
| [`scripts/analysis/map_plot_utils.py`](scripts/analysis/map_plot_utils.py) | Módulo compartido: figura de 3 paneles estímulo/percepción/superposición |
| [`scripts/mapping_analyzer.py`](scripts/mapping_analyzer.py) | Análisis de una sesión individual |
| [`scripts/learning/bayesian_model.py`](scripts/learning/bayesian_model.py) | Modelo Bayesiano (Exp 5) |
| [`scripts/learning/neural_model.py`](scripts/learning/neural_model.py) | Red neuronal (Exp 5) |
| [`implant_explorer/implant_designs/`](../implant_explorer/implant_designs/) | Diseños disponibles (Utah, Comb, Thread) |
