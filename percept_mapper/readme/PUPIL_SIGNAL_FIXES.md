# Mejoras en la señal del Pupil Tracker

Este documento explica los tres problemas encontrados en la integración con Pupil Capture,
los cambios aplicados para resolverlos, cómo ajustar cada parámetro, y cómo revertir
al código anterior si los cambios no funcionan en el lab.

---

## Contexto: cómo llega la señal

Pupil Capture publica datos por ZMQ en dos canales distintos:

```
gaze_on_surfaces      → posición bruta del ojo, frame a frame (60 Hz)
fixations_on_surfaces → posición estabilizada ya calculada por Pupil Capture
```

PhosLab se suscribe a ambos y decide cuál usar. El problema es que
**hasta ahora usaba el canal incorrecto**.

---

## Cambio 1 — Priorizar fixaciones sobre raw gaze

### El problema

El código original miraba primero en `gaze_on_surfaces` (raw) y solo usaba
`fixations_on_surfaces` como fallback. Como casi siempre hay muestras raw
disponibles, las fixaciones limpias nunca se usaban.

### Por qué importa

`gaze_on_surfaces` contiene un punto por frame, incluyendo:
- El movimiento **durante** la sacada (el ojo aún está volando)
- Los frames justo **después** de un parpadeo (ojo inestable)
- Microtremores normales del ojo

`fixations_on_surfaces` ya tiene todo eso resuelto: Pupil Capture espera a que
el ojo lleve al menos 100 ms quieto, calcula el centroide de esa ventana y
publica un único punto limpio. Es exactamente la señal que necesita el
experimento de sacada.

### El cambio

```
ANTES: raw gaze → (si no hay) → fixaciones
DESPUÉS: fixaciones → (si no hay) → raw gaze
```

### Cuándo NO usar fixaciones primero

Si usas `input_mode: gaze` (seguimiento continuo del ojo, no sacada), el raw gaze
es lo correcto porque quieres actualización frame a frame. Para `input_mode: pupil`
con `response_mode: saccade`, las fixaciones son siempre la mejor opción.

---

## Cambio 2 — Rechazo de outliers por velocidad antes del filtro

### El problema

El One Euro filter suaviza trayectorias continuas pero no rechaza saltos
imposibles. Cuando el participante parpadea, el frame inmediatamente posterior
a la reapertura del ojo tiene confidence alta (el ojo ya está visible) pero la
posición es errática. Esa muestra entra al filtro, que la "absorbe" gradualmente
y tarda varios frames en estabilizarse.

### Cómo funciona el rechazo

Antes de pasar cada muestra al One Euro, se calcula la velocidad implícita:

```
velocidad = distancia(posicion_nueva, posicion_anterior) / dt
```

Si esa velocidad supera un umbral físico razonable, la muestra se descarta.
Los ojos humanos en una sacada normal van a 200–500 px/s.
Cualquier salto mayor a ~1500 px/s casi seguro es un artefacto.

### Parámetro: `max_velocity_px_s`

| Valor | Efecto |
|-------|--------|
| `500` | Muy estricto — rechaza sacadas rápidas |
| `1000` | Equilibrado — rechaza artefactos sin cortar sacadas |
| `1500` | Por defecto — conservador, solo rechaza spikes evidentes |
| `3000` | Muy permisivo — solo rechaza teleportaciones |

Ajústalo en `params.yaml`:
```yaml
pupil:
  max_velocity_px_s: 1500   # añadir esta línea
```

Si ves que el sistema "pierde" la primera fijación tras una sacada rápida,
sube el valor. Si sigues viendo spikes, bájalo.

---

## Cambio 3 — One Euro: parámetros para respuesta por sacada

### Qué hace el One Euro filter

El One Euro es un filtro adaptativo: suaviza más cuando el ojo está quieto
y menos cuando se mueve rápido. Tiene tres parámetros:

#### `min_cutoff` — suavizado en reposo

- **Qué controla:** cuánto se suaviza la señal cuando el ojo está quieto.
- **Valores bajos (0.1):** mucho suavizado, mucha inercia → útil para seguimiento
  lento, pero crea LAG y tarda en recuperarse de spikes.
- **Valores altos (1.0–3.0):** menos suavizado, respuesta más rápida → mejor
  para capturar el punto de aterrizaje de una sacada.
- **Recomendado para sacada:** `1.0` (valor por defecto en código, el params.yaml
  lo tenía sobreescrito a `0.1`).

#### `beta` — adaptación a movimientos rápidos

- **Qué controla:** cómo de agresivamente el filtro "deja pasar" la señal
  cuando detecta movimiento rápido.
- **Valores bajos (0.007):** poca adaptación → el filtro no distingue bien entre
  sacada legítima y spike.
- **Valores altos (0.05–0.1):** más adaptación → el filtro se abre más durante
  una sacada real, lo que permite capturarla mejor.
- **Recomendado para sacada:** `0.05`.

#### `d_cutoff` — filtrado de la derivada (velocidad)

- **Qué controla:** cuánto se suaviza el cálculo interno de velocidad del filtro.
- **Cambiar esto raramente es necesario.** Dejarlo en `1.0`.

### Configuración recomendada en `params.yaml`

```yaml
pupil:
  one_euro:
    fps: 60
    min_cutoff: 1.0      # era 0.1 — menos lag, mejor respuesta post-sacada
    beta: 0.05           # era 0.007 — más adaptación a movimientos rápidos
    d_cutoff: 1.0        # sin cambios
  min_confidence: 0.75   # era 0.7 — un poco más estricto con muestras de baja calidad
  max_velocity_px_s: 1500  # nuevo parámetro — rechazo de spikes
```

---

## Resumen de los tres cambios

| # | Fichero | Qué cambia | Impacto esperado |
|---|---------|-----------|-----------------|
| 1 | `core/pupil_tracker.py` | Prioridad fixaciones > raw gaze | Mayor reducción de error, señal más estable |
| 2 | `core/pupil_tracker.py` | Rechazo de outliers por velocidad | Elimina spikes de parpadeo |
| 3 | `params.yaml` | `min_cutoff: 1.0`, `beta: 0.05` | Menos lag, mejor captura del aterrizaje de sacada |

---

## Cómo revertir al código anterior

Si mañana en el lab los cambios no funcionan, ejecuta este comando desde
la raíz del repositorio para volver al estado exacto anterior a estos cambios:

```powershell
cd C:\PhosLab
git diff HEAD -- percept_mapper/core/pupil_tracker.py
```

Primero comprueba qué cambió. Si quieres deshacer solo `pupil_tracker.py`:

```powershell
git checkout b389789 -- percept_mapper/core/pupil_tracker.py
```

Si quieres deshacer también los cambios de `params.yaml`:

```powershell
git checkout b389789 -- percept_mapper/config/params.yaml
```

> `b389789` es el commit anterior a estos cambios. Si hay commits intermedios,
> ajusta el hash al que aparezca en `git log --oneline`.

Para volver a aplicar los cambios si decides que sí funcionan:

```powershell
git checkout HEAD -- percept_mapper/core/pupil_tracker.py
git checkout HEAD -- percept_mapper/config/params.yaml
```

---

## Checklist para el lab

Antes de empezar el experimento con los cambios nuevos:

- [ ] Pupil Capture está corriendo con el Surface Tracker activo
- [ ] La superficie `phoslab_screen` está definida y detectándose
- [ ] Calibración hecha con el protocolo habitual (5 o 9 puntos)
- [ ] Ejecutar `uv run python scripts/pupil_smoke_test.py` y verificar que el
      gaze se mueve suavemente al mover los ojos
- [ ] Hacer un trial de prueba con `num_practice_trials: 3` antes de la sesión real
- [ ] Si la señal sigue siendo abrupta: volver al código anterior con el comando
      de arriba y anotar qué observaste para diagnosticar más

---

## Guía rápida de diagnóstico

| Síntoma | Causa probable | Ajuste |
|---------|---------------|--------|
| Señal salta de golpe y vuelve | Spikes de parpadeo | Bajar `max_velocity_px_s` a 1000 |
| Respuesta siempre cerca del centro | `skip_anchor_radius_px` demasiado grande | Bajar a 30 px en `saccade.idt` |
| No detecta ninguna fixación | `min_duration_ms` demasiado largo | Bajar de 100 a 80 ms |
| Lag visible entre donde miras y donde registra | `min_cutoff` demasiado bajo | Subir a 2.0 |
| El sistema prefiere siempre raw gaze (no fixaciones) | Pupil Capture no publica fixaciones | Verificar que "Online Fixation Detector" está activo en Pupil Capture |
