# Mapping Method — Setup & Usage Guide

Parámetro de configuración que controla el paradigma psicofísico con el que el participante indica la posición percibida de cada fosfeno. Independiente de `response_mode` (cómo se recoge la respuesta) y de `input_mode` (dispositivo de entrada): `mapping_method` solo afecta a lo que el participante ve y hace en la pantalla de respuesta.

Los tres métodos producen exactamente el mismo formato de salida — un canvas PNG del que `mapping_analyzer.py` extrae el centroide — por lo que el pipeline de aprendizaje no necesita cambios.

## Architecture

```
params.yaml
  mapping_method ──▶ main.py
                       │
                       ├─ "absolute"           ──▶ DrawingTablet (sin cross)
                       │                             participante dibuja libremente
                       │
                       ├─ "relative"           ──▶ DrawingTablet + _draw_center_cross()
                       │                             cruz blanca superpuesta en tiempo real
                       │
                       └─ "forced_adjustment"  ──▶ ForcedAdjustmentTablet
                                                     punto aleatorio → participante arrastra

                       └──────────────────────────▶ DrawingResponseCapture (sin cambios)
                                                     │
                                                     ▼
                                              canvas PNG ──▶ mapping_analyzer.py ──▶ centroid_deg
```

`ForcedAdjustmentTablet` y `DrawingTablet` comparten la misma interfaz (`reset` / `update(screen, events) → (finished, canvas)` / `close` / `last_status` / `mode`), por lo que `DrawingResponseCapture` los envuelve sin modificaciones.

## Quick start

En [`config/params.yaml`](config/params.yaml):

```yaml
mapping_method: absolute   # absolute | relative | forced_adjustment
```

Luego ejecuta normalmente:

```
cd percept_mapper
uv run python main.py
```

El resto del experimento (fijación, estimulación, pipeline de aprendizaje) no cambia.

## Los tres métodos

### `absolute` (por defecto)

El participante ve una pantalla negra y dibuja libremente donde percibió el fosfeno, sin ninguna referencia visual. Es el comportamiento original de PhosLab.

**Cuándo usarlo:** experimentos donde se quiere medir la representación interna del participante sin sesgo de anclaje. Mayor varianza entre ensayos, pero es la medida más pura de la posición percibida.

```yaml
mapping_method: absolute
```

---

### `relative`

Igual que `absolute` pero con una cruz blanca fija en el centro de la pantalla durante toda la ventana de respuesta. La cruz actúa como referencia espacial: el participante dibuja *relativo al centro*, lo que reduce el error sistemático de localización cuando los fosfenos caen cerca de la fóvea o en excentricidades pequeñas.

**Cuándo usarlo:** cuando los fosfenos son centrales y los participantes tienen dificultad para estimar si quedaron ligeramente a izquierda o derecha. No introduce sesgo hacia la posición del fosfeno porque la referencia siempre está en el centro, nunca en la predicción.

```yaml
mapping_method: relative
```

La cruz se dibuja con `_draw_center_cross()` en [`phosphene_mapping.py`](scripts/phosphene_mapping.py) sobre la pantalla en cada frame, sin modificar el canvas que se guarda.

---

### `forced_adjustment`

Un punto amarillo aparece en una posición aleatoria de la pantalla (evitando el cuarto central para no coincidir con la predicción). El participante **arrastra** el punto hasta donde percibió el fosfeno y suelta. Si no vio nada (catch trial o umbral no alcanzado), pulsa **ENTER sin arrastrar** para registrar un ensayo vacío.

**Cuándo usarlo:** cuando se quiere reducir la varianza de respuesta mediante ajuste continuo en lugar de localización libre. La posición inicial es aleatoria en cada ensayo para contrarrestar el sesgo de anclaje clásico del método de ajuste.

```yaml
mapping_method: forced_adjustment
```

#### Interacción

| Acción | Resultado |
|--------|-----------|
| Clic sobre el punto + arrastre + soltar | Registra la posición final como respuesta (`status: ok`) |
| ENTER sin haber arrastrado | Ensayo vacío (`status: empty`) — sin dato de posición |
| Movimiento durante el arrastre | Se dibuja un rastro visual (color del pincel) que desaparece en el siguiente ensayo. No se incluye en el canvas guardado |

El canvas de análisis contiene únicamente un punto del tamaño del pincel en la posición final, para que el extractor de centroide funcione igual que en modo dibujo.

**Zona de aparición del punto:** cualquier píxel fuera del rectángulo central `[W/4, H/4, 3W/4, 3H/4]`, con margen mínimo de 40 px a los bordes.

## Metadata

Cada repetición en `metadata.json` incluye:

```json
{
  "mapping_method": "forced_adjustment"
}
```

El campo es el mismo para los tres métodos. `mapping_analyzer.py` y el pipeline de aprendizaje lo ignoran y trabajan directamente con el canvas PNG.

## Compatibilidad

Los tres métodos requieren `response_mode: drawing` — son paradigmas de tablet y no aplican en modo saccade. Con `response_mode: saccade`, el sistema usa `SaccadeScreen` en lugar de cualquier tablet y `mapping_method` no tiene efecto.

| `mapping_method`    | `response_mode: drawing` | `response_mode: saccade` |
|---------------------|--------------------------|--------------------------|
| `absolute`          | ✓                        | ✗ No aplica              |
| `relative`          | ✓                        | ✗ No aplica              |
| `forced_adjustment` | ✓                        | ✗ No aplica              |

## Diagnostic / testing

```
uv run --project percept_mapper python percept_mapper/scripts/mapping_method_smoke_test.py
```

Ejecuta 9 tests sin cabecera de display (SDL dummy driver):

| Test | Qué verifica |
|------|--------------|
| `test_absolute_no_cross` | El modo absolute no dibuja la cruz central |
| `test_relative_cross_visible` | El modo relative dibuja píxeles blancos en el centro |
| `test_forced_pos_not_center` | La posición inicial del punto no cae en el cuarto central |
| `test_forced_drag_completes` | Un arrastre completo devuelve `(True, canvas)` con `status: ok` |
| `test_forced_reset_varies_position` | `reset()` cambia la posición del punto |
| `test_forced_trail_rendered_on_screen` | El rastro se dibuja en pantalla durante el arrastre |
| `test_forced_enter_without_drag_is_empty` | ENTER sin arrastre devuelve `status: empty` |
| `test_forced_no_drag_outside_hit` | Clic fuera del radio de captura no inicia arrastre |
| `test_metadata_includes_mapping_method` | `metadata.json` contiene el campo `mapping_method` |

## Known limitations / TODO

- **`forced_adjustment` con `response_mode: saccade`** — la combinación no está bloqueada por código pero no tiene efecto. Se podría añadir una advertencia en `main.py` si se detecta la combinación.
- **Rastro visual en `forced_adjustment`** — se dibuja con `pygame.mouse.get_pos()` en frames intermedios; en SDL dummy driver devuelve (0,0). En hardware real funciona correctamente.
- **Cruz de referencia en `relative`** — solo visible durante la ventana de respuesta, no durante la estimulación ni la fijación. Una extensión posible sería mantenerla visible durante todo el ensayo.
- **No hay feedback de posición en `absolute` y `relative`** — el participante no ve confirmación de dónde ha dibujado antes de confirmar con ENTER, a diferencia de `forced_adjustment` donde el punto es visible en todo momento.

## File reference

| Ruta | Rol |
|------|-----|
| [`scripts/tablet.py`](scripts/tablet.py) | `DrawingTablet` y `ForcedAdjustmentTablet` — misma interfaz pública. |
| [`scripts/phosphene_mapping.py`](scripts/phosphene_mapping.py) | Recibe `mapping_method` como parámetro; llama `_draw_center_cross()` en modo `relative`. |
| [`main.py`](main.py) | Lee `mapping_method` de config y selecciona la tablet correspondiente. |
| [`scripts/mapping_method_smoke_test.py`](scripts/mapping_method_smoke_test.py) | 9 tests headless del comportamiento de los tres métodos. |
| [`config/params.yaml`](config/params.yaml) | Campo `mapping_method` a nivel raíz del yaml. |
| [`scripts/mapping_analyzer.py`](scripts/mapping_analyzer.py) | Extrae centroide del canvas PNG — sin cambios, compatible con los tres métodos. |
