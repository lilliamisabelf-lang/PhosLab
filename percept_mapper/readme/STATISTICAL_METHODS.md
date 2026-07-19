# Metodología estadística de Exp1-4 — referencia para verificación

Este documento explica, con precisión suficiente para ser auditado por otra
persona o modelo, cómo se calcula cada estadístico usado en los Experimentos
1-4 de PhosLab (r de Pearson, IC 95%, p-valor, test de Mann-Whitney), qué
datos alimentan cada experimento, y qué resultados se obtuvieron. El código
fuente real vive en `percept_mapper/scripts/analysis/stats_utils.py`; este
documento describe ese código, no lo sustituye — para verificar de verdad,
hay que leer también el código y, si es posible, re-ejecutar los scripts.

## 1. El problema que resuelve toda esta metodología: pseudo-replicación

Cada electrodo se estimula 10 veces (10 repeticiones). Esas 10 repeticiones
**no son observaciones independientes**: comparten el sesgo de colocación de
ese electrodo concreto, la posición retinotópica que le asignó el atlas de
Benson, etc. Si se calcula un intervalo de confianza o un p-valor tratando
las 160-200 repeticiones de una sesión como si fueran 160-200 muestras
independientes, el resultado es artificialmente más "seguro" de lo que
realmente es — este error se llama pseudo-replicación.

La solución aplicada en todo `stats_utils.py`: el **tamaño muestral efectivo
de una sesión es su número de electrodos** (16-20 según el experimento), no
su número de ensayos (160-200). Todos los IC y p-valores se calculan
remuestreando **electrodos completos** (con sus 10 repeticiones en bloque),
nunca ensayos individuales sueltos.

## 2. Cómo se calcula cada estadístico (funciones en `stats_utils.py`)

### 2.1 `collect_electrode_data(results)` → (excentricidades, errores)

Lee `consolidated_results.json` y devuelve, por electrodo: su excentricidad
(`hypot(x, y)` de `stimulation_position_deg`) y la lista de sus errores por
repetición (`distance_to_stim_deg` de cada `per_repetition_metrics`).
Electrodos sin datos válidos se descartan.

### 2.2 `pearson_r_by_trial(electrode_ecc, electrode_errs)` → r, n_el, n_obs

r de Pearson estándar (`numpy.corrcoef`) calculado sobre **todos los
ensayos individuales** (no sobre medias por electrodo) — es decir, cada
repetición aporta un punto (excentricidad_del_electrodo, error_del_ensayo)
a la nube de puntos sobre la que se ajusta la correlación. Esto es lo mismo
que se ve en la recta de regresión de las figuras.

### 2.3 `cluster_bootstrap_r(...)` → distribución bootstrap de r (10 000 valores)

Con semilla fija (`seed=42`, reproducible):
1. Remuestrea **electrodos con reemplazo** (no ensayos) — si una sesión
   tiene 20 electrodos, cada remuestreo bootstrap elige 20 electrodos al
   azar con reemplazo (algunos se repiten, otros no aparecen).
2. Para cada electrodo elegido, se incluyen **sus 10 repeticiones en
   bloque** (no se remuestrean por separado).
3. Se recalcula r de Pearson sobre esa muestra reconstruida.
4. Se repite 10 000 veces → distribución bootstrap de r.

### 2.4 `bootstrap_ci(boot_values)` → IC 95%

Percentiles 2,5 y 97,5 de la distribución bootstrap. Método de percentiles
simple (no BCa), consistente con el resto del pipeline.

### 2.5 `bootstrap_p_two_sided(boot_values)` → p-valor

**No se usa `scipy.stats.pearsonr`** para el p-valor (ese cálculo asume
ensayos independientes y da p-valores erróneamente pequeños/optimistas —
exactamente el mismo error de pseudo-replicación del punto 1). En su lugar:
`p = 2 * min(P(bootstrap <= 0), P(bootstrap >= 0))` — el doble de la
proporción de remuestreos bootstrap que caen del lado contrario al signo
del valor observado. Equivale a comprobar si el IC bootstrap cruza el cero,
pero da un valor numérico continuo en vez de solo significativo/no
significativo.

### 2.6 `report_r(results)` → dict con r, r², IC95%, p, n_el, n_obs

Combina 2.1-2.5 en una sola llamada; es lo que usan todos los scripts para
imprimir el resumen de consola.

### 2.7 `mannwhitney_compare(label_a, errs_a, label_b, errs_b)` → comparación entre dos grupos

Para comparar dos sesiones/dispositivos/implantes entre sí (no la
correlación interna de una sesión, sino si A y B difieren):
1. **Mann-Whitney U** (`scipy.stats.mannwhitneyu`) sobre todos los ensayos
   individuales de A vs todos los de B — test no paramétrico, no asume
   normalidad, compara si las distribuciones difieren.
2. **IC 95% bootstrap de la diferencia de medianas** (`diff = mediana_B -
   mediana_A`): remuestrea electrodos completos de A y de B por separado
   (mismo principio que 2.3), recalcula la diferencia de medianas 10 000
   veces.

El motivo de reportar **ambos** (Mann-Whitney + IC bootstrap) en vez de solo
uno: Mann-Whitney da significancia estadística formal (p-valor) pero no
tamaño del efecto; el IC bootstrap da la magnitud de la diferencia con su
incertidumbre, y al estar calculado también por electrodo, es consistente
con el resto de la metodología (no vuelve a caer en pseudo-replicación).

## 3. Qué script usa qué función, y para qué experimento

| Script | Experimento | Qué calcula |
|---|---|---|
| `plot_error_vs_ecc.py` | Exp 1 | `report_r()` sobre la única sesión (WACOM, absoluto) |
| `compare_mapmethod.py` | Exp 2, 4 | `report_r()` por sesión + `mannwhitney_compare()` entre cada par de sesiones (todos los ensayos, sin restringir a un anillo de excentricidad) |
| `compare_implants.py` | Exp 3 | `report_r()` por implante + `mannwhitney_compare()` entre pares de implantes **en un anillo de excentricidad concreto** (`--compare-ring`), para controlar el efecto de confusión excentricidad↔geometría |

Diferencia importante entre Exp3 y Exp4 en cómo se usa `mannwhitney_compare`:
Exp3 compara implantes **dentro del mismo anillo de excentricidad** (porque
los implantes cubren rangos de excentricidad distintos y hay que controlar
esa variable); Exp4 compara métodos de mapeo **sobre todos los ensayos**
(porque los tres métodos comparten el mismo conjunto de 20 electrodos y las
mismas excentricidades, así que no hace falta controlar por anillo).

## 4. Datos de origen por experimento (sesiones reales verificadas)

| Experimento | Condición | Sesión (`mapping_experiments/`) |
|---|---|---|
| Exp 1 | WACOM, absoluto | `mapping_mapeo_multiples_electrodo_20260626_165130` |
| Exp 2 | WACOM | `mapping_mapeo_multiples_electrodo_20260626_165130` (misma que Exp1) |
| Exp 2 | Pupil Core | `mapping_mapeo_multiples_electrodo_20260617_110153` |
| Exp 2 | Pupil Neon | `mapping_mapeo_multiples_electrodo_20260630_185135` |
| Exp 3 | Comb 10x10 | `mapping_mapeo_multiples_electrodo_20260630_200505` |
| Exp 3 | Thread-1024 | `mapping_mapeo_multiples_electrodo_20260630_203952` |
| Exp 3 | Utah Array | `mapping_mapeo_multiples_electrodo_20260630_210035` |
| Exp 4 | Absoluto | `mapping_mapeo_multiples_electrodo_20260626_165130` (misma que Exp1/WACOM) |
| Exp 4 | Relativo | `mapping_mapeo_multiples_electrodo_20260630_194015` |
| Exp 4 | Ajuste forzado | `mapping_mapeo_multiples_electrodo_20260626_175247` |

**Nota de independencia**: la sesión `165130` se reutiliza como WACOM (Exp2)
y como Absoluto (Exp4) — es la misma sesión, no una repetición
independiente. Exp1, la referencia de Exp2 y la condición Absoluto de Exp4
no son tres muestras estadísticamente independientes entre sí.

Estas 8 sesiones (con sus JSON/PNG/txt, sin los CSV de datos crudos por
ensayo) están subidas a GitHub en `percept_mapper/mapping_experiments/`,
junto con las figuras resultantes en `percept_mapper/comparison_results/`,
específicamente para que este análisis sea reproducible ejecutando los
comandos de la sección "Uso (PowerShell)" del docstring de cada script.

## 5. Resultados verificados (re-ejecutados el 2026-07-19, semilla fija = reproducible)

### Exp 1
r=0,254 (r²≈0,065); IC95% [0,155; 0,349]; p<0,001; n_el=20; n_obs=199

### Exp 2 — correlación excentricidad-error por dispositivo
| Dispositivo | r | r² | IC 95% | p | n_el | n_obs |
|---|---|---|---|---|---|---|
| WACOM | 0,254 | 0,065 | [0,155; 0,349] | <0,001 | 20 | 199 |
| Pupil Core | 0,485 | 0,235 | [0,206; 0,719] | 0,001 | 16 | 160 |
| Pupil Neon | 0,262 | 0,068 | [0,004; 0,438] | 0,048 | 20 | 196 |

### Exp 2 — Mann-Whitney entre dispositivos (todos los ensayos)
| Par | diff mediana | U | p | IC95% diff |
|---|---|---|---|---|
| WACOM vs Pupil Core | 0,573° | 26883,0 | <0,0001 | [0,299; 1,000] |
| WACOM vs Pupil Neon | 1,810° | 38314,0 | <0,0001 | [1,586; 2,016] |
| Pupil Core vs Pupil Neon | 1,237° | 26117,0 | <0,0001 | [0,748; 1,589] |

### Exp 3 — correlación excentricidad-error por implante
| Implante | r | r² | IC 95% | p | n_el | n_obs |
|---|---|---|---|---|---|---|
| Comb 10x10 | 0,150 | 0,022 | [0,035; 0,226] | 0,015 | 16 | 160 |
| Thread-1024 | 0,153 | 0,023 | [0,041; 0,280] | 0,007 | 16 | 160 |
| Utah Array | 0,441 | 0,195 | [0,297; 0,570] | <0,001 | 16 | 160 |

Cobertura (convex hull): Comb=3,627°², Thread-1024=68,364°², Utah=1,200°²
→ Thread es 57,0× Utah y 18,85× (≈19×) Comb.

### Exp 3 — Mann-Whitney Comb vs Thread-1024, anillo de 4°
Comb (n_el=6, Md=0,281°) vs Thread-1024 (n_el=3, Md=0,391°): diff=0,110°;
U=1204,0; p=0,009; IC95% diff=[0,072°; 0,198°].
*(Aviso de tamaño muestral: Thread-1024 solo tiene 3 electrodos en este
anillo — indicativo, no concluyente.)*

### Exp 4 — correlación excentricidad-error por método
| Método | r | r² | IC 95% | p | n_el | n_obs |
|---|---|---|---|---|---|---|
| Absoluto | 0,254 | 0,065 | [0,155; 0,349] | <0,001 | 20 | 199 |
| Relativo | 0,449 | 0,202 | [0,280; 0,573] | <0,001 | 20 | 116 |
| Ajuste forzado | 0,275 | 0,076 | [0,154; 0,395] | <0,001 | 20 | 200 |

### Exp 4 — Mann-Whitney entre métodos (todos los ensayos)
| Par | diff mediana | U | p | IC95% diff |
|---|---|---|---|---|
| Absoluto vs Relativo | 0,270° | 17327,0 | <0,0001 | [0,156; 0,413] |
| Absoluto vs Ajuste forzado | 0,992° | 37411,0 | <0,0001 | [0,825; 1,182] |
| Relativo vs Ajuste forzado | 0,722° | 18490,0 | <0,0001 | [0,519; 0,939] |

## 6. Cómo re-verificar estos números desde cero

```powershell
cd percept_mapper

# Exp 1
uv run python scripts/analysis/plot_error_vs_ecc.py `
    --session mapping_experiments/mapping_mapeo_multiples_electrodo_20260626_165130 `
    --out-dir comparison_results/exp1_error_vs_ecc

# Exp 2
uv run python scripts/analysis/compare_mapmethod.py `
    --sessions mapping_experiments/mapping_mapeo_multiples_electrodo_20260626_165130 `
               mapping_experiments/mapping_mapeo_multiples_electrodo_20260617_110153 `
               mapping_experiments/mapping_mapeo_multiples_electrodo_20260630_185135 `
    --labels "WACOM" "Pupil Core" "Pupil Neon" `
    --out-dir comparison_results/exp2_combined

# Exp 3
uv run python scripts/analysis/compare_implants.py `
    --sessions mapping_experiments/mapping_mapeo_multiples_electrodo_20260630_200505 `
               mapping_experiments/mapping_mapeo_multiples_electrodo_20260630_203952 `
               mapping_experiments/mapping_mapeo_multiples_electrodo_20260630_210035 `
    --labels "Comb 10x10" "Thread-1024" "Utah Array" `
    --out-dir comparison_results/exp3_implants_20260630 `
    --compare-ring 4

# Exp 4
uv run python scripts/analysis/compare_mapmethod.py `
    --sessions mapping_experiments/mapping_mapeo_multiples_electrodo_20260626_165130 `
               mapping_experiments/mapping_mapeo_multiples_electrodo_20260630_194015 `
               mapping_experiments/mapping_mapeo_multiples_electrodo_20260626_175247 `
    --labels "Absoluto" "Relativo" "Ajuste forzado" `
    --out-dir comparison_results/exp4_mapping_method
```

Todos usan `seed=42` fijo en `stats_utils.py` (`SEED_DEFAULT`), así que el
bootstrap es determinista — re-ejecutar debe dar exactamente los mismos
números de la sección 5, no solo números "parecidos".

## 7. Puntos concretos a verificar / posibles focos de error

Para quien (persona o IA) audite esto, los puntos donde más vale la pena
mirar con lupa:

1. **¿El bootstrap remuestrea electrodos o ensayos?** Si remuestreara
   ensayos sueltos, los IC serían artificialmente estrechos (pseudo-
   replicación). Verificar en `cluster_bootstrap_r` que el índice aleatorio
   (`rng.integers(0, n_el, n_el)`) opera sobre el array de electrodos, y que
   cada electrodo elegido aporta *todas* sus repeticiones en bloque.
2. **¿El p-valor viene del bootstrap o de `scipy.stats.pearsonr`?** Debe
   venir del bootstrap (`bootstrap_p_two_sided`). Un p-valor paramétrico
   estándar aquí sería inconsistente con el IC (ver punto 4 de la sección
   "Errores y fixes" de esta sesión: dio p no significativos que
   contradecían ICs que ya excluían el cero).
3. **Exp3 controla por anillo de excentricidad; Exp4 no lo necesita.**
   Confirmar que esa diferencia de diseño es intencional (implantes cubren
   rangos de excentricidad distintos; métodos de mapeo comparten el mismo
   conjunto de electrodos) y no un descuido.
4. **La sesión `165130` se reutiliza en Exp1/Exp2/Exp4** — no son muestras
   independientes entre sí. Cualquier meta-análisis que combine resultados
   de esos tres experimentos como si fueran independientes sería incorrecto.
5. **Exp3, anillo de 4°**: Thread-1024 solo aporta 3 electrodos. El p=0,009
   es real pero el tamaño muestral es pequeño — no sobre-interpretar como
   "diferencia sólida y generalizable".
