# Pipeline de Protesis Cortical Visual

## Que es este proyecto
Este repositorio contiene un pipeline completo para explorar una protesis cortical visual de forma simulada. Sirve para:
- definir un implante y sus electrodos,
- generar las coordenadas de fosfenos que producirian esos electrodos,
- ejecutar experimentos con un usuario (o un input de prueba),
- analizar los resultados,
- y entrenar modelos que corrigen el mapa de fosfenos.

En otras palabras: el pipeline intenta aproximar la experiencia visual que producira una protesis cortical y aprender una correccion para que el mapa sea mas preciso.

## Que es un fosfeno (explicacion rapida)
Un fosfeno es un punto de luz percibido al estimular el cortex visual. Cada electrodo de un implante puede producir un fosfeno en una posicion del campo visual. El objetivo es predecir y corregir la posicion de esos fosfenos.

## Modulos del pipeline y para que sirven
El pipeline se organiza en cuatro modulos, que se ejecutan en orden:

1) implant_explorer (phosLab) (diseno del implante y exportacion de coordenadas)
	- Interfaz 3D para definir el implante y ver la cobertura en campo visual.
	- Exporta un CSV con coordenadas (x_deg, y_deg) de cada electrodo.

2) Simulador (experimento y captura de respuestas)
	- Ejecuta un experimento de estimulacion ICMS.
	- Muestra el fosfeno simulado y registra trazos o fijaciones del usuario.
	- Guarda los resultados por electrodo.

3) Analisis
	- Resume resultados de los experimentos.
	- Calcula centroides y metricas por electrodo.
	- Genera archivos JSON con los resultados de cada experimento.

4) Aprendizaje (correccion del mapa)
	- Construye un dataset con pares: prediccion vs observacion real.
	- Entrena un modelo bayesiano (y opcionalmente un modelo neural).
	- Genera un mapa corregido y metricas comparativas.

Todo el flujo se puede controlar desde el lanzador grafico.

## Estructura y carpetas clave
- launcher/: interfaz grafica que orquesta todo el pipeline.
- implant_explorer/: herramienta para definir el implante y exportar el CSV de coordenadas.
- percept_mapper/: motor del experimento, analisis y aprendizaje.
- percept_mapper/config/params.yaml: configuracion central del experimento.
- percept_mapper/mapping_experiments/: resultados del modo mapping.
- percept_mapper/logs/: resultados del modo standard y analisis por experimento.
- percept_mapper/learning_results/: modelos y metricas de aprendizaje.

## Flujo del pipeline paso a paso
### 1) Generacion de coordenadas (implant_explorer / phosLab)
Se define el implante y se exporta un CSV con las coordenadas de los electrodos. El lanzador vigila la carpeta de implant_explorer y copia el CSV a percept_mapper/config/.

### 2) Configuracion del experimento
Desde el lanzador se selecciona el CSV y se actualizan parametros en percept_mapper/config/params.yaml:
- modo de experimento (mapping o standard),
- electrodos a estimular,
- tiempos, corrientes y frecuencia,
- parametros de entrada y pantalla.

### 3) Simulador y analisis
El simulador ejecuta el experimento y guarda:
- trazos y metadatos por electrodo,
- resultados de analisis en analysis_results.json,
- resumenes en mapping_experiments/ o logs/ segun el modo.

### 4) Aprendizaje
El aprendizaje lee los resultados y genera un dataset con pares:
- prediccion del modelo (pred_x, pred_y),
- observacion del usuario (obs_x, obs_y),
- error = obs - pred.

Se entrenan dos modelos:
- Bayesiano: estima el sesgo sistematico del modelo y corrige predicciones.
- Red neuronal: aprende una funcion no lineal pred -> obs.

Los resultados se guardan en percept_mapper/learning_results/:
- bayesian_model.json
- neural_model.pt (si hay red neuronal disponible)
- evaluation_metrics.json
- visual_field_comparison.png
- error_comparison.png
- neural_training.png

## Como se lanza el aprendizaje
El lanzador llama a:

uv run python run_learning.py --model both

Este script:
1) Lee percept_mapper/config/params.yaml para saber donde estan los datos.
2) Carga todos los experimentos en mapping_experiments/ y logs/.
3) Construye el dataset con predicciones y observaciones.
4) Entrena el modelo bayesiano y, si esta disponible, el neural.
5) Evalua errores y guarda metricas y graficas.

Si el entorno no tiene PyTorch, el modelo neural se omite y se continua solo con el bayesiano.

## Ejecucion
### Lanzador grafico
Desde la raiz del repo:

uv run --project launcher python main.py

### implant_explorer (phosLab) (opcional, standalone)
Desde la raiz del repo:

uv run --project implant_explorer python src/implant_explorer.py

### Simulador (opcional, standalone)
Desde la raiz del repo:

uv run --project percept_mapper python main.py

### Aprendizaje (opcional, standalone)
Desde la raiz del repo:

uv run --project percept_mapper python run_learning.py --model both

Modelos disponibles: bayesian, neural, both.

## Requisitos
- Python 3.10 o 3.11.
- uv instalado y operativo en el PATH.
- Dependencias definidas en percept_mapper/pyproject.toml.
- Para el modelo neural se requiere torch.
- dynaphos se instala con uv (ver instalacion).

## Instalacion en otro dispositivo (Windows)
Desde la carpeta raiz del repo:

1) Instala Python 3.11 y uv (https://astral.sh/uv).

2) Sincroniza dependencias por modulo (workspace):
	uv sync --project launcher
	uv sync --project implant_explorer
	uv sync --project percept_mapper

3) Nota sobre torch (modelo neural):
	- El proyecto incluye torch como dependencia.
	- Si falla por una ruta local en percept_mapper/pyproject.toml, elimina o ajusta
	  la entrada de tool.uv.sources para torch y vuelve a ejecutar uv sync.
	- Alternativa manual:
	  uv run --project percept_mapper pip install torch --index-url https://download.pytorch.org/whl/cpu

## Datos incluidos
- La carpeta implant_explorer/data/ se incluye en el repositorio para que phosLab funcione sin descargas adicionales.
- Si necesitas reducir el tamano del repo, puedes mover esos datos a un paquete externo y ajustar las rutas en implant_explorer/src/dataset_adapters.py.

## Notas de reproduccion y trazabilidad
- El CSV de phosLab se copia a percept_mapper/config/ y se referencia en params.yaml.
- Cada experimento guarda metadatos y resultados por electrodo.
- El aprendizaje usa todos los experimentos disponibles para ajustar modelos.
- Las metricas y figuras se generan en learning_results/ (por defecto ignoradas en Git).
