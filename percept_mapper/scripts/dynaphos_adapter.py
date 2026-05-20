"""
Adaptador de Dynaphos - Mapeo retinotópico usando la biblioteca Dynaphos
Envuelve la funcionalidad de Dynaphos para su uso en el simulador
"""

import sys
import os
import csv
import numpy as np
import random
from pathlib import Path
import yaml

try:
    import torch
    from dynaphos import utils
    from dynaphos.simulator import GaussianSimulator as PhospheneSimulator
    import dynaphos.cortex_models as cortex_models
    from dynaphos.cortex_models import Map

    DYNAPHOS_AVAILABLE = True
except ImportError as e:
    print(f"⚠ WARNING: Dynaphos no disponible: {e}")
    print("  Instalar: uv sync")
    DYNAPHOS_AVAILABLE = False


class DynaphosMapper:
    """
    Adaptador para usar Dynaphos en el simulador de prótesis cortical

    Proporciona mapeo retinotópico científicamente validado usando:
    - Modelo cortical de Schira et al. 2010
    - Magnificación cortical realista
    - Simulación de fosfenos gaussianos
    """

    def __init__(
        self,
        electrode_coords_file,
        screen_width=1920,
        screen_height=1080,
        dynaphos_params_file=None,
        dropout=None,
        screen_diagonal_inches=None,  # se usan solo de manera conceptual
        dist_to_screen_cm=None,  # se usan solo de manera conceptual
        vf_scope_deg=None,
        implant_id_filter='all',
    ):
        """
        Inicializa el mapeador con Dynaphos

        Args:
            electrode_coords_file: Path al archivo YAML con coordenadas corticales (mm)
                                   Ej: 'config/coords_shanks_10x16.yaml'
            screen_width: Ancho de pantalla en píxeles
            screen_height: Alto de pantalla en píxeles
            dynaphos_params_file: Path al params.yaml de Dynaphos (opcional)
            dropout: Probabilidad de dropout (0.0 a 1.0) para desactivar electrodos (opcional)
            implant_id_filter: Filtro para seleccionar coordenadas de un implante específico (opcional)
        """
        if not DYNAPHOS_AVAILABLE:
            raise RuntimeError(
                "Dynaphos no está disponible. Instalar: pip install torch dynaphos"
            )

        # Semillas globales
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

        # CLAVE: Crear un RNG explícito con numpy.random.default_rng()
        # que pasaremos a Dynaphos. Esto garantiza determinismo completo.
        self.rng = np.random.default_rng(seed=42)
        self._implant_id_filter = implant_id_filter

        print(f"[DynaphosMapper] Inicializando...")
        print(f"                 Coordenadas: {electrode_coords_file}")

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen_center = (screen_width // 2, screen_height // 2)

        def _as_float_or_none(value):
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        self._screen_diagonal_inches = _as_float_or_none(screen_diagonal_inches)
        self._dist_to_screen_cm = _as_float_or_none(dist_to_screen_cm)

        # Guardar helpers para usar más abajo (sin exponerlos)
        self._as_float_or_none = _as_float_or_none

        # ============================================
        # 1. CARGAR PARÁMETROS DE DYNAPHOS
        # ============================================
        if dynaphos_params_file is None:
            # Usar dynaphos_params.yaml incluido en el proyecto
            dynaphos_params_file = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "config",
                "dynaphos_params.yaml",
            )

        self.params = utils.load_params(dynaphos_params_file)

        # Campo visual (view_angle) para Dynaphos.
        # Importante: Dynaphos usa 'run.view_angle' para decidir qué puntos están dentro
        # del FOV (y para la escala interna en grados). Para mantener coherencia con el
        # simulador, lo fijamos al FOV total asumido: 2 * vf_scope_deg.
        vf_scope_deg = self._as_float_or_none(vf_scope_deg)
        if vf_scope_deg is None:
            vf_scope_deg = 15.0
        vf_scope_deg = abs(vf_scope_deg)
        assumed_fov_total_deg = float(2.0 * vf_scope_deg)

        if "run" not in self.params or not isinstance(self.params["run"], dict):
            self.params["run"] = {}
        self.params["run"]["origin"] = [0.0, 0.0]
        self.params["run"]["view_angle"] = float(assumed_fov_total_deg)

        # Forzar display specs a la geometría REAL de esta sesión (portabilidad)
        # Esto afecta a tamaños en dva dentro de Dynaphos (no al mapeo cortex→campo visual).
        if "display" not in self.params or not isinstance(self.params["display"], dict):
            self.params["display"] = {}
        self.params["display"]["screen_resolution"] = [
            int(self.screen_width),
            int(self.screen_height),
        ]
        if self._screen_diagonal_inches is not None:
            self.params["display"]["screen_diagonal"] = float(
                self._screen_diagonal_inches
            )
        if self._dist_to_screen_cm is not None:
            # Dynaphos usa mm
            self.params["display"]["dist_to_screen"] = (
                float(self._dist_to_screen_cm) * 10.0
            )

        # Aplicar dropout si se especifica (con semilla fija para determinismo)
        if dropout is not None:
            if "training" not in self.params:
                self.params["training"] = {}
            self.params["training"]["dropout"] = dropout
            print(f"                 Dropout configurado: {dropout}")
        else:
            # Si no se especifica, respetar el valor del params.yaml de Dynaphos
            current_dropout = self.params.get("training", {}).get("dropout", 0.0)
            print(
                f"                 Dropout: {current_dropout} (desde Dynaphos params)"
            )

        print(
            f"                 Modelo cortical: {self.params['cortex_model']['model']}"
        )

        # ============================================
        # 2. CARGAR COORDENADAS (CORTEX mm o VISUAL deg)
        # ============================================
        self._coord_source = None
        coords_path = Path(electrode_coords_file)

        if coords_path.suffix.lower() == ".csv":
            x_deg, y_deg, meta = self._load_visual_coordinates_csv(
                coords_path,
                implant_id_filter=self._implant_id_filter,)
            self.num_electrodes = len(x_deg)
            self._coord_source = {
                "type": "phoslab_csv_visual_deg",
                **meta,
            }
            print(
                f"                 Electrodos cargados (CSV visual, grados): {self.num_electrodes}"
            )

            # En este modo no hay coordenadas corticales.
            self.coordinates_cortex = None

            # Pasar directamente coordenadas visuales (en grados) a Dynaphos.
            self.phosphene_coords = Map(x=x_deg, y=y_deg)
            self.visual_coords_deg = self.phosphene_coords.cartesian

        else:
            x_mm, y_mm = self._load_electrode_coordinates(electrode_coords_file)
            self.num_electrodes = len(x_mm)
            self._coord_source = {
                "type": "yaml_cortex_mm",
            }
            print(
                f"                 Electrodos cargados (YAML cortex, mm): {self.num_electrodes}"
            )

            # Crear objeto Map de Dynaphos (cortex)
            self.coordinates_cortex = Map(x=x_mm, y=y_mm)

            # ============================================
            # 3. MAPEAR CORTEX → CAMPO VISUAL
            # ============================================
            # Dynaphos hace el mapeo retinotópico con magnificación cortical
            # CLAVE: Pasar self.rng para garantizar determinismo
            self.phosphene_coords = (
                cortex_models.get_visual_field_coordinates_from_cortex(
                    self.params["cortex_model"],
                    self.coordinates_cortex,
                    rng=self.rng,  # ← RNG EXPLÍCITO PARA DETERMINISMO
                )
            )

            # Obtener coordenadas visuales en grados
            self.visual_coords_deg = self.phosphene_coords.cartesian  # (x_deg, y_deg)

        # ============================================
        # 4. CONFIGURAR PARÁMETROS VISUALES
        # ============================================
        # Extraer del params.yaml de Dynaphos
        fov_params = self.params.get("fov", {})
        if not isinstance(fov_params, dict):
            fov_params = {}

        # Campo visual en grados para conversión deg→px.
        # Convenio único (pipeline):
        # - vf_scope_deg es el semiancho: rango [-vf_scope_deg, +vf_scope_deg]
        # - assumed_fov_total_deg = 2 * vf_scope_deg
        # NOTA: vf_scope_deg ya fue normalizado arriba para fijar run.view_angle.

        self.fov_x_deg = [-(vf_scope_deg), (vf_scope_deg)]
        self.fov_y_deg = [-(vf_scope_deg), (vf_scope_deg)]

        # Calcular píxeles por grado
        fov_width_deg = self.fov_x_deg[1] - self.fov_x_deg[0]
        fov_height_deg = self.fov_y_deg[1] - self.fov_y_deg[0]

        self.pixels_per_degree_x = screen_width / fov_width_deg
        self.pixels_per_degree_y = screen_height / fov_height_deg

        print(f"                 Campo visual: {fov_width_deg}° × {fov_height_deg}°")
        print(
            f"                 Escala: {self.pixels_per_degree_x:.1f} px/grado (X), {self.pixels_per_degree_y:.1f} px/grado (Y)"
        )

        # ============================================
        # 5. CONVERTIR A PÍXELES
        # ============================================
        self.electrode_positions_visual_px = self._convert_degrees_to_pixels()

        # ============================================
        # 6. SELECCIÓN DE ELECTRODOS ACTIVOS
        # ============================================
        self.active_electrodes = np.zeros(self.num_electrodes, dtype=bool)

        # ============================================
        # 7. INICIALIZAR SIMULADOR DE FOSFENOS (opcional)
        # ============================================
        # CLAVE: Pasar self.rng para garantizar determinismo
        self.simulator = PhospheneSimulator(
            self.params,
            self.phosphene_coords,
            rng=self.rng,  # ← RNG EXPLÍCITO PARA DETERMINISMO
        )

        print("[DynaphosMapper] OK Inicializado correctamente")

        self._display_metadata = {
            "screen_resolution_px": [int(self.screen_width), int(self.screen_height)],
            "screen_center_px": [
                int(self.screen_center[0]),
                int(self.screen_center[1]),
            ],
            # Convenio del pipeline (alineado con phosLab):
            # VF_scope es el semiancho (max ecc) en grados.
            "vf_scope_deg": float(vf_scope_deg),
            "assumed_fov_total_deg": float(assumed_fov_total_deg),
            "fov_x_deg_range": [float(self.fov_x_deg[0]), float(self.fov_x_deg[1])],
            "fov_y_deg_range": [float(self.fov_y_deg[0]), float(self.fov_y_deg[1])],
            "pixels_per_degree_x": float(self.pixels_per_degree_x),
            "pixels_per_degree_y": float(self.pixels_per_degree_y),
            "screen_diagonal_inches": self._screen_diagonal_inches,
            "dist_to_screen_cm": self._dist_to_screen_cm,
            "coord_source": dict(self._coord_source) if self._coord_source else None,
        }

    def get_display_metadata(self):
        """Devuelve un dict JSON-serializable con la geometría usada en esta sesión."""
        return dict(self._display_metadata)

    def _load_electrode_coordinates(self, coords_file):
        """
        Carga coordenadas de electrodos desde YAML

        Args:
            coords_file: Path al archivo YAML

        Returns:
            tuple: (x_array, y_array) coordenadas en mm
        """
        coords_path = Path(coords_file)

        if not coords_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {coords_file}")

        with open(coords_path, "r") as f:
            data = yaml.safe_load(f)

        x = np.array(data["x"])
        y = np.array(data["y"])

        if len(x) != len(y):
            raise ValueError("Número de coordenadas x e y no coincide")

        return x, y

    def _rf_to_xy_deg(self, ecc_deg, polar_deg, pol_convention="standard"):
        """Convierte (ecc, polar) en grados a (x_deg, y_deg).

        Convenciones (alineadas con PhosLab):
        - standard: x=ecc*cos(pol), y=ecc*sin(pol)
        - neuropythy: x=ecc*sin(pol), y=ecc*cos(pol)
        """
        pol_rad = np.radians(float(polar_deg))
        if str(pol_convention).strip().lower() == "neuropythy":
            return float(ecc_deg * np.sin(pol_rad)), float(ecc_deg * np.cos(pol_rad))
        return float(ecc_deg * np.cos(pol_rad)), float(ecc_deg * np.sin(pol_rad))

    def _load_visual_coordinates_csv(self, csv_path: Path, implant_id_filter: str ='all'):
        """Carga coordenadas visuales (en grados) desde un CSV exportado por PhosLab.

        Columnas soportadas (mínimo):
        - ID: `electrode_index` o `contact_id` (si falta, se usa el orden de filas 0..N-1)
        - Coordenadas: (`x_deg`,`y_deg`) o (`ecc_deg`,`polar_deg`)

        La salida queda ordenada para que `electrode_index == id` (0-index).
        """
        if not csv_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {csv_path}")

        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if rows:
            field_map_tmp = {str(k).strip().lower(): k for k in (rows[0].keys() or [])}
            implant_id_key = field_map_tmp.get("implant_id")
            if implant_id_key is not None:
                implant_ids_found = list(dict.fromkeys(
                    str(r.get(implant_id_key,'')).strip() for r in rows
                ))
                print (f" implantes encontrados en csv: {implant_ids_found}")
                if implant_id_filter != 'all':
                    rows = [r for r in rows if str(r.get(implant_id_key,'')).strip() == str(implant_id_filter).strip()]
                    print (f"filtrando por id_implante= '{implant_id_filter}': {len(rows)} filas restantes")

                    if not rows:
                        raise ValueError(
                            f"no se encontraronn filas con implant_id='{implant_id_filter}'"
                            f"implantes disponibles= {implant_ids_found}"
                        )

        # Detectar claves (case-insensitive)
        field_map = {str(k).strip().lower(): k for k in (rows[0].keys() or [])}
        id_key = field_map.get("electrode_index") or field_map.get("contact_id")
        x_key = field_map.get("x_deg")
        y_key = field_map.get("y_deg")
        ecc_key = field_map.get("ecc_deg") or field_map.get("ecc")
        pol_key = (
            field_map.get("polar_deg") or field_map.get("pol") or field_map.get("polar")
        )
        polconv_key = field_map.get("pol_convention")

        if (x_key is None or y_key is None) and (ecc_key is None or pol_key is None):
            raise ValueError(
                "CSV inválido: se requiere (x_deg,y_deg) o (ecc_deg,polar_deg). "
                f"Columnas encontradas: {list(rows[0].keys())}"
            )

        entries = []
        pol_convention = "standard"
        implant_id_key = field_map.get("implant_id")

        for row_i, row in enumerate(rows):
            implant_id = str(row.get(implant_id_key, 'default')).strip() if implant_id_key else 'default'
            if id_key is None:
                idx = int(row_i)
            else:
                raw = row.get(id_key)
                if raw is None or str(raw).strip() == "":
                    raise ValueError(f"Fila {row_i}: ID vacío en columna '{id_key}'.")
                idx = int(float(str(raw).strip()))

            if polconv_key is not None:
                pol_convention = str(row.get(polconv_key) or pol_convention)

            if x_key is not None and y_key is not None:
                x_deg = float(row.get(x_key))
                y_deg = float(row.get(y_key))
            else:
                ecc_deg = float(row.get(ecc_key))
                polar_deg = float(row.get(pol_key))
                x_deg, y_deg = self._rf_to_xy_deg(
                    ecc_deg, polar_deg, pol_convention=pol_convention
                )

            entries.append((implant_id, idx, x_deg, y_deg))

        unique_keys = [(e[0], e[1]) for e in entries]
        if len(set(unique_keys)) != len(unique_keys):
            raise ValueError(
                "CSV inválido: combinaciones de (implant_id, electrode_index/contact_id) duplicadas"
            )
        self._electrode_index_map = {}
        max_id = max(e[1] for e in entries) if entries else -1

        implant_ids_in_entries = list(dict.fromkeys(e[0] for e in entries))
        if len(implant_ids_in_entries) == 1:
            n= max_id + 1
            x = np.full((n,), np.nan, dtype=np.float64)
            y = np.full((n,), np.nan, dtype=np.float64)
            for implant_id, idx, x_deg, y_deg in entries:
                x[idx] = x_deg
                y[idx] = y_deg
                self._electrode_index_map[idx] = (implant_id, idx)
        else:
            implant_offsets = {}
            current_offset = 0
            for imp_id in implant_ids_in_entries:
                implant_offsets[imp_id] = current_offset
                max_idx_for_implant = max(
                    e[1] for e in entries if e[0] == imp_id)
                current_offset += max_idx_for_implant + 1

            n = current_offset
            x = np.full((n,), np.nan, dtype=np.float64)
            y = np.full((n,), np.nan, dtype=np.float64)
            for implant_id, idx, x_deg, y_deg in entries:
                global_idx = implant_offsets[implant_id] + idx
                x[global_idx] = x_deg
                y[global_idx] = y_deg
                self._electrode_index_map[global_idx] = (implant_id, idx)
            print(f"                Offsets:")
            for imp_id, offset in implant_offsets.items():
                print(f"                 - Implant '{imp_id}': offset {offset}")

        meta = {
            "path": str(csv_path),
            "pol_convention": str(pol_convention).strip().lower(),
            "id_column": id_key if id_key is not None else "row_index",
            "coord_columns": (
                "x_deg,y_deg" if (x_key and y_key) else "ecc_deg,polar_deg"
            ),
            "n_electrodes_total": int(n),
            "n_electrodes_valid": int(np.sum(np.isfinite(x) & np.isfinite(y))),
            "missing_indices": [i for i in range(n) if not (np.isfinite(x[i]) and np.isfinite(y[i]))],
            "implants": implant_ids_in_entries,
            "electrode_index_map": {
                str(k): list(v) for k, v in self._electrode_index_map.items()
            }
        
        }
        return x, y, meta

    def _convert_degrees_to_pixels(self):
        """
        Convierte coordenadas visuales de grados a píxeles

        Returns:
            numpy.ndarray: Coordenadas en píxeles (N, 2)
        """
        x_deg, y_deg = self.visual_coords_deg

        # Convertir grados → píxeles
        # Dynaphos usa origen en centro del FOV, igual que nuestra pantalla
        x_px = self.screen_center[0] + x_deg * self.pixels_per_degree_x
        y_px = self.screen_center[1] - y_deg * self.pixels_per_degree_y  # Invertir Y

        positions_px = np.column_stack([x_px, y_px])

        return positions_px

    def set_active_electrodes(self, electrode_indices):
        """
        Activa electrodos específicos

        Args:
            electrode_indices: Lista de índices de electrodos (0-indexed)
        """
        # Resetear todos
        self.active_electrodes[:] = False

        # Activar seleccionados
        for idx in electrode_indices:
            if 0 <= idx < self.num_electrodes:
                self.active_electrodes[idx] = True
            else:
                print(
                    f"⚠ WARNING: Índice {idx} fuera de rango (0-{self.num_electrodes-1})"
                )

        num_active = np.sum(self.active_electrodes)
        print(
            f"[DynaphosMapper] Electrodos activos: {num_active}/{self.num_electrodes}"
        )

    def _get_valid_electrode_indices(self):
        """Devuelve lista de índices globales que tienen coordenadas válidas (no NaN)."""
        px = self.electrode_positions_visual_px
        return [
            int(i)
            for i in range(self.num_electrodes)
            if np.isfinite(px[i][0]) and np.isfinite(px[i][1])
        ]

    def configure_electrodes_from_selection(self, selection_config):
        """
        Configura electrodos activos basándose en el diccionario de configuración

        Args:
            selection_config: Diccionario con la configuración de electrode_selection
        """
        mode = selection_config.get("mode", "manual")

        if mode == "manual":
            # Modo manual: usar lista de índices
            indices = selection_config.get("indices", [])
            self.set_active_electrodes(indices)

        elif mode == "all":
            # Todos los electrodos con coordenadas válidas en el CSV
            indices = self._get_valid_electrode_indices()
            print(
                f"[ALL] Seleccionados {len(indices)} electrodos válidos: {indices[:8]}{'...' if len(indices) > 8 else ''}"
            )
            self.set_active_electrodes(indices)

        elif mode == "pattern":
            # Modo pattern: usar patrones predefinidos
            pattern = selection_config.get("pattern", "random")
            n_electrodes = selection_config.get("n_electrodes", 10)

            if pattern == "random":
                # N electrodos aleatorios
                indices = random.sample(
                    range(self.num_electrodes), min(n_electrodes, self.num_electrodes)
                )
            elif pattern == "grid":
                # Patrón de rejilla: cada N electrodos
                step = max(1, self.num_electrodes // n_electrodes)
                indices = list(range(0, self.num_electrodes, step))[:n_electrodes]
            elif pattern == "center":
                # Electrodos centrales (menor excentricidad)
                visual_x, visual_y = self.visual_coords_deg
                eccentricities = np.sqrt(visual_x**2 + visual_y**2)
                indices = np.argsort(eccentricities)[:n_electrodes].tolist()
            elif pattern == "periphery":
                # Electrodos periféricos (mayor excentricidad)
                visual_x, visual_y = self.visual_coords_deg
                eccentricities = np.sqrt(visual_x**2 + visual_y**2)
                indices = np.argsort(eccentricities)[-n_electrodes:].tolist()
            else:
                print(f"⚠ WARNING: Patrón '{pattern}' no reconocido, usando 'random'")
                indices = random.sample(
                    range(self.num_electrodes), min(n_electrodes, self.num_electrodes)
                )

            self.set_active_electrodes(indices)

        elif mode == "range":
            # Modo range: selecciona electrodos cuyo electrode_index esté
            # dentro del rango [start, end) con paso step.
            # Se excluyen automáticamente índices sin coordenadas en el CSV (NaN).
            start = selection_config.get("start", 0)
            end = selection_config.get("end", self.num_electrodes)
            step = selection_config.get("step", 1)
            requested = set(range(int(start), int(end), int(step)))
            valid_set = set(self._get_valid_electrode_indices())
            indices = sorted(requested & valid_set)
            skipped = sorted(requested - valid_set)
            if skipped:
                print(f"[RANGE] Índices pedidos sin coordenadas en el CSV (ignorados): {skipped}")
            print(f"[RANGE] Seleccionados {len(indices)} electrodos: {indices[:8]}{'...' if len(indices) > 8 else ''}")
            self.set_active_electrodes(indices)

        else:
            print(f"⚠ WARNING: Modo '{mode}' no reconocido, usando modo manual con []")
            self.set_active_electrodes([])

    def get_active_phosphene_positions(self):
        """
        Obtiene posiciones visuales (píxeles) de electrodos activos

        Returns:
            list: Lista de tuplas (x, y) en píxeles
        """
        active_indices = np.where(self.active_electrodes)[0]
        positions = []

        for idx in active_indices:
            x, y = self.electrode_positions_visual_px[idx]
            if not (np.isfinite(x) and np.isfinite(y)):
                continue  # Saltar coordenadas no válidas
            positions.append((int(x), int(y)))

        return positions

    def get_phosphene_position(self, electrode_index):
        """
        Obtiene la posición visual de un electrodo específico

        Args:
            electrode_index: Índice del electrodo

        Returns:
            tuple: (x, y) posición en píxeles
        """
        if not (0 <= electrode_index < self.num_electrodes):
            raise ValueError(
                f"Índice de electrodo fuera de rango: {electrode_index} (debe estar entre 0 y {self.num_electrodes-1})"
            )

        x, y = self.electrode_positions_visual_px[electrode_index]
        if not (np.isfinite(x) and np.isfinite(y)):
            raise ValueError(
                f"Coordenadas no válidas para electrodo {electrode_index}: (x={x}, y={y})"
            )

        return (int(x), int(y))

    def get_electrode_info(self, electrode_index):
        """
        Información detallada de un electrodo

        Args:
            electrode_index: Índice del electrodo

        Returns:
            dict: Información del electrodo
        """
        if not (0 <= electrode_index < self.num_electrodes):
            raise ValueError(f"Índice fuera de rango: {electrode_index}")

        if self.coordinates_cortex is None:
            cortex_x, cortex_y = None, None
        else:
            cortex_x, cortex_y = self.coordinates_cortex.cartesian
        visual_deg_x, visual_deg_y = self.visual_coords_deg
        visual_px = self.electrode_positions_visual_px[electrode_index]

        # Calcular excentricidad
        eccentricity_deg = np.sqrt(
            visual_deg_x[electrode_index] ** 2 + visual_deg_y[electrode_index] ** 2
        )

        # Convertir a tipos Python para que sea JSON-serializable (evitar numpy.float64, etc.)
        return {
            "index": int(electrode_index),
            "cortex_position_mm": (
                None
                if cortex_x is None
                else [
                    float(cortex_x[electrode_index]),
                    float(cortex_y[electrode_index]),
                ]
            ),
            "visual_position_deg": [
                float(visual_deg_x[electrode_index]),
                float(visual_deg_y[electrode_index]),
            ],
            "visual_position_px": [int(visual_px[0]), int(visual_px[1])],
            "eccentricity_deg": float(eccentricity_deg),
            "is_active": bool(self.active_electrodes[electrode_index]),
        }

    def simulate_phosphenes(self, current_amplitudes_uA=None):
        """
        Simula la imagen de fosfenos usando Dynaphos

        Args:
            current_amplitudes_uA: Array con corrientes por electrodo (microamperios)
                                   Si None, usa valores de params.yaml o 90uA por defecto

        Returns:
            numpy.ndarray: Imagen simulada de fosfenos
        """
        if current_amplitudes_uA is None:
            # Cargar corrientes desde params.yaml
            stimulation_currents = self.params.get(
                "stimulation_currents_uA", [90] * self.num_electrodes
            )
            stim = torch.zeros(self.num_electrodes)
            active_indices = np.where(self.active_electrodes)[0]
            for idx in active_indices:
                current = (
                    stimulation_currents[idx] if idx < len(stimulation_currents) else 90
                )
                stim[idx] = float(current) * 1e-6  # Convertir a amperios
        else:
            # Convertir a tensor directamente en microamperios
            stim = torch.tensor(current_amplitudes_uA, dtype=torch.float32) * 1e-6

        # Simular
        self.simulator.reset()
        phosphene_image = self.simulator(stim)

        return phosphene_image.numpy()

    def visualize_mapping(self, output_file="dynaphos_mapping.png"):
        """
        Visualiza el mapeo cortical → visual

        Args:
            output_file: Nombre del archivo de salida
        """
        try:
            import matplotlib.pyplot as plt
            from dynaphos.plotting import plot_cortex_flatmap, plot_fov
        except ImportError:
            print("⚠ Matplotlib no disponible")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # ============================================
        # PANEL 1: ESPACIO CORTICAL
        # ============================================
        cortex_x, cortex_y = self.coordinates_cortex.cartesian
        ax1.scatter(
            cortex_x,
            cortex_y,
            c=self.active_electrodes,
            cmap="RdYlGn",
            s=50,
            alpha=0.7,
        )
        plot_cortex_flatmap(self.params, ax1)
        ax1.set_title("Electrodos en Córtex Visual")
        ax1.set_xlabel("X (mm)")
        ax1.set_ylabel("Y (mm)")

        # ============================================
        # PANEL 2: CAMPO VISUAL
        # ============================================
        visual_x, visual_y = self.visual_coords_deg
        ax2.scatter(
            visual_x,
            visual_y,
            c=self.active_electrodes,
            cmap="RdYlGn",
            s=50,
            alpha=0.7,
        )
        plot_fov(self.params, ax2)
        ax2.set_title("Fosfenos en Campo Visual")
        ax2.set_xlabel("X (grados)")
        ax2.set_ylabel("Y (grados)")

        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        print(f"[DynaphosMapper] OK Visualización guardada: {output_file}")
        plt.close()


# ============================================
# FUNCIONES DE UTILIDAD
# ============================================


def _resolve_simulador_path(path_like: str | Path) -> Path:
    """Resuelve rutas relativas respecto a la raíz del proyecto `simulador/`."""
    p = Path(path_like)
    if p.is_absolute():
        return p
    simulador_root = Path(__file__).resolve().parent.parent
    return (simulador_root / p).resolve()


# FUNCION DE CONFIGURACION DE ELECTRODOS
def load_active_electrodes_config(config_file="config/params.yaml"):
    """
    Carga configuración de electrodos activos desde params.yaml

    Args:
        config_file: Ruta al archivo params.yaml

    Returns:
        dict: Configuración de mapeo retinotópico (incluye dropout)
    """
    config_path = _resolve_simulador_path(config_file)
    with open(config_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    # Devolver la sección de mapeo retinotópico (incluye dropout)
    return params.get("retinotopic_mapping", {})


# FUNCION DE TIEMPOS
def load_timing_config(config_file="config/params.yaml"):
    """
    Carga configuración de tiempos del experimento desde params.yaml

    Args:
        config_file: Ruta al archivo params.yaml

    Returns:
        dict: Configuración de tiempos (en milisegundos)
    """
    config_path = _resolve_simulador_path(config_file)
    with open(config_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    # Devolver la sección de timing con valores por defecto
    timing = params.get("timing", {})

    return {
        "prestimulation_ms": timing.get("prestimulation", 200),
        "stimulation_ms": timing.get("stimulation", 400),
        "poststimulation_ms": timing.get("poststimulation", 100),
        "interstimulation_ms": timing.get("interstimulation", 500),
        "max_fixation_wait_ms": timing.get("max_fixation_wait", 10000),
    }


# FUNCION DE PATRONES DE ELECTRODOS
def create_electrode_pattern(pattern_type, num_electrodes):
    """
    Crea patrones de selección de electrodos

    Args:
        pattern_type: 'all', 'random', 'grid', 'center', 'periphery'
        num_electrodes: Número total de electrodos

    Returns:
        list: Índices de electrodos a activar
    """
    if pattern_type == "all":
        return list(range(num_electrodes))

    elif pattern_type == "random":
        n = min(10, num_electrodes)
        return np.random.choice(num_electrodes, size=n, replace=False).tolist()

    elif pattern_type == "grid":
        step = max(1, num_electrodes // 20)
        return list(range(0, num_electrodes, step))

    elif pattern_type == "center":
        start = num_electrodes // 4
        end = 3 * num_electrodes // 4
        return list(range(start, end))

    elif pattern_type == "periphery":
        quarter = num_electrodes // 4
        return list(range(quarter)) + list(range(3 * quarter, num_electrodes))

    else:
        raise ValueError(f"Patrón desconocido: {pattern_type}")
