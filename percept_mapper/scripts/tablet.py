"""
Tablet de dibujo - Versión simplificada
Pantalla negra donde el usuario dibuja y confirma con ENTER
"""

import pygame


class DrawingTablet:
    """
    Interfaz de dibujo simple para capturar la respuesta del usuario
    """

    def __init__(
        self,
        screen_width,
        screen_height,
        brush_size=5,
        brush_color=(255, 255, 0),
        mode="both",
        instructions_text=None,
        hide_cursor=False,
        cursor_clip_rect=None,
        allow_empty=False,
    ):
        """
        Inicializa la tablet de dibujo.

        Args:
            screen_width: Ancho de la pantalla principal
            screen_height: Alto de la pantalla principal
            brush_size: Tamaño del pincel (radio en píxeles). Default: 5
            brush_color: Color del pincel en RGB. Default: (255, 255, 0) amarillo
            mode: "mouse" | "tablet" | "both". Solo afecta a la UI (texto,
                  cursor); pygame no distingue eventos de ratón y stylus, así
                  que ambos dispositivos siempre funcionan físicamente.
            instructions_text: Texto opcional para el subtítulo. Si es None,
                  se elige uno por defecto según `mode`.
            hide_cursor: Si True, oculta el cursor del sistema durante el
                  dibujo (útil en modo tablet si el stylus muestra su propio
                  indicador en pantalla).
            allow_empty: Si True, permite confirmar con ENTER aunque no haya trazos.
        """
        print(f"[DrawingTablet] Inicializando (mode={mode})...")

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.mode = mode
        self.hide_cursor = bool(hide_cursor)
        self._cursor_clip_rect = cursor_clip_rect
        self._cursor_clip_active = False
        self.allow_empty = bool(allow_empty)

        # Colores
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.YELLOW = (255, 255, 0)

        # Canvas de dibujo (toda la pantalla es el canvas)
        self.canvas = pygame.Surface((screen_width, screen_height))
        self.canvas.fill(self.BLACK)

        # Configuración del pincel (desde params.yaml o valores por defecto)
        self.brush_size = brush_size
        self.brush_color = brush_color

        # Texto de instrucciones según modo
        if instructions_text is None:
            instructions_text = {
                "mouse": "Dibuja con el ratón y presiona ENTER",
                "tablet": "Dibuja con la tablet y presiona ENTER",
                "both": "Dibuja (ratón o tablet) y presiona ENTER",
            }.get(mode, "Dibuje ahora y presione ENTER")
        self.title_text = instructions_text

        # Estado del dibujo
        self.drawing = False
        self.strokes = []  # Lista de trazos
        self.current_stroke = []  # Trazo actual
        self.finished = False
        self.last_status = "unknown"
        self._cursor_prev_visible = None

        # Fuente para título
        self.font = pygame.font.Font(None, 64)

        if self.hide_cursor:
            self._cursor_prev_visible = pygame.mouse.get_visible()
            pygame.mouse.set_visible(False)

        print("[DrawingTablet] ✓ Inicializado")

    def update(self, screen, events):
        """
        Actualiza la tablet de dibujo

        Args:
            screen: Superficie de pygame donde dibujar
            events: Lista de eventos de pygame

        Returns:
            tuple: (finished: bool, canvas: Surface or None)
        """
        # Procesar eventos
        for event in events:
            # Ratón presionado - empezar a dibujar
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.drawing = True
                self.current_stroke = [pygame.mouse.get_pos()]

            # Ratón soltado - terminar trazo
            elif event.type == pygame.MOUSEBUTTONUP:
                if self.drawing:
                    self.drawing = False
                    if len(self.current_stroke) > 0:
                        self.strokes.append(self.current_stroke.copy())
                    self.current_stroke = []

            # Teclado
            elif event.type == pygame.KEYDOWN:
                # ENTER - confirmar dibujo
                if event.key == pygame.K_RETURN:
                    if len(self.strokes) > 0:
                        print(
                            f"[DrawingTablet] Dibujo confirmado ({len(self.strokes)} trazos)"
                        )
                        self.finished = True
                        self.last_status = "ok"
                        self._release_cursor_clip()
                        return (True, self.canvas.copy())
                    if self.allow_empty:
                        print("[DrawingTablet] Confirmando dibujo vacio")
                        self.finished = True
                        self.last_status = "empty"
                        self._release_cursor_clip()
                        return (True, self.canvas.copy())
                    else:
                        print("[DrawingTablet] No hay trazos para confirmar")

                # X - borrar último trazo
                elif event.key == pygame.K_x:
                    if len(self.strokes) > 0:
                        self.strokes.pop()
                        self._redraw_canvas()
                        print(
                            f"[DrawingTablet] Trazo borrado (quedan {len(self.strokes)})"
                        )
                    else:
                        print("[DrawingTablet] No hay trazos para borrar")

        # Dibujar mientras mantiene presionado el ratón
        if self.drawing:
            pos = pygame.mouse.get_pos()
            pygame.draw.circle(self.canvas, self.brush_color, pos, self.brush_size)
            self.current_stroke.append(pos)

        # Dibujar interfaz
        self.draw(screen)

        return (False, None)

    def draw(self, screen):
        """Dibuja la interfaz en pantalla"""
        # Copiar canvas a la pantalla
        screen.blit(self.canvas, (0, 0))

        # Título en la parte superior
        title_surface = self.font.render(self.title_text, True, self.WHITE)
        title_rect = title_surface.get_rect(center=(self.screen_width // 2, 50))
        screen.blit(title_surface, title_rect)

        # Instrucción X para borrar
        instruction = "X = Borrar último trazo"
        instruction_surface = self.font.render(instruction, True, self.WHITE)
        instruction_rect = instruction_surface.get_rect(
            center=(self.screen_width // 2, 100)
        )
        screen.blit(instruction_surface, instruction_rect)

    def _redraw_canvas(self):
        """Redibuja el canvas desde cero con los trazos guardados"""
        # Limpiar canvas
        self.canvas.fill(self.BLACK)

        # Redibujar todos los trazos
        for stroke in self.strokes:
            for point in stroke:
                pygame.draw.circle(
                    self.canvas, self.brush_color, point, self.brush_size
                )

    def reset(self):
        """Resetea la tablet para un nuevo dibujo"""
        self.canvas.fill(self.BLACK)
        self.strokes = []
        self.current_stroke = []
        self.drawing = False
        self.finished = False
        self.last_status = "unknown"
        self._apply_cursor_clip()
        print("[DrawingTablet] Reseteado para nuevo dibujo")

    def close(self):
        """Libera el confinamiento del cursor si está activo (idempotente)."""
        self._release_cursor_clip()
        if self._cursor_prev_visible is not None:
            pygame.mouse.set_visible(self._cursor_prev_visible)
            self._cursor_prev_visible = None

    def _apply_cursor_clip(self):
        if self._cursor_clip_rect is None or self._cursor_clip_active:
            return
        try:
            from scripts.cursor_clip import clip_cursor

            if clip_cursor(self._cursor_clip_rect):
                self._cursor_clip_active = True
                print(f"[DrawingTablet] Cursor confinado a {self._cursor_clip_rect}")
        except Exception as e:
            print(f"[DrawingTablet] ⚠ no se pudo aplicar cursor clip: {e}")

    def _release_cursor_clip(self):
        if not self._cursor_clip_active:
            return
        try:
            from scripts.cursor_clip import clip_cursor

            clip_cursor(None)
        except Exception as e:
            print(f"[DrawingTablet] ⚠ no se pudo liberar cursor clip: {e}")
        finally:
            self._cursor_clip_active = False


class ForcedAdjustmentTablet:
    """Pantalla de ajuste forzado: un punto aparece en posición aleatoria y
    el usuario lo arrastra hasta donde percibió el fosfeno.

    Implementa la misma interfaz que DrawingTablet (update / reset / close /
    last_status / mode) para ser compatible con DrawingResponseCapture sin
    cambios en el pipeline de análisis.
    """

    DOT_RADIUS = 12
    DOT_COLOR = (255, 220, 0)   # amarillo
    HIT_FACTOR = 3              # radio de captura = DOT_RADIUS × HIT_FACTOR

    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        brush_size: int = 3,
        brush_color: tuple = (255, 255, 0),
        min_dist_px: float = 80.0,
        max_dist_px: float = 170.0,
        ppd: float | None = None,
        screen_cx: int | None = None,
        screen_cy: int | None = None,
    ):
        import random
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.brush_size = brush_size
        self.brush_color = brush_color
        self.min_dist_px = max(1.0, float(min_dist_px))
        self.max_dist_px = max(self.min_dist_px + 1, float(max_dist_px))
        self._ppd = float(ppd) if ppd else None
        self._screen_cx = screen_cx if screen_cx is not None else screen_width // 2
        self._screen_cy = screen_cy if screen_cy is not None else screen_height // 2
        self._anchor_px: tuple[int, int] | None = None
        self._rng = random.Random()
        self.mode = "forced_adjustment"
        self.last_status = "unknown"
        self.finished = False
        self._dragging = False
        self._drag_done = False      # True cuando se ha completado al menos un arrastre
        self._pos: tuple[int, int] = self._random_pos()
        self._trail: list[tuple[int, int]] = []
        self._font = None   # lazy: pygame may not be inited yet at import time
        print("[ForcedAdjustmentTablet] Inicializado")

    # ------------------------------------------------------------------ #
    # Interfaz pública                                                     #
    # ------------------------------------------------------------------ #

    def update(self, screen, events):
        """Procesa eventos y dibuja. Devuelve (finished, canvas | None)."""
        if self._font is None:
            self._font = pygame.font.Font(None, 48)

        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                hit_r = self.DOT_RADIUS * self.HIT_FACTOR
                if (mx - self._pos[0]) ** 2 + (my - self._pos[1]) ** 2 <= hit_r ** 2:
                    self._dragging = True
                    self._trail = [self._pos]

            elif event.type == pygame.MOUSEBUTTONUP:
                if self._dragging:
                    self._dragging = False
                    self._pos = event.pos
                    self._trail.append(self._pos)
                    self._drag_done = True
                    canvas = self._render_result()
                    self.finished = True
                    self.last_status = "ok"
                    return (True, canvas)

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                # ENTER sin arrastre = catch trial / no vio nada → canvas negro vacío
                if not self._drag_done:
                    canvas = pygame.Surface((self.screen_width, self.screen_height))
                    canvas.fill((0, 0, 0))
                    self.finished = True
                    self.last_status = "empty"
                    return (True, canvas)

        if self._dragging:
            self._pos = pygame.mouse.get_pos()
            self._trail.append(self._pos)

        self._draw(screen)
        return (False, None)

    def reset(self):
        self._pos = self._random_pos()
        self._dragging = False
        self._drag_done = False
        self._trail = []
        self.finished = False
        self.last_status = "unknown"
        print("[ForcedAdjustmentTablet] Reseteado para nuevo ensayo")

    def close(self):
        pass

    # ------------------------------------------------------------------ #
    # Helpers privados                                                     #
    # ------------------------------------------------------------------ #

    def set_anchor_px(self, x_px: int, y_px: int) -> None:
        """Actualiza la posición del fosfeno actual (en píxeles) para el próximo reset()."""
        self._anchor_px = (int(x_px), int(y_px))

    def set_anchor_deg(self, x_deg: float, y_deg: float) -> None:
        """Actualiza la posición del fosfeno actual (en grados) para el próximo reset().
        Convierte a píxeles usando ppd y el centro de pantalla."""
        if self._ppd is None:
            return
        px = int(round(self._screen_cx + x_deg * self._ppd))
        py = int(round(self._screen_cy - y_deg * self._ppd))
        self._anchor_px = (px, py)

    def _random_pos(self) -> tuple[int, int]:
        """Posición aleatoria en un anillo alrededor del fosfeno (si hay anchor),
        o en cualquier zona de la pantalla evitando el cuarto central (fallback)."""
        import math
        margin = max(self.DOT_RADIUS * 2, 40)
        if self._anchor_px is not None:
            ax, ay = self._anchor_px
            for _ in range(300):
                angle = self._rng.uniform(0, 2 * math.pi)
                r = self._rng.uniform(self.min_dist_px, self.max_dist_px)
                x = int(round(ax + r * math.cos(angle)))
                y = int(round(ay + r * math.sin(angle)))
                if margin <= x <= self.screen_width - margin and \
                   margin <= y <= self.screen_height - margin:
                    return (x, y)
        # Fallback: posición aleatoria evitando el cuarto central
        mx, my = self.screen_width // 2, self.screen_height // 2
        excl_x = self.screen_width // 4
        excl_y = self.screen_height // 4
        while True:
            x = self._rng.randint(margin, self.screen_width - margin)
            y = self._rng.randint(margin, self.screen_height - margin)
            if abs(x - mx) > excl_x or abs(y - my) > excl_y:
                return (x, y)

    def _draw(self, screen):
        screen.fill((0, 0, 0))
        # Rastro visual del arrastre (solo durante el drag, no se guarda en canvas)
        for pt in self._trail:
            pygame.draw.circle(screen, self.brush_color, pt, self.brush_size)
        # Punto principal
        pygame.draw.circle(screen, self.DOT_COLOR, self._pos, self.DOT_RADIUS)
        if not self._dragging:
            pygame.draw.circle(screen, (255, 255, 255), self._pos,
                               self.DOT_RADIUS + 2, 2)
        text = self._font.render(
            "Arrastra el punto hasta donde viste el fosfeno y presiona ENTER",
            True, (255, 255, 255),
        )
        screen.blit(text, text.get_rect(center=(self.screen_width // 2, 50)))

    def _render_result(self) -> pygame.Surface:
        """Canvas de análisis: fondo negro con solo el punto en la posición final.
        El rastro no se incluye para que el analizador extraiga un centroide limpio."""
        canvas = pygame.Surface((self.screen_width, self.screen_height))
        canvas.fill((0, 0, 0))
        pygame.draw.circle(canvas, self.brush_color, self._pos, self.brush_size)
        return canvas


class LineDrawingTablet:
    """Pantalla de respuesta para el método de mapeo *pareado* (paired).

    En cada ensayo se estimulan dos electrodos (A primero, 1 s de descanso,
    luego B). El participante traza una línea *dirigida* desde donde percibió
    A hasta donde percibió B. Lo que importa NO es la línea dibujada sino sus
    dos extremos ordenados (endpoint_A, endpoint_B); su diferencia es el
    vector de desplazamiento Δ(A→B) que alimenta a embed_displacement_lsq().

    Interacción (dos clics, robusta para ratón y stylus):
      1) primer clic  → fija el extremo A (donde se vio el primer fosfeno)
      2) segundo clic → fija el extremo B (donde se vio el segundo fosfeno)
      ENTER           → confirma (status="ok" si ambos extremos están puestos)

    Casos parciales (esenciales para no inventar datos):
      - "solo vi UNO": pulsar 1 (solo A) o 2 (solo B) antes de ENTER →
        status="partial"; el extremo no visto queda en None y el par se
        descarta como restricción de desplazamiento aguas abajo.
      - "no vi NINGUNO": ENTER sin clics → status="empty".

    Implementa la misma interfaz que DrawingTablet/ForcedAdjustmentTablet
    (update / reset / close / last_status / mode) para encajar en
    DrawingResponseCapture sin tocar el pipeline. Además expone los extremos
    ordenados como atributos (endpoint_a / endpoint_b) y los serializa al
    metadata vía save_result()."""

    DOT_RADIUS = 9
    COLOR_A = (80, 200, 255)    # azul → primer fosfeno (A)
    COLOR_B = (255, 200, 80)    # naranja → segundo fosfeno (B)
    LINE_COLOR = (200, 200, 200)
    R_DELETE_LAST = pygame.K_x

    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        brush_size: int = 4,
        brush_color: tuple = (255, 255, 0),
        instructions_text: str | None = None,
        hide_cursor: bool = False,
        cursor_clip_rect=None,
    ):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.brush_size = brush_size
        self.brush_color = brush_color
        self.mode = "paired_line"
        self.hide_cursor = bool(hide_cursor)
        self._cursor_clip_rect = cursor_clip_rect
        self._cursor_clip_active = False

        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)

        # Estado de respuesta
        self.endpoint_a: tuple[int, int] | None = None
        self.endpoint_b: tuple[int, int] | None = None
        self.finished = False
        self.last_status = "unknown"

        self.title_text = instructions_text or (
            "Marca DÓNDE viste el 1er punto (azul), luego el 2º (naranja). ENTER"
        )
        self._font = None  # lazy: pygame puede no estar inicializado al importar
        self._cursor_prev_visible = None
        if self.hide_cursor:
            self._cursor_prev_visible = pygame.mouse.get_visible()
            pygame.mouse.set_visible(False)
        print("[LineDrawingTablet] Inicializado")

    # ------------------------------------------------------------------ #
    # Interfaz pública                                                     #
    # ------------------------------------------------------------------ #

    def update(self, screen, events):
        """Procesa eventos y dibuja. Devuelve (finished, canvas | None)."""
        if self._font is None:
            self._font = pygame.font.Font(None, 48)

        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                self._place_next(event.pos)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    return self._confirm()
                elif event.key == self.R_DELETE_LAST:
                    self._undo_last()
                elif event.key == pygame.K_1:
                    # "solo vi el primero": fuerza modo parcial-solo-A.
                    self.endpoint_b = None
                    print("[LineDrawingTablet] Marcado: solo se vio A")
                elif event.key == pygame.K_2:
                    # "solo vi el segundo": fuerza modo parcial-solo-B.
                    self.endpoint_a = None
                    print("[LineDrawingTablet] Marcado: solo se vio B")

        self._draw(screen)
        return (False, None)

    def reset(self):
        self.endpoint_a = None
        self.endpoint_b = None
        self.finished = False
        self.last_status = "unknown"
        self._apply_cursor_clip()
        print("[LineDrawingTablet] Reseteado para nuevo par")

    def close(self):
        self._release_cursor_clip()
        if self._cursor_prev_visible is not None:
            pygame.mouse.set_visible(self._cursor_prev_visible)
            self._cursor_prev_visible = None

    def save_result(self, output_dir, *, drawing_filename, saccade_filename=None):
        """Guarda un PNG limpio (dos puntos) para el analizador y devuelve un
        ResponseResult cuyo `extras` lleva los extremos ordenados y el vector
        de desplazamiento en píxeles — la carga útil real del método pareado.

        Se importa ResponseResult de forma perezosa para no crear un ciclo de
        importación con response_capture (que importa de schemas, no de aquí)."""
        from pathlib import Path
        from scripts.response_capture import ResponseResult

        output_dir = Path(output_dir)
        canvas = self._render_result()
        pygame.image.save(canvas, str(output_dir / drawing_filename))

        ax = list(self.endpoint_a) if self.endpoint_a is not None else None
        bx = list(self.endpoint_b) if self.endpoint_b is not None else None
        disp = None
        if ax is not None and bx is not None:
            disp = [bx[0] - ax[0], bx[1] - ax[1]]

        return ResponseResult(
            mode="paired_line",
            status=self.last_status or "ok",
            response_file=drawing_filename,
            response_file_type="png",
            debug={
                "endpoint_a_px": ax,
                "endpoint_b_px": bx,
                "displacement_px": disp,
            },
        )

    # ------------------------------------------------------------------ #
    # Helpers privados                                                     #
    # ------------------------------------------------------------------ #

    def _place_next(self, pos):
        """Coloca el siguiente extremo libre: primero A, luego B. Un tercer
        clic re-coloca B (corrección rápida del último extremo)."""
        p = (int(pos[0]), int(pos[1]))
        if self.endpoint_a is None:
            self.endpoint_a = p
        else:
            self.endpoint_b = p

    def _undo_last(self):
        if self.endpoint_b is not None:
            self.endpoint_b = None
        elif self.endpoint_a is not None:
            self.endpoint_a = None

    def _confirm(self):
        has_a = self.endpoint_a is not None
        has_b = self.endpoint_b is not None
        if has_a and has_b:
            self.last_status = "ok"
        elif has_a or has_b:
            self.last_status = "partial"
        else:
            self.last_status = "empty"
        self.finished = True
        self._release_cursor_clip()
        print(
            f"[LineDrawingTablet] Confirmado (status={self.last_status}, "
            f"A={self.endpoint_a}, B={self.endpoint_b})"
        )
        return (True, self._render_result())

    def _draw(self, screen):
        screen.fill(self.BLACK)
        # Línea dirigida A→B (solo si ambos extremos existen)
        if self.endpoint_a is not None and self.endpoint_b is not None:
            pygame.draw.line(screen, self.LINE_COLOR, self.endpoint_a,
                             self.endpoint_b, 2)
        if self.endpoint_a is not None:
            pygame.draw.circle(screen, self.COLOR_A, self.endpoint_a,
                               self.DOT_RADIUS)
        if self.endpoint_b is not None:
            pygame.draw.circle(screen, self.COLOR_B, self.endpoint_b,
                               self.DOT_RADIUS)

        title = self._font.render(self.title_text, True, self.WHITE)
        screen.blit(title, title.get_rect(center=(self.screen_width // 2, 50)))
        hint = self._font.render(
            "X=deshacer  1=solo vi el 1º  2=solo vi el 2º", True, self.WHITE)
        screen.blit(hint, hint.get_rect(center=(self.screen_width // 2, 100)))

    def _render_result(self) -> pygame.Surface:
        """Canvas de análisis: fondo negro con los dos extremos como puntos del
        color de pincel (sin la línea ni los colores A/B) para que un extractor
        de centroides genérico funcione. Los datos ordenados viven en el
        metadata, no en los píxeles."""
        canvas = pygame.Surface((self.screen_width, self.screen_height))
        canvas.fill(self.BLACK)
        for pt in (self.endpoint_a, self.endpoint_b):
            if pt is not None:
                pygame.draw.circle(canvas, self.brush_color, pt, self.brush_size)
        return canvas

    def _apply_cursor_clip(self):
        if self._cursor_clip_rect is None or self._cursor_clip_active:
            return
        try:
            from scripts.cursor_clip import clip_cursor

            if clip_cursor(self._cursor_clip_rect):
                self._cursor_clip_active = True
        except Exception as e:
            print(f"[LineDrawingTablet] ⚠ no se pudo aplicar cursor clip: {e}")

    def _release_cursor_clip(self):
        if not self._cursor_clip_active:
            return
        try:
            from scripts.cursor_clip import clip_cursor

            clip_cursor(None)
        except Exception as e:
            print(f"[LineDrawingTablet] ⚠ no se pudo liberar cursor clip: {e}")
        finally:
            self._cursor_clip_active = False
