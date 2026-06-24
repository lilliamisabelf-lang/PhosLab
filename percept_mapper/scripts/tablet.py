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
    ):
        import random
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.brush_size = brush_size
        self.brush_color = brush_color
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

    def _random_pos(self) -> tuple[int, int]:
        """Posición aleatoria evitando el cuarto central de la pantalla."""
        mx, my = self.screen_width // 2, self.screen_height // 2
        excl_x = self.screen_width // 4
        excl_y = self.screen_height // 4
        margin = max(self.DOT_RADIUS * 2, 40)
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
