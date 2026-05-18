"""
Tablet de dibujo - Versión simplificada
Pantalla negra donde el usuario dibuja y confirma con ENTER
"""

import pygame


class DrawingTablet:
    """
    Interfaz de dibujo simple para capturar la respuesta del usuario
    """

    def __init__(self, screen_width, screen_height, brush_size=5, brush_color=(255, 255, 0)):
        """
        Inicializa la tablet de dibujo

        Args:
            screen_width: Ancho de la pantalla principal
            screen_height: Alto de la pantalla principal
            brush_size: Tamaño del pincel (radio en píxeles). Default: 5
            brush_color: Color del pincel en RGB. Default: (255, 255, 0) amarillo
        """
        print("[DrawingTablet] Inicializando...")

        self.screen_width = screen_width
        self.screen_height = screen_height

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

        # Estado del dibujo
        self.drawing = False
        self.strokes = []  # Lista de trazos
        self.current_stroke = []  # Trazo actual
        self.finished = False

        # Fuente para título
        self.font = pygame.font.Font(None, 48)

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
        title = "Dibuje ahora y presione ENTER"
        title_surface = self.font.render(title, True, self.WHITE)
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
        print("[DrawingTablet] Reseteado para nuevo dibujo")
