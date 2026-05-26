"""Anchor screen with traffic-light state machine.

Color scheme is consistent across all phases:
  - white outline  : idle / fixation not yet acquired
  - green outline  : fixation acquired (ready for stim)
  - red filled     : stimulation active (handled by StimulationScreen)
  - dim white      : response phase (handled by SaccadeScreen / DrawingTablet)

The YAML field `anchor_circle.color` is ignored on purpose — colors carry
state semantics now, so they must not be configurable.
"""

import pygame


_COLOR_IDLE = (200, 200, 200)
_COLOR_ACQUIRED = (0, 220, 0)
_TEXT_IDLE = (200, 200, 200)
_TEXT_ACQUIRED = (0, 220, 0)


class AnchorScreen:
    """Black background + anchor circle. Color encodes fixation state."""

    def __init__(self, params, eye_tracker, fixation_tick=None):
        """`fixation_tick` is an optional pygame.mixer.Sound that fires
        once per low→high transition of `is_looking`. Use to give the
        participant a soft audio confirmation that the system saw their
        fixation (mirrors the green anchor color)."""
        self.params = params
        self.eye_tracker = eye_tracker

        screen_config = params["screen"]
        self.width = screen_config["width"]
        self.height = screen_config["height"]
        self.background_color = tuple(screen_config["background_color"])

        anchor_config = screen_config["anchor_circle"]
        self.circle_radius = anchor_config["radius"]
        self.circle_thickness = anchor_config["thickness"]
        self.tolerance_radius = anchor_config["tolerance_radius"]
        self.circle_center = (self.width // 2, self.height // 2)

        self.is_looking = False
        self._prev_is_looking = False
        self.fixation_tick = fixation_tick
        self.font = pygame.font.Font(None, 44)
        print("[AnchorScreen] Inicializado")

    def reset_fixation_edge(self):
        """Forget the previous fixation state so the next acquisition fires
        the tick. Call at the start of each prestim phase."""
        self._prev_is_looking = False

    def update(self, screen, eye_tracker_frame):
        if self.eye_tracker and eye_tracker_frame is not None:
            self.is_looking = self.eye_tracker.is_looking_at_point(
                frame=eye_tracker_frame,
                target_point=self.circle_center,
                screen_size=(self.width, self.height),
                tolerance_radius=self.tolerance_radius,
            )
        else:
            self.is_looking = True

        if self.is_looking and not self._prev_is_looking and self.fixation_tick is not None:
            try:
                self.fixation_tick.play()
            except Exception as e:
                print(f"[AnchorScreen] ⚠ fixation_tick.play falló: {e}")
        self._prev_is_looking = self.is_looking

        self.draw(screen)
        return self.is_looking

    def draw(self, screen):
        screen.fill(self.background_color)
        circle_color = _COLOR_ACQUIRED if self.is_looking else _COLOR_IDLE
        pygame.draw.circle(
            screen,
            circle_color,
            self.circle_center,
            self.circle_radius,
            self.circle_thickness,
        )
        status_text = "Fijación detectada" if self.is_looking else "Mira al círculo"
        text_color = _TEXT_ACQUIRED if self.is_looking else _TEXT_IDLE
        text_surface = self.font.render(status_text, True, text_color)
        text_rect = text_surface.get_rect(center=(self.width // 2, 80))
        screen.blit(text_surface, text_rect)
    