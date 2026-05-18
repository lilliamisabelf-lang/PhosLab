"""
Visualizador de Webcam con MediaPipe Face Mesh
Muestra la imagen de la webcam en tiempo real con landmarks faciales
"""

import cv2
import mediapipe as mp
import numpy as np


class WebcamViewer:
    """
    Ventana independiente para visualizar webcam con Face Mesh
    Usa los frames del EyeTracker existente en lugar de abrir una nueva cámara
    """

    def __init__(self, eye_tracker, window_name="Webcam - Face Mesh"):
        """
        Inicializa el visualizador de webcam

        Args:
            eye_tracker: Objeto EyeTracker del cual obtener frames
            window_name: Nombre de la ventana
        """
        print("[WebcamViewer] Inicializando...")

        self.window_name = window_name
        self.eye_tracker = eye_tracker

        if self.eye_tracker is None:
            raise Exception("EyeTracker no disponible")

        # ============================================
        # 2. INICIALIZAR MEDIAPIPE FACE MESH
        # ============================================
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # Incluir landmarks de iris
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Crear ventana de OpenCV
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 640, 480)

        # Posicionar la ventana en la esquina superior derecha
        # (para que no tape la pantalla de Pygame en fullscreen)
        cv2.moveWindow(self.window_name, 100, 100)

        # Hacer que la ventana esté siempre al frente (Windows)
        try:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)
        except:
            pass  # Si falla, no es crítico

        print("[WebcamViewer] ✓ Inicializado correctamente")

    def update(self):
        """
        Captura un frame del EyeTracker y lo muestra con face mesh

        Returns:
            bool: True si debe continuar, False si se cierra la ventana
        """
        # Obtener frame del EyeTracker
        frame = self.eye_tracker.get_frame()

        if frame is None:
            print("[WebcamViewer] ⚠ No se pudo capturar frame")
            return False

        # Voltear horizontalmente (efecto espejo)
        frame = cv2.flip(frame, 1)

        # Convertir BGR → RGB para MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detectar face mesh
        results = self.face_mesh.process(rgb_frame)

        # Dibujar face mesh si se detecta
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Dibujar malla facial completa
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                )

                # Dibujar contornos faciales
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style(),
                )

                # Dibujar iris (si refine_landmarks=True)
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
                )

        # Mostrar frame (sin waitKey para evitar interferencias con Pygame)
        cv2.imshow(self.window_name, frame)

        # Verificar si la ventana se cerró (sin bloquear con waitKey)
        try:
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                return False
        except cv2.error:
            return False

        return True

    def release(self):
        """
        Libera recursos (solo cierra la ventana, no la cámara)
        """
        print("[WebcamViewer] Liberando recursos...")
        cv2.destroyWindow(self.window_name)
        print("[WebcamViewer] ✓ Recursos liberados")
