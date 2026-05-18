"""
Eye Tracker - Detección de mirada usando MediaPipe
"""

import cv2
import mediapipe as mp
import numpy as np


class EyeTracker:
    """
    Detecta la mirada del usuario usando la webcam y MediaPipe Face Mesh
    """

    def __init__(self, camera_index=0):
        """
        Inicializa el eye tracker

        Args:
            camera_index: Índice de la cámara (0 = webcam por defecto)
        """
        print("[EyeTracker] Inicializando...")

        # ============================================
        # 1. ABRIR WEBCAM
        # ============================================
        # Intentar con diferentes backends y índices
        self.cap = None

        # Lista de backends a probar en Windows
        backends = [
            cv2.CAP_DSHOW,  # DirectShow (Windows)
            cv2.CAP_MSMF,  # Microsoft Media Foundation
            cv2.CAP_ANY,  # Autodetección
        ]

        # Intentar abrir la cámara con diferentes backends
        for backend in backends:
            print(
                f"[EyeTracker] Intentando abrir cámara {camera_index} con backend {backend}..."
            )
            self.cap = cv2.VideoCapture(camera_index, backend)

            if self.cap.isOpened():
                # Verificar que realmente funciona capturando un frame
                ret, test_frame = self.cap.read()
                if ret and test_frame is not None:
                    print(
                        f"[EyeTracker] ✓ Cámara {camera_index} abierta correctamente con backend {backend}"
                    )
                    break
                else:
                    self.cap.release()
                    self.cap = None
            else:
                if self.cap:
                    self.cap.release()
                self.cap = None

        # Si no funcionó, intentar con otros índices de cámara (0, 1, 2)
        if self.cap is None or not self.cap.isOpened():
            print("[EyeTracker] Intentando otros índices de cámara...")
            for cam_idx in [1, 2]:
                for backend in backends:
                    print(
                        f"[EyeTracker] Probando cámara {cam_idx} con backend {backend}..."
                    )
                    self.cap = cv2.VideoCapture(cam_idx, backend)

                    if self.cap.isOpened():
                        ret, test_frame = self.cap.read()
                        if ret and test_frame is not None:
                            print(
                                f"[EyeTracker] ✓ Cámara {cam_idx} funciona (backend {backend})"
                            )
                            camera_index = cam_idx
                            break
                        else:
                            self.cap.release()
                            self.cap = None
                    else:
                        if self.cap:
                            self.cap.release()
                        self.cap = None

                if self.cap and self.cap.isOpened():
                    break

        # Si aún no funciona, lanzar excepción con información útil
        if self.cap is None or not self.cap.isOpened():
            error_msg = (
                "No se pudo abrir ninguna cámara.\n"
                "Posibles soluciones:\n"
                "1. Verifica que la cámara esté conectada\n"
                "2. Verifica permisos de cámara en Windows (Configuración > Privacidad > Cámara)\n"
                "3. Cierra otras aplicaciones que usen la cámara (Zoom, Teams, etc.)\n"
                "4. Prueba desconectar/reconectar la cámara USB"
            )
            raise Exception(error_msg)

        # Configurar resolución (más pequeña = más rápido)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # ============================================
        # 2. INICIALIZAR MEDIAPIPE FACE MESH
        # ============================================
        # MediaPipe Face Mesh detecta 478 puntos en la cara
        # Incluye puntos específicos para los iris (ojos)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,  # Solo detectar 1 cara
            refine_landmarks=True,  # Incluir landmarks de iris (⭐ IMPORTANTE)
            min_detection_confidence=0.2,  # Confianza mínima para detectar
            min_tracking_confidence=0.2,  # Confianza mínima para seguir
        )

        # ============================================
        # 3. ÍNDICES DE LOS IRIS
        # ============================================
        # MediaPipe Face Mesh tiene puntos específicos para los iris:
        # - Iris izquierdo: landmarks 474-477 (4 puntos)
        # - Iris derecho: landmarks 469-472 (4 puntos)
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]

        # ============================================
        # 4. BUFFER PARA SUAVIZADO TEMPORAL
        # ============================================
        # Guardar las últimas N detecciones para promediar
        # Esto reduce el "jitter" (temblor) en la detección
        self.gaze_buffer = []
        self.buffer_size = 10  # Promediar últimas 10 detecciones

        # Last computed gaze points (exposed for external use)
        self.last_raw_gaze = None  # Raw unsmoothed gaze point (x, y)
        self.last_smooth_gaze = None  # Smoothed gaze point (x, y)

        print("[EyeTracker] ✓ Inicializado correctamente")

    def get_frame(self):
        """
        Captura un frame de la webcam

        Returns:
            numpy.ndarray: Frame capturado (formato BGR de OpenCV)
        """
        ret, frame = self.cap.read()

        if not ret:
            print("[EyeTracker] ⚠ No se pudo capturar frame")
            return None

        return frame

    def is_looking_at_point(
        self, frame, target_point, screen_size, tolerance_radius=100
    ):
        """
        Detecta si el usuario está mirando a un punto específico

        Args:
            frame: Frame de la webcam (numpy array BGR)
            target_point: (x, y) punto objetivo en coordenadas de pantalla
            screen_size: (width, height) tamaño de la pantalla
            tolerance_radius: Radio de tolerancia en píxeles

        Returns:
            bool: True si está mirando al punto, False si no
        """
        if frame is None:
            return False

        # ============================================
        # PASO 1: CONVERTIR BGR → RGB
        # ============================================
        # OpenCV usa BGR, MediaPipe usa RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ============================================
        # PASO 2: DETECTAR FACE MESH
        # ============================================
        results = self.face_mesh.process(rgb_frame)

        # Si no detecta ninguna cara, retornar False
        if not results.multi_face_landmarks:
            return False

        # ============================================
        # PASO 3: OBTENER LANDMARKS DE LA CARA
        # ============================================
        # Tomar la primera cara detectada
        face_landmarks = results.multi_face_landmarks[0]

        # ============================================
        # PASO 4: CALCULAR PUNTO DE MIRADA
        # ============================================
        gaze_point = self._get_gaze_point(face_landmarks, frame.shape, screen_size)

        if gaze_point is None:
            return False

        # Store raw gaze point
        self.last_raw_gaze = gaze_point

        # ============================================
        # PASO 5: SUAVIZADO TEMPORAL
        # ============================================
        # Añadir detección actual al buffer
        self.gaze_buffer.append(gaze_point)

        # Mantener solo las últimas N detecciones
        if len(self.gaze_buffer) > self.buffer_size:
            self.gaze_buffer.pop(0)

        # Calcular promedio de las últimas detecciones
        avg_gaze = np.mean(self.gaze_buffer, axis=0)

        # Store smoothed gaze point
        self.last_smooth_gaze = (float(avg_gaze[0]), float(avg_gaze[1]))

        # ============================================
        # PASO 6: CALCULAR DISTANCIA AL OBJETIVO
        # ============================================
        distance = np.sqrt(
            (avg_gaze[0] - target_point[0]) ** 2 + (avg_gaze[1] - target_point[1]) ** 2
        )

        is_within = distance <= tolerance_radius
        if not is_within:
            print(
                f"[EyeTracker] Gaze miss: raw=({gaze_point[0]:.0f},{gaze_point[1]:.0f}) "
                f"smooth=({avg_gaze[0]:.0f},{avg_gaze[1]:.0f}) "
                f"target=({target_point[0]},{target_point[1]}) "
                f"dist={distance:.1f}px tol={tolerance_radius}px"
            )

        return is_within

    def _get_gaze_point(self, face_landmarks, frame_shape, screen_size):
        """
        Calcula el punto de mirada en coordenadas de pantalla

        Args:
            face_landmarks: Landmarks de MediaPipe
            frame_shape: (height, width, channels) del frame de la webcam
            screen_size: (width, height) de la pantalla

        Returns:
            tuple: (x, y) punto de mirada en coordenadas de pantalla, o None
        """
        frame_h, frame_w = frame_shape[:2]
        screen_w, screen_h = screen_size

        # ============================================
        # PASO 1: OBTENER COORDENADAS DE LOS IRIS
        # ============================================
        left_iris_points = []
        right_iris_points = []

        # Iris izquierdo (landmarks 474-477)
        for idx in self.LEFT_IRIS:
            landmark = face_landmarks.landmark[idx]
            # Convertir coordenadas normalizadas [0,1] a píxeles
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            left_iris_points.append([x, y])

        # Iris derecho (landmarks 469-472)
        for idx in self.RIGHT_IRIS:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            right_iris_points.append([x, y])

        # ============================================
        # PASO 2: CALCULAR CENTRO DE CADA IRIS
        # ============================================
        # Promedio de los 4 puntos de cada iris
        left_center = np.mean(left_iris_points, axis=0)
        right_center = np.mean(right_iris_points, axis=0)

        # ============================================
        # PASO 3: PROMEDIO DE AMBOS OJOS
        # ============================================
        # Usar el promedio de ambos ojos para mayor precisión
        iris_center = (left_center + right_center) / 2

        # ============================================
        # PASO 4: NORMALIZAR A RANGO [0, 1]
        # ============================================
        # Invertir X porque la webcam es como un espejo
        normalized_x = 1.0 - (iris_center[0] / frame_w)
        normalized_y = iris_center[1] / frame_h

        # ============================================
        # PASO 5: MAPEAR A COORDENADAS DE PANTALLA
        # ============================================
        gaze_x = normalized_x * screen_w
        gaze_y = normalized_y * screen_h

        return (gaze_x, gaze_y)

    def release(self):
        """Libera la webcam"""
        print("[EyeTracker] Liberando recursos...")

        if self.cap:
            self.cap.release()

        print("[EyeTracker] ✓ Recursos liberados")
