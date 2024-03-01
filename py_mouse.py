import cv2
import mediapipe as mp
import pyautogui
import numpy as np

class MouseCamara:
    
    nombre_ventana = 'Mouse control - python'
    
    def __init__(self, mp_dibujo, mp_manos, ancho_pantalla, altura_pantalla, cap):
        self.mp_dibujo = mp_dibujo
        self.mp_manos = mp_manos
        self.ancho_pantalla = ancho_pantalla
        self.altura_pantalla = altura_pantalla
        self.cap = cap
        self.ultimo_x = 0
        self.ultimo_y = 0
        self.factor_suavizado = 0.3333
    
    
    def configuracion(self):
        # Colocar la pantalla de la camara normal
        pyautogui.FAILSAFE = False
        cv2.namedWindow(MouseCamara.nombre_ventana, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(MouseCamara.nombre_ventana, self.ancho_pantalla, self.altura_pantalla)
        

    def evento_mouse(self):
        self.configuracion()
        
        with self.mp_manos.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            
            is_abierto = cap.isOpened()
            while is_abierto:        
                exito, imagen = cap.read()
                
                if not exito:
                    print("Ignoring empty camera frame.")
                    continue
                    
                imagen = cv2.flip(imagen, 1)
                imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
                imagen.flags.writeable = False
                    
                resultados = hands.process(imagen)

                imagen.flags.writeable = True
                imagen = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)
                
                referencia_varias_manos = resultados.multi_hand_landmarks
                if referencia_varias_manos:
                    for item_manos in referencia_varias_manos:
                        
                        # Encuentra el dedo índice
                        tip_indice = item_manos.landmark[self.mp_manos.HandLandmark.INDEX_FINGER_TIP]

                        # Encuentra el pulgar
                        tip_pulgar = item_manos.landmark[self.mp_manos.HandLandmark.THUMB_TIP] 
            
                        h, w, _ = imagen.shape
                        x, y = int(tip_indice.x * w), int(tip_indice.y * h)
                        
                        indice_x, indice_y = int(tip_indice.x * w), int(tip_indice.y * h)
                        pulgar_x, pulgar_y = int(tip_pulgar.x * w), int(tip_pulgar.y * h)
                        
                        # Calcula la distancia euclidiana entre el índice y el pulgar
                        distancia = np.sqrt((pulgar_x - indice_x) ** 2 + (pulgar_y - indice_y) ** 2)

                        # Mapea las coordenadas del video a las de la pantalla
                        pantalla_x = np.interp(x, (0, w), (0, self.ancho_pantalla))
                        pantalla_y = np.interp(y, (0, h), (0, self.altura_pantalla))

                        # Aplica suavizado
                        suave_x = self.ultimo_x + (pantalla_x - self.ultimo_x) * self.factor_suavizado
                        suave_y = self.ultimo_y + (pantalla_y - self.ultimo_y) * self.factor_suavizado

                        self.ultimo_x, self.ultimo_y = suave_x, suave_y
                        pyautogui.moveTo(suave_x, suave_y)

                        if distancia < 50:
                            pyautogui.click()

                        self.mp_dibujo.draw_landmarks(
                            imagen, item_manos, self.mp_manos.HAND_CONNECTIONS)
            
                cv2.imshow(MouseCamara.nombre_ventana, imagen)
            
                if cv2.waitKey(5) & 0xFF == 27:
                    print("Terminado.")
                    is_abierto = False
                
        cap.release()
        cv2.destroyAllWindows()
    

if __name__ == "__main__":
        
    mp_dibujo = mp.solutions.drawing_utils
    mp_manos = mp.solutions.hands

    ancho_pantalla, altura_pantalla = pyautogui.size()

    cap = cv2.VideoCapture(0)

    mouseCamara = MouseCamara(mp_dibujo, mp_manos, ancho_pantalla, altura_pantalla, cap)
    mouseCamara.evento_mouse()
