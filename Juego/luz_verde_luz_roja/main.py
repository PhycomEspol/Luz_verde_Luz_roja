import mediapipe as mp
import pyautogui
import cv2
import webbrowser
import imutils
import math

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

image = cv2.imread('gafa2.png', cv2.IMREAD_UNCHANGED)


# Iniciar la detección de manos
with mp_hands.Hands(
    max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    # Iniciar la captura de video
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret == False:
        # Liberar recursos
        cap.release()
        cv2.destroyAllWindows()

    # Inicializar la variable booleana
    hands_closed = False

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("No se pudo obtener la imagen")
            break

        # Convertir la imagen a RGB y detectar las manos
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Dibujar los landmarks de las manos en la imagen
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Obtener la posición de las manos si hay al menos 2 manos detectadas
            if len(results.multi_hand_landmarks) >= 2:
                hand_image = cv2.imread('gafa2.png')


                left_hand_x = results.multi_hand_landmarks[0].landmark[0].x
                left_hand_y = results.multi_hand_landmarks[0].landmark[0].y

                right_hand_x = results.multi_hand_landmarks[1].landmark[0].x
                right_hand_y = results.multi_hand_landmarks[1].landmark[0].y


        # Mostrar la imagen en una ventana
        cv2.imshow("Hands Detection", image)

        # Salir del loop si se presiona la tecla 'q'
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

