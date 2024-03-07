import cv2
import numpy as np
import mediapipe as mp

mp_drawing=mp.solutions.drawing_utils
mp_holistic=mp.solutions.holistic
mp2_holistic=mp.solutions.holistic

#Video captura
cap= cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

#DETECCION DE IMAGEN


with mp_holistic.Holistic(
	static_image_mode=True,
	model_complexity=1)as holistic:

	# lectura de prenda
		image=cv2.imread("maniqui.jpg")
		image_rgb=cv2.cvtColor(image,cv2.COLOR_BGR5652RGB)

		results2 = holistic.process(image_rgb)
		mp.drawing.draw_landmarks(
			image,results2.pose_landmarks,mp_holistic.POSE_CONNECTIONS,
			mp_drawing.DrawingSpec(Color=(129,0,255),thickness=2, circle_radius=1),
			mp_drawing.DrawingSpec(Color=(129, 0, 255), thickness=2)
		)



with mp_holistic.Holistic(
	static_image_mode=False,
	model_complexity=1)as holistic:
	while True:
		# lectura de la video captura
		ret, frame = cap.read()
		# filtro gaussiano
		filtro = cv2.GaussianBlur(frame, (31, 31), 0)
		if ret ==False:
			break

		frame_rgb= cv2.cvtColor(frame,cv2.COLOR_BGR5652RGB)
		results=holistic.process(frame_rgb)
		mp.drawing.draw_landmarks(
			frame,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS,
			mp_drawing.DrawingSpec(Color=(129,0,255),thickness=2, circle_radius=1),
			mp_drawing.DrawingSpec(Color=(129, 0, 255), thickness=2)
		)
		cv2.imshow("Frame",frame)

		if cv2.waitKey(1)& 0xFF ==27:
			break
cap.release()
cv2.destroyAllWindows()ç


















