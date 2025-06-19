import cv2 as cv
import mediapipe as mp
import pickle
import numpy as np

with open('./model/model.pickle', 'rb') as f:
    model_dict = pickle.load(f)
    model = model_dict['model']

cap = cv.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

counter = 0
ALPHABET_DICT = {i: chr(65 + i) for i in range(26) if chr(65 + i) not in ['J', 'Z']}
while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        try:
            if counter % 20 == 0 :
                prediction = model.predict([np.asarray(data_aux)])

            predicted_character = ALPHABET_DICT[int(prediction[0])]

            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv.putText(frame, predicted_character, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv.LINE_AA)

        except Exception as e:
            print("Erro ao prever:", e)

    cv.imshow('frame', frame)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break

    counter += 1

cap.release()
cv.destroyAllWindows()