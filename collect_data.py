import os
import cv2 as cv

DATA_DIR = './data'
NUMBER_OF_CLASSES = 26
DATASET_SIZE = 150
ALPHABET_DICT = {i: chr(65 + i) for i in range(NUMBER_OF_CLASSES) if chr(65 + i) not in ['J', 'Z']}
HANDS = ['Right', 'Left']

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

cap = cv.VideoCapture(0)
for j in ALPHABET_DICT.keys():
    if os.path.exists(os.path.join(DATA_DIR, str(j))):
        continue

    os.makedirs(os.path.join(DATA_DIR, str(j)))

    print(f'Collecting data for class {j} - {ALPHABET_DICT[j]}')
    
    counter = 0
    for i, hand in enumerate(HANDS):
        while True:
            ret, frame = cap.read()
            cv.putText(frame,
                       f'Press "m" to capture {hand} hand',
                       (50, 50),
                       cv.FONT_HERSHEY_SIMPLEX,
                       1,
                       (0, 255, 0),
                       2,
                       cv.LINE_AA)
            cv.imshow('frame', frame)
            if cv.waitKey(25) & 0xFF == ord('m'):
                break

        while counter < DATASET_SIZE * (i + 1):
            ret, frame = cap.read() 
            cv.imshow('frame', frame)
            cv.waitKey(25)
            cv.imwrite(os.path.join(DATA_DIR, str(j), f'{counter}.jpg'), frame)
            counter += 1

cap.release()
cv.destroyAllWindows()