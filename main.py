from lib import config
import time
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
import os
import copy
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from threading import Thread
import threading
import uvicorn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),
          activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.load_weights(config.MODEL_PATH)

# Declraing a lock
lock = threading.Lock()

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:9000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

success = False

stop = False

frame = None

cap = cv2.VideoCapture(config.URL_RSTP)

overlay = cv2.imread('overlay.png')

tlx, tly, brx, bry = 185, 15, 1735, 885


facecasc = cv2.CascadeClassifier(config.CASCADE_PATH)

# dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful",
                3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


def get_frame():
    global success, frame, cap
    while (not stop):
        time.sleep(0.01)
        lock.acquire()
        success, frame = cap.read()
        lock.release()

        if not success:
            print('Restarting video...')
            cap = cv2.VideoCapture(
                config.URL_RSTP)


def detect_emoction(frame):
    global facecasc, emotion_dict, model
    image = copy.copy(frame)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = facecasc.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(
            cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        cv2.putText(image, emotion_dict[maxindex], (x+20, y-60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return image


i = 0


def generate():
    global success, frame, i

    try:
        while (True):
            i += 1
            if success:
                # wait until the lock is acquired
                with lock:
                    test = copy.deepcopy(frame)
                    test = cv2.resize(test, (brx-tlx, bry-tly))
                    print('Running')
                    if i % 2 == 0:
                        img_out = detect_emoction(test)
                        overlay[tly:bry, tlx:brx, :] = img_out
                    else:
                        overlay[tly:bry, tlx:brx, :] = test
                    img_out = overlay

                    if img_out is None:
                        print(img_out)
                        continue

                    # encode the frame in JPEG format
                    (flag, encodedImage) = cv2.imencode(".jpg", img_out)

                    if not flag:
                        continue

                # yield the output frame in the byte format
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                       bytearray(encodedImage) + b'\r\n')

    finally:
        print("caught cancelled error")


@app.get("/")
def video_feed() -> StreamingResponse:
    response = StreamingResponse(
        generate(), media_type="multipart/x-mixed-replace;boundary=frame")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response


if __name__ == "__main__":
    thread = Thread(target=get_frame)
    thread.start()
    uvicorn.run(app, host="0.0.0.0", port=8083)
    stop = True
    thread.join()
    if not thread.is_alive():
        print('Thread killed.')
