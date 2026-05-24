import cv2
import numpy as np

IMG_SIZE = 128

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def preprocess_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30)
    )

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
    face = img[y : y + h, x : x + w]
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face = face / 255.0

    return face.astype(np.float32)
