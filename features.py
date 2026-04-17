import cv2
import numpy as np
from scipy.fftpack import fft2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def extract_features(image_path):
    img = cv2.imread(image_path)

    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30))

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    face_img = gray[y : y + h, x : x + w]

    face_img = cv2.resize(face_img, (128, 128))

    features = []

    # Texture
    features.append(np.var(face_img))

    # Blur detection
    features.append(cv2.Laplacian(face_img, cv2.CV_64F).var())

    # FFT
    fft = np.abs(fft2(face_img))
    features.append(np.mean(fft))

    # Symmetry
    left = face_img[:, :64]
    right = face_img[:, 64:]
    features.append(np.mean(np.abs(left - np.fliplr(right))))

    return np.array(features)
