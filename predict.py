import numpy as np
from tensorflow.keras.models import load_model

from features import preprocess_image

model = load_model("cnn_model.h5")


def predict_image(image_path):
    img = preprocess_image(image_path)

    if img is None:
        return "No face detected"

    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img, verbose=0)[0]
    confidence = np.max(prediction)
    predicted_class = np.argmax(prediction)

    if predicted_class == 1:
        return f"⚠️ Deepfake Detected | Confidence: {confidence:.2f}"

    return f"✅ Real Image | Confidence: {confidence:.2f}"


# test
if __name__ == "__main__":
    print(predict_image("test.jpg"))
