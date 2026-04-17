import joblib

from features import extract_features

model = joblib.load("model.pkl")


def predict_image(image_path):
    features = extract_features(image_path)

    if features is None:
        return "No face detected"

    prediction = model.predict([features])[0]
    confidence = max(model.predict_proba([features])[0])

    if prediction == 1:
        return f"⚠️ Deepfake Detected | Confidence: {confidence:.2f}"
    else:
        return f"✅ Real Image | Confidence: {confidence:.2f}"


# test
if __name__ == "__main__":
    result = predict_image("test.jpg")
    print(result)
