import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from features import extract_features

DATASET_PATH = "dataset"

X = []
y = []

for label, category in enumerate(["real", "fake"]):
    path = os.path.join(DATASET_PATH, category)

    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        print("Processing:", img_path)

        features = extract_features(img_path)

        if features is None:
            print("❌ No face detected:", img_name)
        else:
            print("✅ Face detected:", img_name)
            X.append(features)
            y.append(label)

X = np.array(X)
y = np.array(y)

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# evaluate
preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))

# save model
joblib.dump(model, "model.pkl")
