import math
import os

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import Sequence, to_categorical

from features import IMG_SIZE, preprocess_image

DATASET_PATH = "dataset"
MODEL_PATH = "cnn_model.h5"
BATCH_SIZE = 16
EPOCHS = 10


class FaceDataSequence(Sequence):
    def __init__(self, image_paths, labels, batch_size, shuffle=True):
        self.image_paths = list(image_paths)
        self.labels = list(labels)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.image_paths))
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.image_paths) / self.batch_size)

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.indices))

        images = []
        labels = []

        # Keep moving forward until we collect at least one valid sample.
        while start < len(self.indices) and not images:
            batch_indices = self.indices[start:end]

            for batch_index in batch_indices:
                image_path = self.image_paths[batch_index]
                label = self.labels[batch_index]
                image = preprocess_image(image_path)

                if image is None:
                    continue

                images.append(image)
                labels.append(label)

            start = end
            end = min(start + self.batch_size, len(self.indices))

        if not images:
            raise ValueError("No valid images found in the dataset. Check face detection.")

        return np.array(images, dtype=np.float32), to_categorical(labels, num_classes=2)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def collect_samples():
    image_paths = []
    labels = []
    categories = {"real": 0, "fake": 1}

    for category, label in categories.items():
        path = os.path.join(DATASET_PATH, category)

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)

            if os.path.isfile(img_path):
                image_paths.append(img_path)
                labels.append(label)

    if not image_paths:
        raise ValueError("No training images found. Check your dataset folders.")

    return image_paths, labels


image_paths, labels = collect_samples()

train_paths, test_paths, train_labels, test_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42, stratify=labels
)

train_data = FaceDataSequence(train_paths, train_labels, batch_size=BATCH_SIZE, shuffle=True)
test_data = FaceDataSequence(test_paths, test_labels, batch_size=BATCH_SIZE, shuffle=False)

model = Sequential(
    [
        Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(2, activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(train_data, epochs=EPOCHS, validation_data=test_data)

loss, accuracy = model.evaluate(test_data, verbose=0)
print("Test Accuracy:", accuracy)

model.save(MODEL_PATH)
