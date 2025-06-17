import os
import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def detect_and_align_face(frame):
    """Detect a face with MTCNN and align it by the eye line."""
    detector = getattr(detect_and_align_face, "detector", None)
    if detector is None:
        detector = MTCNN()
        detect_and_align_face.detector = detector
    results = detector.detect_faces(frame)
    if not results:
        return None
    det = results[0]
    x, y, w, h = det["box"]
    kpts = det.get("keypoints", {})
    left = kpts.get("left_eye")
    right = kpts.get("right_eye")
    if left and right:
        center = ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)
        dx = right[0] - left[0]
        dy = right[1] - left[1]
        angle = np.degrees(np.arctan2(dy, dx))
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rows, cols = frame.shape[:2]
        rotated = cv2.warpAffine(frame, M, (cols, rows), flags=cv2.INTER_LINEAR)
        pts = np.float32([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
        pts = np.hstack([pts, np.ones((4, 1), dtype=np.float32)])
        trans = pts.dot(M.T)
        min_x = int(np.min(trans[:, 0])); max_x = int(np.max(trans[:, 0]))
        min_y = int(np.min(trans[:, 1])); max_y = int(np.max(trans[:, 1]))
        min_x = max(min_x, 0); min_y = max(min_y, 0)
        max_x = min(max_x, cols); max_y = min(max_y, rows)
        return rotated[min_y:max_y, min_x:max_x]
    return frame[y:y + h, x:x + w]

def preprocess_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (96, 96))
    norm = resized.astype("float32") / 255.0
    return norm.reshape(96, 96, 1)


def load_dataset(root):
    data = []
    labels = []
    for name, label in [("sober", 0), ("drunk", 1)]:
        folder = os.path.join(root, name)
        if not os.path.isdir(folder):
            continue
        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            img = cv2.imread(path)
            if img is None:
                continue
            face = detect_and_align_face(img)
            if face is None:
                continue
            arr = preprocess_face(face)
            data.append(arr)
            labels.append(label)
    return np.array(data), np.array(labels)

def build_model():
    model = Sequential([
        Flatten(input_shape=(96, 96, 1)),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def train_and_save(dataset_dir, out_path="facial_drunk.keras", epochs=10, batch_size=32):
    X, y = load_dataset(dataset_dir)
    if len(X) == 0:
        raise RuntimeError("No training data found")
    model = build_model()
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    model.save(out_path)
    print(f"Model saved to {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train MLP for drunk detection")
    parser.add_argument("dataset", help="Path to dataset with 'sober' and 'drunk' folders")
    parser.add_argument("--out", default="facial_drunk.keras", help="Output model file")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    args = parser.parse_args()
    train_and_save(args.dataset, args.out, args.epochs, args.batch)


# Если запускать через MLP обучение - то нужно учитывать что он заменит старый формат facial_drunk.keras
# Старый facial_drunk.keras - находится в директории old-keras если нужно использовать его
# КОМАНДА ДЛЯ ЗАПУСКА:
# python3 train_mlp.py ./dataset
