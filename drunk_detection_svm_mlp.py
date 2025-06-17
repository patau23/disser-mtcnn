import cv2
import numpy as np
from mtcnn import MTCNN
import joblib
from tensorflow.keras.models import load_model

# Initialize the MTCNN detector once
_detector = MTCNN()

# Try loading the models. Users should provide the appropriate paths.
try:
    _svm_model = joblib.load("drunk_svm.joblib")
except Exception as e:
    print(f"Warning: could not load SVM model: {e}")
    _svm_model = None

try:
    _mlp_model = load_model("facial_drunk.keras")
except Exception as e:
    print(f"Warning: could not load MLP model: {e}")
    _mlp_model = None


def detect_and_align_face(frame):
    """Detects the largest face in the frame and aligns it by eye line."""
    results = _detector.detect_faces(frame)
    if not results:
        return None, None

    det = results[0]
    x, y, w, h = det["box"]
    keypoints = det.get("keypoints", {})
    left_eye = keypoints.get("left_eye")
    right_eye = keypoints.get("right_eye")

    if left_eye and right_eye:
        eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))
        M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
        rows, cols = frame.shape[:2]
        rotated = cv2.warpAffine(frame, M, (cols, rows), flags=cv2.INTER_LINEAR)

        box_points = np.float32([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
        ones = np.ones((4, 1), dtype=np.float32)
        box_points = np.hstack([box_points, ones])
        transformed = box_points.dot(M.T)
        min_x = int(np.min(transformed[:, 0]))
        max_x = int(np.max(transformed[:, 0]))
        min_y = int(np.min(transformed[:, 1]))
        max_y = int(np.max(transformed[:, 1]))
        min_x = max(min_x, 0)
        min_y = max(min_y, 0)
        max_x = min(max_x, cols)
        max_y = min(max_y, rows)
        face_img = rotated[min_y:max_y, min_x:max_x]
        face_box = (min_x, min_y, max_x - min_x, max_y - min_y)
    else:
        face_img = frame[y:y + h, x:x + w]
        face_box = (x, y, w, h)

    return face_img, face_box


def preprocess_face(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (96, 96))
    normalized = resized.astype("float32") / 255.0
    return normalized


def classify_with_svm(face_array):
    if _svm_model is None:
        return "Unknown", 0.0
    flat = face_array.flatten().reshape(1, -1)
    pred = _svm_model.predict(flat)[0]
    label = "Drunk" if pred == 1 or str(pred).lower() == "drunk" else "Sober"
    confidence = 1.0
    if hasattr(_svm_model, "predict_proba"):
        proba = _svm_model.predict_proba(flat)[0]
        if hasattr(_svm_model, "classes_"):
            if label == "Drunk":
                idx = list(_svm_model.classes_).index(1) if 1 in _svm_model.classes_ else list(_svm_model.classes_).index("Drunk")
            else:
                idx = list(_svm_model.classes_).index(0) if 0 in _svm_model.classes_ else list(_svm_model.classes_).index("Sober")
        else:
            idx = 1 if label == "Drunk" else 0
        confidence = float(proba[idx])
    elif hasattr(_svm_model, "decision_function"):
        score = _svm_model.decision_function(flat)
        prob = 1 / (1 + np.exp(-score[0]))
        confidence = float(prob) if label == "Drunk" else float(1 - prob)
    return label, confidence


def classify_with_mlp(face_array):
    if _mlp_model is None:
        return "Unknown", 0.0
    tensor = face_array.reshape(1, 96, 96, 1)
    preds = _mlp_model.predict(tensor)
    if preds.shape[-1] == 1:
        prob_drunk = float(preds[0][0])
        if prob_drunk >= 0.5:
            return "Drunk", prob_drunk
        return "Sober", 1 - prob_drunk
    else:
        prob_vec = preds[0]
        class_index = int(np.argmax(prob_vec))
        label = "Drunk" if class_index == 1 else "Sober"
        return label, float(prob_vec[class_index])


def main():
    current_model = "svm"

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera")
        return
    print(f"Starting detection. Current model: {current_model.upper()}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face_img, face_box = detect_and_align_face(frame)
        if face_img is not None:
            face_array = preprocess_face(face_img)
            if current_model == "svm":
                label, conf = classify_with_svm(face_array)
            else:
                label, conf = classify_with_mlp(face_array)
            x, y, w, h = face_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"{label}: {conf * 100:.1f}%"
            ty = y - 10 if y - 10 > 20 else y + 20
            cv2.putText(frame, text, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Drunk Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        if key == ord('m'):
            current_model = "mlp" if current_model == "svm" else "svm"
            print(f"Switched to {current_model.upper()}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
