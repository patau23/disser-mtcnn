import cv2
import numpy as np
from mtcnn import MTCNN
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier


def get_red_ratio(eyes, roi_color, roi_gray):
    ratios = []
    for (ex, ey, ew, eh) in eyes:
        eye_color = roi_color[ey:ey + eh, ex:ex + ew]
        hsv = cv2.cvtColor(eye_color, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
        red_eye = cv2.bitwise_and(eye_color, eye_color, mask=mask)
        red_eye_gray = cv2.cvtColor(red_eye, cv2.COLOR_BGR2GRAY)
        ratio = cv2.countNonZero(red_eye_gray) / float(ew * eh)
        ratios.append(ratio)
    return max(ratios) if ratios else 0


detector = MTCNN()

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

eye_model = tf.keras.models.load_model('facial_drunk.keras')

# Simple dataset for demonstration of the RandomForest classifier
X_train = np.array([[0.0], [0.05], [0.1], [0.2], [0.4], [0.6], [0.8], [1.0]])
y_train = np.array([0, 0, 0, 1, 1, 1, 1, 1])

# Train the RandomForest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    faces = detector.detect_faces(rgb_frame)
    
    for face in faces:
        x, y, width, height = face['box']
        roi_color = frame[y:y + height, x:x + width]
        roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
        
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)

        red_ratio = get_red_ratio(eyes, roi_color, roi_gray)
        prediction = rf_classifier.predict([[red_ratio]])[0]

        if prediction == 0:
            label = "Sober"
            color = (0, 255, 0)
        else:
            label = "Drunk"
            color = (0, 0, 255)
        
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    cv2.imshow('Drunk Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
