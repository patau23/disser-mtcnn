import cv2
import dlib
import numpy as np
import os

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

FACE_LANDMARKS_IDXS = {
    "jaw": (0, 17),
    "right_eyebrow": (17, 22),
    "left_eyebrow": (22, 27),
    "nose": (27, 36),
    "right_eye": (36, 42),
    "left_eye": (42, 48),
    "mouth": (48, 68),
}

YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)


def draw_landmarks(image, landmarks, regions=None, color=(0, 255, 255)):
    if regions is None:
        regions = FACE_LANDMARKS_IDXS.keys()
    for name in regions:
        (start, end) = FACE_LANDMARKS_IDXS[name]
        points = [(landmarks.part(i).x, landmarks.part(i).y)
                  for i in range(start, end)]
        for p in points:
            # увеличиваем радиус круга
            cv2.circle(image, p, 2, color, -1)
        is_closed = name not in ["jaw", "nose"]
        # увеличиваем толщину линий
        cv2.polylines(
            image,
            [np.array(points)],
            isClosed=is_closed,
            color=color,
            thickness=3
        )
    return image


def process_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if not faces:
        raise Exception("No face found")

    landmarks = predictor(gray, faces[0])

    # Варианты отображения:
    variants = []

    # 1. Все 68 точек (жёлтый)
    img1 = image.copy()
    draw_landmarks(img1, landmarks, color=YELLOW)
    variants.append(img1)

    # 2. Без jaw
    img2 = image.copy()
    draw_landmarks(img2, landmarks, regions=[k for k in FACE_LANDMARKS_IDXS if k != "jaw"], color=YELLOW)
    variants.append(img2)

    # 3. Только глаза, брови, рот
    img3 = image.copy()
    draw_landmarks(img3, landmarks, regions=["right_eye", "left_eye", ], color=YELLOW)
    variants.append(img3)
    
    # 3. Только глаза, брови, рот
    img4 = image.copy()
    draw_landmarks(img4, landmarks, regions=[ "right_eyebrow", "left_eyebrow", ], color=YELLOW)
    variants.append(img4)
    
    # 3. Только глаза, брови, рот
    img5 = image.copy()
    draw_landmarks(img5, landmarks, regions=[ "mouth"], color=YELLOW)
    variants.append(img5)

    # 4. Только брови и рот
    img6 = image.copy()
    draw_landmarks(img6, landmarks, regions=["right_eye", "left_eye", "right_eyebrow", "left_eyebrow", "mouth"], color=YELLOW)
    variants.append(img6)

    # 5. Все в зелёном
    img7 = image.copy()
    draw_landmarks(img7, landmarks, color=GREEN)
    variants.append(img7)

    return variants


if __name__ == "__main__":
    variants = process_image("person01.png")  # ← здесь укажи путь, если другое имя

    # Горизонтальное объединение
    combined = np.hstack([cv2.resize(im, (300, 300)) for im in variants])
    cv2.imshow("Landmark Variants", combined)
    cv2.imwrite("landmark_variants_combined.png", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
