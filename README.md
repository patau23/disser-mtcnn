# MTCNN Drunk Detection using Facial Features

![Untitled design](https://github.com/devanys/MTCNN-drunk-recognition/assets/145944367/1d615d4a-9a5b-472e-b255-8536e72c26bf)

This Python script utilizes the MTCNN (Multi-Task Cascaded Convolutional Networks) algorithm for detecting signs of drunken behavior in a person's face using facial features. It combines computer vision techniques with deep learning models for drunk detection to identify potential signs of intoxication.

## Features

![Screenshot 2024-05-24 043239](https://github.com/devanys/MTCNN-drunk-recognition/assets/145944367/8500c152-78d2-4296-b8fc-090b64095531) ![Screenshot 2024-05-24 043303](https://github.com/devanys/MTCNN-drunk-recognition/assets/145944367/a41a2225-6690-4ab4-82a3-db62d4c19726)

- **Real-time detection**: Captures video frames from a webcam and processes them in real-time.
- **Facial detection**: Utilizes the MTCNN (Multi-Task Cascaded Convolutional Networks) algorithm for detecting faces in the video stream.
- **Eye detection**: Uses OpenCV's Haar cascade classifier for detecting eyes within each detected face region.
- **Drunk behavior detection**: Analyzes eye regions for signs of redness, indicative of drunken behavior.
- **Simple interface**: Provides visual feedback by drawing rectangles around detected faces and displaying labels ("Drunk" or "Sober") on the video stream.

## How to Use

1. Clone the repository to your local machine.
2. Install the required dependencies (OpenCV, NumPy, MTCNN, TensorFlow).
3. Run the Python script (`drunk_detection.py`).
   For an alternative approach that allows switching between an SVM and an MLP
   classifier, run `drunk_detection_svm_mlp.py` instead. Press `m` while the
   window is focused to change the active model in real time.
4. Ensure your webcam is connected and accessible.
5. Observe the real-time video stream with detected faces and their associated labels.

## Note

- This script is provided for demonstration purposes and should not be relied upon for real-world applications without proper testing and validation.
- The eye detection model (`facial_drunk.keras`) used in this script should be trained separately using appropriate data and methodologies.
- To use the optional SVM classifier you must supply a trained model saved as `drunk_svm.joblib`. The repository does not include one by default.

### Training your own SVM model

1. Collect a dataset of face images divided into two folders: `sober/` and `drunk/`.
2. Run `python train_svm.py <path_to_dataset>` to train a linear SVM on the images.
3. The script detects faces with MTCNN, normalizes them to 96×96 pixels and saves the model to `drunk_svm.joblib`.
4. Place the resulting file next to `drunk_detection_svm_mlp.py` to enable the SVM mode.

### Training your own MLP model

1. Use the same folder structure with `sober/` and `drunk/` images.
2. Run `python train_mlp.py <path_to_dataset>` to train a small neural network on the images.
3. The script detects faces with MTCNN, normalizes them to 96×96 pixels and saves the model to `facial_drunk.keras`.
4. Put the resulting file next to `drunk_detection_svm_mlp.py` to enable the MLP mode.

## Dependencies

- OpenCV (cv2)
- NumPy
- MTCNN (Multi-Task Cascaded Convolutional Networks)
- TensorFlow

## MTCNN (Multi-Task Cascaded Convolutional Networks)

![image](https://github.com/devanys/MTCNN-drunk-recognition/assets/145944367/c2106ae1-03f1-4901-a5ff-de9af909199f)

MTCNN is a deep learning-based face detection algorithm that detects faces in images and provides bounding boxes and facial landmark points. It consists of three stages:

1. **Proposal Network (P-Net)**: Generates candidate bounding boxes for faces using a convolutional neural network (CNN) and applies bounding box regression to refine the proposals.
2. **Refinement Network (R-Net)**: Filters the candidate bounding boxes generated by the P-Net and performs further refinement to improve the accuracy of face detection.
3. **Output Network (O-Net)**: The final stage performs additional filtering and refinement to produce the final bounding boxes and facial landmark points.

MTCNN is widely used for face detection due to its accuracy and efficiency in detecting faces under various conditions, including variations in scale, pose, and illumination.

### How MTCNN is Used in the Program

1. **Initialization**: The MTCNN detector is initialized at the beginning of the `detect_drunk` function using the `MTCNN()` constructor.
2. **Face Detection**: Once the PNG image is read and converted to RGB format, MTCNN is used to detect faces within the image. This is done by calling the `detect_faces` method of the MTCNN detector, passing the RGB image as input.
3. **Face Analysis**: For each detected face, the program analyzes the facial features to determine signs of redness, which may indicate drunken behavior. This analysis includes extracting a region of interest (ROI) corresponding to the detected face, converting it to the HSV color space, and applying a red color mask to isolate red pixels.
