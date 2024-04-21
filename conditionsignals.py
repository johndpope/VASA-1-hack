import cv2
import dlib
import numpy as np

# Initialize dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def estimate_gaze(image_path):
    # Load the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = detector(gray)
    for face in faces:
        # Get landmarks
        landmarks = predictor(gray, face)
        
        # Assuming that the eye landmarks are as follows:
        # 36-41: Left eye, 42-47: Right eye
        left_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)])
        right_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)])
        
        # Simple gaze direction: compute the centroid of the eye landmarks
        left_eye_center = left_eye.mean(axis=0)
        right_eye_center = right_eye.mean(axis=0)
        
        # Compute gaze direction by eye center position
        gaze_direction = right_eye_center - left_eye_center
        
        # Display results
        cv2.circle(img, tuple(left_eye_center.astype(int)), 5, (255, 0, 0), -1)
        cv2.circle(img, tuple(right_eye_center.astype(int)), 5, (0, 0, 255), -1)
        cv2.arrowedLine(img, tuple(left_eye_center.astype(int)), tuple(left_eye_center.astype(int) + gaze_direction), (0, 255, 0), 2)
    
    cv2.imshow('Gaze Estimation', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Usage
estimate_gaze('path_to_your_image.jpg')



