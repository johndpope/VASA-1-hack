import cv2
import mediapipe as mp
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Initialize the Emotion Recognizer
model_name = 'enet_b0_8_va_mtl'
fer = HSEmotionRecognizer(model_name=model_name)
emotion_idx_to_class = {0: 'angry', 1: 'contempt', 2: 'disgust', 3: 'fear', 4: 'happy', 5: 'neutral', 6: 'sad', 7: 'surprise'}

# Load an image
image_path = 'path_to_your_image.jpg'
image = cv2.imread(image_path)
if image is None:
    print("Image not found.")
    exit()

# Convert the image to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect faces
results = face_detection.process(image_rgb)
if results.detections:
    for detection in results.detections:
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = image.shape
        bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
        x, y, w, h = bbox
        face_img = image_rgb[y:y+h, x:x+w]

        if face_img.size != 0:
            emotion, _ = fer.predict_emotions(face_img)
            emotion_label = emotion_idx_to_class[emotion]
            print("Detected emotion:", emotion_label)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
else:
    print("No face detected.")

# Display the image
cv2.imshow("Emotion Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
