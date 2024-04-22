import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from camera import Camera
from math import cos, sin
import cv2
import mediapipe as mp
from decord import VideoReader
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
from insightface.app import FaceAnalysis

class FaceHelper:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        # Initialize FaceDetection once here
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

        self.HEAD_POSE_LANDMARKS = [33, 263, 1, 61, 291, 199]


        # Initialize the Emotion Recognizer
        model_name = 'enet_b0_8_va_mtl'  # Adjust as needed depending on the model availability
        self.fer = HSEmotionRecognizer(model_name=model_name)
        self.emotion_idx_to_class = {0: 'angry', 1: 'contempt', 2: 'disgust', 3: 'fear', 4: 'happy', 
                                     5: 'neutral', 6: 'sad', 7: 'surprise'}
        
        # Initialize the FaceAnalysis module with the ArcFace model
        self.app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
    def __del__(self):
        self.face_detection.close()
        self.face_mesh.close()

    def extract_identity_features(self,img):
        # Convert the input image to RGB format
        img_rgb = img.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5
        img_rgb = (img_rgb * 255).astype('uint8')

        # Perform face detection and recognition using insightface
        faces = self.app.get(img_rgb)

        if len(faces) == 0:
            # If no face is detected, return None
            return None
        else:
            # Take the features of the first detected face
            id_features = faces[0].embedding

            # Convert the features to a torch tensor
            id_features = torch.from_numpy(id_features).float().cuda()

            return id_features

    def detect_emotions(self, image_path):
        """
        Detects emotions on faces in the given image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            dict: A dictionary with coordinates of faces and their corresponding emotions.
        """
        image = cv2.imread(image_path)
        if image is None:
            return "Image not found."

        # Convert the image to RGB as MediaPipe requires RGB images
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces
        results = self.face_detection.process(image_rgb)
        emotions = {}

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                face_img = image_rgb[y:y+h, x:x+w]

                if face_img.size != 0:
                    emotion_idx, _ = self.fer.predict_emotions(face_img)
                    emotion_label = self.emotion_idx_to_class[emotion_idx]
                    emotions[(x, y, w, h)] = emotion_label

                    # Optionally draw detected emotion on the image
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(image, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            return "No face detected."

        # Display the image with detected emotions
        cv2.imshow("Emotion Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return emotions

    def estimate_gaze(self, image):
        """
        Estimate the gaze direction in spherical coordinates (theta, phi) based on eye landmarks.

        Args:
            image (np.array): The input image to process.

        Returns:
            tuple: (theta, phi) if eyes are detected, otherwise None.
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract left and right eye landmarks
                left_eye = np.array([(face_landmarks.landmark[i].x * image.shape[1], 
                                      face_landmarks.landmark[i].y * image.shape[0]) for i in range(362, 374)])
                right_eye = np.array([(face_landmarks.landmark[i].x * image.shape[1], 
                                       face_landmarks.landmark[i].y * image.shape[0]) for i in range(374, 386)])

                left_eye_center = left_eye.mean(axis=0)
                right_eye_center = right_eye.mean(axis=0)

                # Calculate gaze direction in Cartesian coordinates
                gaze_vector = right_eye_center - left_eye_center

                # Convert to spherical coordinates
                x, y = gaze_vector
                r = np.sqrt(x**2 + y**2)
                theta = np.arctan2(y, x)  # Azimuth
                phi = np.arctan2(r, 0)  # Elevation, simplified as z=0 in image plane

                return np.degrees(theta), np.degrees(phi)

        return None

    def generate_face_region_mask(self,frame_image, video_id=0,frame_idx=0):
        frame_np = np.array(frame_image.convert('RGB'))  # Ensure the image is in RGB
        return self.generate_face_region_mask_np_image(video_id,frame_idx,frame_np)

    def generate_face_region_mask_np_image(self,frame_np, video_id=0,frame_idx=0, padding=10):
        # Convert from RGB to BGR for MediaPipe processing
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        height, width, _ = frame_bgr.shape

        # Create a blank mask with the same dimensions as the frame
        mask = np.zeros((height, width), dtype=np.uint8)

        # Optionally save a debug image
        debug_image = mask
        # Detect faces
        detection_results = self.face_detection.process(frame_bgr)
        if detection_results.detections:
            for detection in detection_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                xmin = int(bboxC.xmin * width)
                ymin = int(bboxC.ymin * height)
                bbox_width = int(bboxC.width * width)
                bbox_height = int(bboxC.height * height)

                # Draw a rectangle on the debug image for each detection
                cv2.rectangle(debug_image, (xmin, ymin), (xmin + bbox_width, ymin + bbox_height), (0, 255, 0), 2)
        # Check that detections are not None
        if detection_results.detections:
            for detection in detection_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                xmin = int(bboxC.xmin * width)
                ymin = int(bboxC.ymin * height)
                bbox_width = int(bboxC.width * width)
                bbox_height = int(bboxC.height * height)

                # Calculate padded coordinates
                pad_xmin = max(0, xmin - padding)
                pad_ymin = max(0, ymin - padding)
                pad_xmax = min(width, xmin + bbox_width + padding)
                pad_ymax = min(height, ymin + bbox_height + padding)

                # Draw a white padded rectangle on the mask
                mask[pad_ymin:pad_ymax, pad_xmin:pad_xmax] = 255

               
                # cv2.rectangle(debug_image, (pad_xmin, pad_ymin), 
                            #   (pad_xmax, pad_ymax), (255, 255, 255), thickness=-1)
                # cv2.imwrite(f'./temp/debug_face_mask_{video_id}-{frame_idx}.png', debug_image)

        return mask

    
    def generate_face_region_mask_pil_image(self,frame_image,video_id=0, frame_idx=0):
        # Convert from PIL Image to NumPy array in BGR format
        frame_np = np.array(frame_image.convert('RGB'))  # Ensure the image is in RGB
        return self.generate_face_region_mask_np_image(frame_np,video_id,frame_idx,)
    


    def calculate_pose(self, face2d):
            """Calculates head pose from detected facial landmarks using 
            Perspective-n-Point (PnP) pose computation:
            
            https://docs.opencv.org/4.6.0/d5/d1f/calib3d_solvePnP.html
            """
            # print('Computing head pose from tracking data...')
            # for idx, time in enumerate(self.face2d['time']):
            #     # print(time)
            #     self.pose['time'].append(time)
            #     self.pose['frame'].append(self.face2d['frame'][idx])
            #     face2d = self.face2d['key landmark positions'][idx]
            face3d = [[0, -1.126865, 7.475604], # 1
                        [-4.445859, 2.663991, 3.173422], # 33
                        [-2.456206,	-4.342621, 4.283884], # 61
                        [0, -9.403378, 4.264492], # 199
                        [4.445859, 2.663991, 3.173422], # 263
                        [2.456206, -4.342621, 4.283884]] # 291
            face2d = np.array(face2d, dtype=np.float64)
            face3d = np.array(face3d, dtype=np.float64)

            camera = Camera()
            success, rot_vec, trans_vec = cv2.solvePnP(face3d,
                                                        face2d,
                                                        camera.internal_matrix,
                                                        camera.distortion_matrix,
                                                        flags=cv2.SOLVEPNP_ITERATIVE)
            
            rmat = cv2.Rodrigues(rot_vec)[0]

            P = np.hstack((rmat, np.zeros((3, 1), dtype=np.float64)))
            eulerAngles =  cv2.decomposeProjectionMatrix(P)[6]
            yaw = eulerAngles[1, 0]
            pitch = eulerAngles[0, 0]
            roll = eulerAngles[2,0]
            
            if pitch < 0:
                pitch = - 180 - pitch
            elif pitch >= 0: 
                pitch = 180 - pitch
            
            yaw *= -1
            pitch *= -1
            
            # if nose2d:
            #     nose2d = nose2d
            #     p1 = (int(nose2d[0]), int(nose2d[1]))
            #     p2 = (int(nose2d[0] - yaw * 2), int(nose2d[1] - pitch * 2))
            
            return yaw, pitch, roll 

    def draw_axis(self, img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):
        # Referenced from HopeNet https://github.com/natanielruiz/deep-head-pose
        pitch = pitch * np.pi / 180
        yaw = -(yaw * np.pi / 180)
        roll = roll * np.pi / 180

        if tdx != None and tdy != None:
            tdx = tdx
            tdy = tdy
        else:
            height, width = img.shape[:2]
            tdx = width / 2
            tdy = height / 2

        # X-Axis pointing to right. drawn in red
        x1 = size * (cos(yaw) * cos(roll)) + tdx
        y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

        # Y-Axis | drawn in green
        #        v
        x2 = size * (-cos(yaw) * sin(roll)) + tdx
        y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

        # Z-Axis (out of the screen) drawn in blue
        x3 = size * (sin(yaw)) + tdx
        y3 = size * (-cos(yaw) * sin(pitch)) + tdy

        cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),2)
        cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),2)
        cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)
        return img

    def get_head_pose(self, image_path):
        """
        Given an image, estimate the head pose (roll, pitch, yaw angles).

        Args:
            image: Image to estimate head pose.

        Returns:
            tuple: Roll, Pitch, Yaw angles if face landmarks are detected, otherwise None.
        """


    # Define the landmarks that represent the head pose.

        image = cv2.imread(image_path)
        # Convert the image to RGB.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image to detect face landmarks.
        results = self.mp_face_mesh.process(image_rgb)

        img_h, img_w, _ = image.shape
        face_3d = []
        face_2d = []


        if results.multi_face_landmarks:       
            for face_landmarks in results.multi_face_landmarks:
                key_landmark_positions=[]
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in self.HEAD_POSE_LANDMARKS:
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])

                        landmark_position = [x,y]
                        key_landmark_positions.append(landmark_position)
                # Convert to numpy arrays
                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                # Camera matrix
                focal_length = img_w  # Assuming fx = fy
                cam_matrix = np.array(
                    [[focal_length, 0, img_w / 2],
                    [0, focal_length, img_h / 2],
                    [0, 0, 1]]
                )

                # Distortion matrix
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP to get rotation vector
                success, rot_vec, trans_vec = cv2.solvePnP(
                    face_3d, face_2d, cam_matrix, dist_matrix
                )
                yaw, pitch, roll = self.calculate_pose(key_landmark_positions)
                print(f'Roll: {roll:.4f}, Pitch: {pitch:.4f}, Yaw: {yaw:.4f}')
                self.draw_axis(image, yaw, pitch, roll)
                debug_image_path = image_path.replace('.jpg', '_debug.jpg')  # Modify as needed
                cv2.imwrite(debug_image_path, image)
                print(f'Debug image saved to {debug_image_path}')
                
                return roll, pitch, yaw 

        return None




    def get_head_pose_velocities_at_frame(self, video_reader: VideoReader, frame_index, n_previous_frames=2):

        # Adjust frame_index if it's larger than the total number of frames
        total_frames = len(video_reader)
        frame_index = min(frame_index, total_frames - 1)

        # Calculate starting index for previous frames
        start_index = max(0, frame_index - n_previous_frames)

        head_poses = []
        for idx in range(start_index, frame_index + 1):
            # idx is the frame index you want to access
            frame_tensor = video_reader[idx]

            #  check emodataset decord.bridge.set_bridge('torch')  # Optional: This line sets decord to directly output PyTorch tensors.
            # Assert that frame_tensor is a PyTorch tensor
            assert isinstance(frame_tensor, torch.Tensor), "Expected a PyTorch tensor"

            image = video_reader[idx].numpy()
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)

            img_h, img_w, _ = image.shape
            face_3d = []
            face_2d = []

            if results.multi_face_landmarks:       
                for face_landmarks in results.multi_face_landmarks:
                    key_landmark_positions=[]
                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx in self.HEAD_POSE_LANDMARKS:
                            x, y = int(lm.x * img_w), int(lm.y * img_h)
                            face_2d.append([x, y])
                            face_3d.append([x, y, lm.z])

                            landmark_position = [x,y]
                            key_landmark_positions.append(landmark_position)
                    # Convert to numpy arrays
                    face_2d = np.array(face_2d, dtype=np.float64)
                    face_3d = np.array(face_3d, dtype=np.float64)

                    # Camera matrix
                    focal_length = img_w  # Assuming fx = fy
                    cam_matrix = np.array(
                        [[focal_length, 0, img_w / 2],
                        [0, focal_length, img_h / 2],
                        [0, 0, 1]]
                    )

                    # Distortion matrix
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)

                    # Solve PnP to get rotation vector
                    success, rot_vec, trans_vec = cv2.solvePnP(
                        face_3d, face_2d, cam_matrix, dist_matrix
                    )
                    yaw, pitch, roll = self.calculate_pose(key_landmark_positions)
                    head_poses.append((roll, pitch, yaw))

        # Calculate velocities
        head_velocities = []
        for i in range(len(head_poses) - 1):
            roll_diff = head_poses[i + 1][0] - head_poses[i][0]
            pitch_diff = head_poses[i + 1][1] - head_poses[i][1]
            yaw_diff = head_poses[i + 1][2] - head_poses[i][2]
            head_velocities.append((roll_diff, pitch_diff, yaw_diff))

        return head_velocities

