import cv2
import numpy as np
from deepface import DeepFace
import dlib

class FaceAnalyzer:
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
        self.last_face = None
        self.last_age_gender = None
        self.face_threshold = 50  # pixels

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.face_detector(gray)

    def get_landmarks(self, frame, face):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.predictor(gray, face)

    def is_new_face(self, face):
        if self.last_face is None:
            return True
        current_center = np.array([(face.left() + face.right()) // 2, (face.top() + face.bottom()) // 2])
        last_center = np.array([(self.last_face.left() + self.last_face.right()) // 2, 
                                (self.last_face.top() + self.last_face.bottom()) // 2])
        return np.linalg.norm(current_center - last_center) > self.face_threshold

    def analyze_face(self, frame, face):
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_img = frame[y:y+h, x:x+w]
        
        analysis = {}
        
        if self.is_new_face(face):
            try:
                deep_analysis = DeepFace.analyze(face_img, actions=['age', 'gender'], enforce_detection=False)
                self.last_age_gender = {
                    'age': deep_analysis[0]['age'],
                    'gender': deep_analysis[0]['dominant_gender']
                }
            except Exception as e:
                print(f"Error in age/gender analysis: {e}")
                self.last_age_gender = {'age': 'Unknown', 'gender': 'Unknown'}
            
            self.last_face = face

        analysis.update(self.last_age_gender)

        try:
            emotion_analysis = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
            analysis['emotion'] = emotion_analysis[0]['dominant_emotion']
        except Exception as e:
            print(f"Error in emotion analysis: {e}")
            analysis['emotion'] = 'Unknown'

        landmarks = self.get_landmarks(frame, face)
        analysis['left_eye'] = self.get_eye_status(landmarks, 'left')
        analysis['right_eye'] = self.get_eye_status(landmarks, 'right')
        analysis['mouth'] = self.get_mouth_status(landmarks)

        return analysis

    def get_eye_status(self, landmarks, eye):
        if eye == 'left':
            points = [36, 37, 38, 39, 40, 41]
        else:
            points = [42, 43, 44, 45, 46, 47]
        
        eye_points = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in points])
        ear = self.eye_aspect_ratio(eye_points)
        return 'Open' if ear > 0.2 else 'Closed'

    def eye_aspect_ratio(self, eye):
        vertical_dist = np.linalg.norm(eye[1] - eye[5]) + np.linalg.norm(eye[2] - eye[4])
        horizontal_dist = np.linalg.norm(eye[0] - eye[3])
        return vertical_dist / (2.0 * horizontal_dist)

    def get_mouth_status(self, landmarks):
        mouth_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)])
        mar = self.mouth_aspect_ratio(mouth_points)
        return 'Open' if mar > 0.5 else 'Closed'

    def mouth_aspect_ratio(self, mouth):
        vertical_dist = np.linalg.norm(mouth[2] - mouth[10]) + np.linalg.norm(mouth[4] - mouth[8])
        horizontal_dist = np.linalg.norm(mouth[0] - mouth[6])
        return vertical_dist / (2.0 * horizontal_dist)

    def draw_analysis(self, frame, face, analysis, fps):
        height, width = frame.shape[:2]
        
        # Draw face border
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Prepare text lines
        lines = [
            f"Age: {analysis['age']}",
            f"Gender: {analysis['gender']}",
            f"Emotion: {analysis['emotion']}",
            f"Left Eye: {analysis['left_eye']}",
            f"Right Eye: {analysis['right_eye']}",
            f"Mouth: {analysis['mouth']}",
            f"FPS: {fps:.2f}"
        ]
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        color = (0, 255, 0)  # Light green
        
        # Calculate text positions
        text_y = height - 10 - (len(lines) - 1) * 20
        for line in lines:
            cv2.putText(frame, line, (10, text_y), font, font_scale, color, font_thickness)
            text_y += 20
        
        return frame