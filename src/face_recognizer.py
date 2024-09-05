import dlib
import cv2
import os
import numpy as np
import pickle

class FaceRecognizer:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
        self.face_rec_model = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")
        self.known_faces = {}
        self.load_known_faces()

    def load_known_faces(self):
        if os.path.exists('data/known_faces.pkl'):
            with open('data/known_faces.pkl', 'rb') as f:
                self.known_faces = pickle.load(f)
        else:
            for filename in os.listdir('data/known_faces'):
                if filename.endswith(('.jpg', '.png')):
                    name = os.path.splitext(filename)[0]
                    image_path = os.path.join('data/known_faces', filename)
                    face_encoding = self.get_face_encoding(cv2.imread(image_path))
                    if face_encoding is not None:
                        self.known_faces[name] = face_encoding
            
            with open('data/known_faces.pkl', 'wb') as f:
                pickle.dump(self.known_faces, f)

    def get_face_encoding(self, image):
        faces = self.detector(image)
        if len(faces) > 0:
            shape = self.shape_predictor(image, faces[0])
            face_encoding = np.array(self.face_rec_model.compute_face_descriptor(image, shape))
            return face_encoding
        return None

    def recognize_face(self, face_encoding):
        if len(self.known_faces) == 0:
            return "Unknown"
        
        distances = []
        for name, known_encoding in self.known_faces.items():
            dist = np.linalg.norm(face_encoding - known_encoding)
            distances.append((dist, name))
        
        distances.sort()
        if distances[0][0] < 0.6:  # Adjust this threshold as needed
            return distances[0][1]
        else:
            return "Unknown"

    def add_known_face(self, name, image):
        face_encoding = self.get_face_encoding(image)
        if face_encoding is not None:
            self.known_faces[name] = face_encoding
            with open('data/known_faces.pkl', 'wb') as f:
                pickle.dump(self.known_faces, f)
            return True
        return False