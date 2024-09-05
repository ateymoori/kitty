import cv2
from imutils.video import VideoStream
import imutils
import time

class VideoStreamHandler:
    def __init__(self, src=0, width=640):
        self.stream = VideoStream(src=src)
        self.width = width
        self.start()

    def start(self):
        self.stream.start()
        time.sleep(2.0)  # Warm up the camera

    def read_frame(self):
        frame = self.stream.read()
        if frame is None:
            raise ValueError("Failed to capture frame from camera")
        frame = imutils.resize(frame, width=self.width)
        frame = cv2.flip(frame, 1) 
        return frame

    def release(self):
        self.stream.stop()

    @staticmethod
    def show_frame(frame, window_name="Face Detection"):
        cv2.imshow(window_name, frame)

    @staticmethod
    def wait_key(delay=1):
        return cv2.waitKey(delay) & 0xFF