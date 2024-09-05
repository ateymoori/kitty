import cv2
import sys
from face_detector import FaceDetector
from video_stream import VideoStreamHandler

def main():
    try:
        face_detector = FaceDetector()
        video_stream = VideoStreamHandler()

        print("Starting video stream. Press 'q' to quit.")

        while True:
            try:
                frame = video_stream.read_frame()
            except ValueError as e:
                print(f"Error: {e}")
                print("Please check if your camera is connected and not in use by another application.")
                break

            faces = face_detector.detect_faces(frame)
            frame_with_faces = face_detector.draw_faces(frame, faces)

            video_stream.show_frame(frame_with_faces)

            if video_stream.wait_key() == ord('q'):
                break

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if 'video_stream' in locals():
            video_stream.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()