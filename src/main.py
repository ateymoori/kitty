import cv2
import time
from face_analyzer import FaceAnalyzer
from video_stream import VideoStreamHandler

def main():
    face_analyzer = FaceAnalyzer()
    video_stream = VideoStreamHandler()

    print("Starting video stream. Press 'q' to quit.")

    frame_count = 0
    start_time = time.time()
    fps = 0

    while True:
        frame = video_stream.read_frame()
        frame_count += 1

        faces = face_analyzer.detect_faces(frame)
        
        if len(faces) > 0:
            face = faces[0]
            analysis = face_analyzer.analyze_face(frame, face)
            
            if analysis:
                frame = face_analyzer.draw_analysis(frame, face, analysis, fps)

        video_stream.show_frame(frame)

        if video_stream.wait_key(1) == ord('q'):
            break

        # Calculate FPS every 30 frames
        if frame_count % 30 == 0:
            end_time = time.time()
            fps = 30 / (end_time - start_time)
            start_time = time.time()

    video_stream.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()