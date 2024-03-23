import face_recognition
import cv2
from datetime import datetime

class BlurVideo:
    INPUT_DIR = "/app/images/input/"
    OUTPUT_DIR = "/app/images/output/" 

    def __init__(self, filename):
        self.filename = filename
        self.inputfile = f"{self.INPUT_DIR}{self.filename}"
     
    def run(self):
        movie = cv2.VideoCapture(self.inputfile)
        fps = movie.get(cv2.CAP_PROP_FPS)
        width = int(movie.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(movie.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        formatted_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
        new_file_name = f"output_{formatted_datetime}.mp4"
        writer = cv2.VideoWriter(f"{self.OUTPUT_DIR}{new_file_name}", fourcc, fps, (width, height))
        count = 0
        need_show_detail = False
        start_time = datetime.now().strftime("%Y%m%d%H%M%S")
        while True:
            ret, frame = movie.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_locations = face_recognition.face_locations(image, model='hog')
            if len(face_locations) == 1 or need_show_detail:
                if need_show_detail:
                    need_show_detail = False
                else:
                    need_show_detail = True
            for top, right, bottom, left in face_locations:
                face_image = frame[top:bottom, left:right]
                face_image = cv2.GaussianBlur(face_image, (99, 99), 30)
                frame[top:bottom, left:right] = face_image
            writer.write(frame)
            count += 1
            print(count)

        end_time = datetime.now().strftime("%Y%m%d%H%M%S")
        print(f"exec duration: {int(end_time) - int(start_time)} sec")
        writer.release()
        movie.release()
        cv2.destroyAllWindows()

# Usage example:
# BlurVideo("sample.mp4").run()
