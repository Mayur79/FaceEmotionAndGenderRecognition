# Main code

from tkinter import *
from threading import Thread
from keras.models import load_model
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
from PIL import Image, ImageTk

root = Tk()
root.title("Emotion and Gender Recognition")
# root.iconbitmap('icon.ico')
root.configure(bg='white')

# Create a canvas to display the video feed
canvas = Canvas(root, width=640, height=480, bg='#F0F0F0')
canvas.pack(pady=10)


class Application(Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()
        self.stop_flag = False

    def create_widgets(self):
        button_frame = Frame(self, bg='#ECECEC')
        button_frame.pack(pady=10)
        self.start_button = Button(
            self, text="Start", command=self.start_recognition, font=("Arial", 14), bg="#008CBA", fg="black")
        self.start_button.pack(side="left", padx=10, pady=10)
        self.stop_button = Button(
            self, text="Stop", command=self.stop_recognition, font=("Arial", 14), bg="#f44336", fg="black")
        self.stop_button.pack(side="left", padx=10, pady=10)

    def start_recognition(self):
        self.recognition_thread = Thread(target=self.recognition_loop)
        self.recognition_thread.start()

    def stop_recognition(self):
        self.stop_flag = True
        self.recognition_thread.join()
        # release the VideoCapture object to stop the camera
        self.cap.release()

    def recognition_loop(self):
        face_classifier = cv2.CascadeClassifier(
            'C:\\Users\\mayur_iyd6xcu\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
        emotion_model = load_model(
            'C:\\Users\\mayur_iyd6xcu\\OneDrive\\Desktop\\FaceEmotionGenderProject\\faceemotion\\model.h5')
        gender_model = load_model(
            'C:\\Users\\mayur_iyd6xcu\\OneDrive\\Desktop\\FaceEmotionGenderProject\\faceemotion\\gender_classification_model.h5')

        class_labels = ['Angry', 'Disgust', 'Fear',
                        'Happy', 'Neutral', 'Sad', 'Surprise']
        gender_labels = ['Female', 'Male']

        cap = cv2.VideoCapture(0)

        # while True:
        while not self.stop_flag:
            ret, frame = cap.read()
            labels = []

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48),
                                      interpolation=cv2.INTER_AREA)

                # Get image ready for prediction
                roi = roi_gray.astype('float')/255.0  # Scale
                roi = img_to_array(roi)
                # Expand dims to get it ready for prediction (1, 48, 48, 1)
                roi = np.expand_dims(roi, axis=0)

                # Yields one hot encoded result for 7 classes
                preds = emotion_model.predict(roi)[0]
                label = class_labels[preds.argmax()]  # Find the label
                label_position = (x, y)
                cv2.putText(frame, label, label_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Gender
                roi_color = frame[y:y+h, x:x+w]
                roi_color = cv2.resize(
                    roi_color, (150, 150), interpolation=cv2.INTER_AREA)
                gender_predict = gender_model.predict(
                    np.array(roi_color).reshape(-1, 150, 150, 3))
                gender_predict = (gender_predict >= 0.5).astype(int)[:, 0]
                gender_label = gender_labels[gender_predict[0]]
                # 50 pixels below to move the label outside the face
                gender_label_position = (x, y+h+50)
                cv2.putText(frame, gender_label, gender_label_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img = ImageTk.PhotoImage(image=img)
                canvas.create_image(0, 0, anchor=NW, image=img)
                root.update()
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


app = Application(master=root)
app.mainloop()
