import cv2
import os
from tkinter import Tk, Label, Button, filedialog, Canvas, PhotoImage
from PIL import Image, ImageTk


def load_cascade(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing cascade file: {path}")
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise IOError(f"Could not load cascade: {path}")
    return cascade

faceCascade = load_cascade('haarcascade_frontalface_default.xml')
eyeCascade = load_cascade('haarcascade_eye.xml')
noseCascade = load_cascade('Nariz.xml')
mouthCascade = load_cascade('Mouth.xml')

# ---------- Detection Functions ----------
def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, text, (x, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords

def detect(img):
    colors = {"face": (255,0,0), "eye": (0,0,255), "nose": (0,255,0), "mouth": (255,255,255)}
    coords = draw_boundary(img, faceCascade, 1.1, 10, colors["face"], "Face")
    if len(coords) == 4:
        x, y, w, h = coords
        roi_img = img[y:y+h, x:x+w]
        draw_boundary(roi_img, eyeCascade, 1.1, 12, colors["eye"], "Eye")
        draw_boundary(roi_img, noseCascade, 1.1, 4, colors["nose"], "Nose")
        draw_boundary(roi_img, mouthCascade, 1.1, 20, colors["mouth"], "Mouth")
    return img

#  Tkinter 
class FaceApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Face Detection App")
        self.video_capture = cv2.VideoCapture(0)
        self.canvas = Canvas(window, width=640, height=480)
        self.canvas.pack()

        self.btn_snapshot = Button(window, text="Take Snapshot", command=self.take_snapshot)
        self.btn_snapshot.pack()
        self.btn_quit = Button(window, text="Quit", command=self.quit_app)
        self.btn_quit.pack()

        self.snapshots = []
        self.update_frame()

    def update_frame(self):
        ret, frame = self.video_capture.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))
            frame = detect(frame)
            # Convert for Tkinter
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2image))
            self.canvas.create_image(0, 0, image=self.photo, anchor='nw')
        self.window.after(10, self.update_frame)

    def take_snapshot(self):
        ret, frame = self.video_capture.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))
            frame = detect(frame)
            filename = f"snapshot_{len(self.snapshots)+1}.jpg"
            cv2.imwrite(filename, frame)
            self.snapshots.append(filename)
            print(f"Snapshot saved: {filename}")

    def quit_app(self):
        self.video_capture.release()
        self.window.destroy()




root = Tk()
app = FaceApp(root)
root.mainloop()
