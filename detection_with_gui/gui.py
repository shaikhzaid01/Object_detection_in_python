







'''
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
import cv2
import cvzone
import math
import uuid
from PIL import Image, ImageTk

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.select_file_button = tk.Button(self, text="Select video file", command=self.select_file)
        self.select_file_button.grid(row=0, column=0, padx=5, pady=5)
        self.process_button = tk.Button(self, text="Process video", command=self.process_video, state="disabled")
        self.process_button.grid(row=0, column=1, padx=5, pady=5)
        self.quit = tk.Button(self, text="Quit", fg="red", command=self.master.destroy)
        self.quit.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
        self.canvas = tk.Canvas(self, width=640, height=480)
        self.canvas.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

    def select_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=(("Video files", "*.mp4"), ("All files", "*.*")))
        if self.file_path:
            self.process_button.config(state="normal")

    def process_video(self):
        cap = cv2.VideoCapture(self.file_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out_file = f"output_{uuid.uuid4()}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_file, fourcc, fps, (width, height))
        model = YOLO("../Yolo-Weights/yolov8n.pt")
        with open("../Yolo-Weights/coco.names", "r") as f:
            classNames = f.read().splitlines()

        # Create a label to display the video stream
        self.video_label = tk.Label(self.canvas)
        self.video_label.pack()

        while True:
            success, img = cap.read()
            if not success:
                break
            results = model(img, stream=True)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    cvzone.cornerRect(img, (x1, y1, w, h))
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1,
                                       thickness=1)

            out.write(img)  # Write the frame to the output video

            # Display the frame in the OpenCV window
            # cv2.imshow("Processed video", img)

            # Convert the frame to a format that can be displayed in the Tkinter window
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tk = ImageTk.PhotoImage(Image.fromarray(frame))

            # Update the image in the Tkinter Label
            self.video_label.config(image=img_tk)
            self.video_label.image = img_tk

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release everything
        cap.release()
        out.release()
        cv2.destroyAllWindows()

root = tk.Tk()
app = Application(master=root)
app.mainloop()
'''