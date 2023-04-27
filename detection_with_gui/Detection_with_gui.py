import tkinter as tk
from tkinter import filedialog
# from PIL import ImageTk,Image
from ultralytics import YOLO # pip install ultralytics=8.0.79   very important
import cv2 # pip install opencv-python=4.5.4.60
import cvzone # pip install cvzone=1.5.6
import math
import uuid

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()



    def create_widgets(self):
        # Set frame title
        self.master.title("My Application")
        # Set frame geometry

        self.master.configure(background="white")
        self.master.geometry("800x600")

        self.select_file_button = tk.Button(self, text="Select video file", command=self.select_file)
        self.select_file_button.configure(bg="blue", fg="white", font=("Arial", 12))
        self.select_file_button.grid(row=0, column=0, padx=8, pady=8)

        self.process_button = tk.Button(self, text="Process video", command=self.process_video, state="disabled")
        self.process_button.configure(bg="blue", fg="white", font=("Arial", 12))
        self.process_button.grid(row=0, column=1, padx=8, pady=8)

        self.quit = tk.Button(self, text="Quit", fg="red", command=self.master.destroy)
        self.quit.configure(bg="gray", font=("Arial", 12))
        self.quit.grid(row=1, column=0, columnspan=2, padx=8, pady=8)



    def select_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=(("Video files", "*.mp4"), ("All files", "*.*")))
        if self.file_path:
            self.process_button.config(state="normal")

    def process_video(self):
        cap = cv2.VideoCapture(self.file_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        # Generate a random UUID as the filename
        out_file = f"output_{uuid.uuid4()}.mp4"
        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_file, fourcc, fps, (width, height))
        model = YOLO("../Yolo-Weights/yolov8n.pt")
        with open("../Yolo-Weights/coco.names", "r") as f:
            classNames = f.read().splitlines()

        cv2.namedWindow("Processed video", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Processed video",640,480)

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
                    cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

            out.write(img)  # Write the frame to the output video
            cv2.imshow("Processed video", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # out.write(img)  # Write the frame to the output video
            # # Display the frame on the canvas
            # self.canvas.delete("all")
            # photo = ImageTk.PhotoImage(image=Image.fromarray(img))
            # self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            # self.canvas.image = photo  # Keep a reference to the image to avoid garbage collection
            # self.update_idletasks()  # Update the GUI
            #
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        # Release everything
        cap.release()
        out.release()
        cv2.destroyAllWindows()

root = tk.Tk()
app = Application(master=root)
app.mainloop()
