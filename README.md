# gesture-volume-control
You can control the volume with your hands
import tkinter as tk
from tkinter import ttk
import threading
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from PIL import Image, ImageTk
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

class VolumeControlApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Volume Control")
        self.root.geometry("900x700")

        # Load TensorFlow model
        self.model = tf.keras.models.load_model("gesture_volume_model.h5")

        # Setup audio control
        self.devices = AudioUtilities.GetSpeakers()
        self.interface = self.devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(self.interface, POINTER(IAudioEndpointVolume))
        self.min_vol, self.max_vol = self.volume.GetVolumeRange()[:2]

        # Mediapipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_draw = mp.solutions.drawing_utils

        # UI Elements
        self.label = tk.Label(root)
        self.label.pack()

        self.volume_label = ttk.Label(root, text="Predicted Volume: 0.00", font=("Helvetica", 14))
        self.volume_label.pack(pady=10)

        self.start_btn = ttk.Button(root, text="Start", command=self.start_camera)
        self.start_btn.pack(pady=5)

        self.stop_btn = ttk.Button(root, text="Stop", command=self.stop_camera)
        self.stop_btn.pack(pady=5)

        self.running = False
        self.cap = None

    def start_camera(self):
        if not self.running:
            self.running = True
            self.cap = cv2.VideoCapture(0)
            threading.Thread(target=self.update_frame).start()

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.label.config(image='')

    def update_frame(self):
        while self.running and self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                lm_list = [(lm.x, lm.y) for lm in landmarks.landmark]

                if len(lm_list) == 21:
                    input_data = np.array([[pt[0] for pt in lm_list] + [pt[1] for pt in lm_list]])
                    pred_volume = self.model.predict(input_data, verbose=0)[0][0]
                    pred_volume = np.clip(pred_volume, 0, 1)
                    sys_volume = np.interp(pred_volume, [0, 1], [self.min_vol, self.max_vol])
                    self.volume.SetMasterVolumeLevel(sys_volume, None)
                    self.volume_label.config(text=f"Predicted Volume: {pred_volume:.2f}")

                self.mp_draw.draw_landmarks(frame, landmarks, self.mp_hands.HAND_CONNECTIONS)

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)

        if self.cap:
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = VolumeControlApp(root)
    root.mainloop()
