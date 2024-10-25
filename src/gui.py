import tkinter as tk
from tkinter import filedialog

import cv2
from PIL import Image, ImageTk
from preprocess import preprocess_image
from model import predict_gesture

def start_gui(model):
    root = tk.Tk()
    root.title("Gesture Recognition")

    def upload_image():
        # Mở hộp thoại để chọn ảnh
        file_path = filedialog.askopenfilename()
        if file_path:
            # Hiển thị ảnh
            img = Image.open(file_path)
            img = img.resize((300, 300))
            img_tk = ImageTk.PhotoImage(img)
            panel = tk.Label(root, image=img_tk)
            panel.image = img_tk
            panel.grid(row=1, column=0, columnspan=2)

            # Tiền xử lý ảnh và dự đoán cử chỉ
            img_cv = cv2.imread(file_path)
            preprocessed_img = preprocess_image(img_cv)
            predicted_class = predict_gesture(model, preprocessed_img)

            # Hiển thị kết quả dự đoán
            result_label.config(text=f"Predicted Gesture Class: {predicted_class}")

    upload_btn = tk.Button(root, text="Upload Image", command=upload_image)
    upload_btn.grid(row=0, column=0)

    result_label = tk.Label(root, text="")
    result_label.grid(row=0, column=1)

    root.mainloop()
