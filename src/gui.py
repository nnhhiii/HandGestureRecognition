import os
import tkinter as tk
from tkinter import filedialog

import cv2
from PIL import Image, ImageTk
from preprocess import preprocess_image
from predict import predict_gesture, gesture_history


def start_gui(model):
    root = tk.Tk()
    root.title("Nhận dạng cử chỉ tay")

    def upload_media():
        file_path = filedialog.askopenfilename(filetypes=[("Image and Video files", "*.jpg *.jpeg *.png *.mp4 *.avi")])
        if not file_path:
            return

        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if ext in [".jpg", ".jpeg", ".png"]:
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
            predicted_class, confidence = predict_gesture(model, preprocessed_img)

            # Kiểm tra độ chính xác và hiển thị kết quả dự đoán
            if confidence > 0.5:
                result_label.config(text=f"Cử chỉ được nhận diện: {predicted_class} với độ chính xác {confidence:.2f}")
                gesture_history.append(predicted_class)
                history_label.config(text="Lịch sử nhận diệnn: " + ", ".join(gesture_history))
            else:
                result_label.config(text="Không nhận diện được cử chỉ. Vui lòng thực hiện lại.")
        elif ext in [".mp4", ".avi"]:
            cap = cv2.VideoCapture(file_path)
            frame_skip = 10  # Chỉ nhận diện 1 khung hình mỗi 10 khung hình
            prev_gesture = None  # Lưu cử chỉ nhận diện trước đó

            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_skip == 0:  # Chỉ xử lý khung hình mỗi 10 khung
                    preprocessed_frame = preprocess_image(frame)
                    predicted_class, confidence = predict_gesture(model, preprocessed_frame)

                    if confidence > 0.5 and predicted_class != prev_gesture:
                        result_label.config(
                            text=f"Cử chỉ được nhận diện: {predicted_class} với độ chính xác {confidence:.2f}"
                        )
                        # Cập nhật cử chỉ trước đó và thêm vào lịch sử
                        prev_gesture = predicted_class
                        gesture_history.append(predicted_class)
                        history_label.config(text="Lịch sử nhận diện: " + ", ".join(gesture_history))
                    else:
                        result_label.config(text="Không nhận diện được cử chỉ. Vui lòng thực hiện lại.")

                frame_count += 1
                cv2.imshow("Video Recognition", frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
    upload_btn = tk.Button(root, text="Chọn ảnh", command=upload_media)
    upload_btn.grid(row=0, column=0)

    result_label = tk.Label(root, text="")
    result_label.grid(row=0, column=1)

    history_label = tk.Label(root, text="", wraplength=300, justify="left")
    history_label.grid(row=2, column=0, columnspan=2)

    root.mainloop()
