import os
from tkinter import filedialog
import cv2
from PIL import Image
from preprocess import preprocess_image
from predict import predict_gesture, gesture_history
import customtkinter as ctk

cap = None
after_event_id_recognition = None

def start_gui(model):
    root = ctk.CTk()
    root.configure(fg_color="white")

    root.title("Nhận dạng cử chỉ tay")

    gesture =  ctk.CTkLabel(root, text="", font=("Verdana", 18, "bold"), text_color="#58CC01")
    gesture.grid(row=1, column=4, columnspan=3, sticky="e", padx=10)

    result_label = ctk.CTkLabel(root, text="")
    result_label.grid(row=8, column=4, columnspan=3)

    title = ctk.CTkLabel(root, text="NHẬN DẠNG CỬ CHỈ TAY", font=("Verdana", 23, "bold"), text_color="#0E3469")
    title.grid(row=0, column=1, columnspan=6, pady=15)

    label = ctk.CTkLabel(root, text="")
    label.grid(row=8, column=2)

    history_label = ctk.CTkLabel(root, text="", wraplength=300, justify="left")
    history_label.grid(row=10, column=4, columnspan=3)

    recognize_btn = ctk.CTkButton(root, text="Nhận diện", text_color="white", corner_radius=50, width=100, height=50, fg_color="#FF86D0", hover_color="#CC6BA7")
    recognize_btn.grid_forget()

    capture_image_btn = ctk.CTkButton(root, text="Chụp", text_color="white", corner_radius=15, fg_color="#04CD9C", width=50, height=60, hover_color="#08A47D")
    start_video_btn = ctk.CTkButton(root, text="Quay", text_color="white", corner_radius=15, fg_color="#FF9601", width=50, height=60, hover_color="#CD7807")
    stop_video_btn = ctk.CTkButton(root, text="Dừng", text_color="white", corner_radius=15, fg_color="#FF4C4A", width=50, height=60, hover_color="#CD3D3C")

    camera_label = ctk.CTkLabel(root, text="")
    camera_label.grid(row=2, column=1, columnspan=3, rowspan=6)

    image_label = ctk.CTkLabel(root, text="")


    def clear_display_area():
        global after_event_id_recognition, cap
        gesture.configure(text="")
        gesture.grid(row=1, column=4, columnspan=3, sticky="e", padx=10)
        result_label.configure(text="")
        result_label.grid(row=8, column=4, columnspan=3)
        label.configure(text="")
        history_label.configure(text="")
        history_label.grid(row=10, column=4, columnspan=3)
        # Xóa ảnh khỏi label
        image_label.configure(image=None)
        image_label.image = None
        image_label.grid_forget()
        camera_label.configure(image=None)
        camera_label.image = None
        camera_label.grid_forget()

        recognize_btn.grid_forget()
        capture_image_btn.grid_forget()
        start_video_btn.grid_forget()
        stop_video_btn.grid_forget()
        gesture_history.clear()
        if after_event_id_recognition is not None:
            root.after_cancel(after_event_id_recognition)
            after_event_id_recognition = None

    def show_result(predicted_class, confidence):
        if confidence > 0.5:
            gesture.configure(text=predicted_class)
            result_label.configure(text=f"Cử chỉ được nhận diện: {predicted_class}     Độ chính xác: {confidence:.2f}")
            gesture_history.append(predicted_class)
            history_label.configure(text="Lịch sử nhận diện: " + ", ".join(gesture_history))
        else:
            result_label.configure(text="Không nhận diện được cử chỉ. Vui lòng thực hiện lại.")

    def upload_media():
        global cap
        clear_display_area()
        if cap is not None:
            cap.release()
        file_path = filedialog.askopenfilename(filetypes=[("Image and Video files", "*.jpg *.jpeg *.png *.mp4 *.avi")])
        if not file_path:
            return

        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if ext in [".jpg", ".jpeg", ".png"]:
            # Hiển thị ảnh
            img = Image.open(file_path)
            img_tk = ctk.CTkImage(size=(400, 300), light_image=img)

            # Xóa ảnh cũ nếu có trước khi hiển thị ảnh mới
            image_label.grid_forget()
            image_label.configure(image=img_tk)
            image_label.image = img_tk
            image_label.grid(row=2, column=4, columnspan=3, rowspan=6, pady=15, padx=15)

            def recognize_gesture():
                img_cv = cv2.imread(file_path)
                preprocessed_img = preprocess_image(img_cv)
                predicted_class, confidence = predict_gesture(model, preprocessed_img)
                show_result(predicted_class, confidence)

            recognize_btn.configure(command=recognize_gesture)
            recognize_btn.grid(row=9, column=5)
        elif ext in [".mp4", ".avi"]:
            cap = cv2.VideoCapture(file_path)
            frame_skip = 10
            prev_gesture = None  # Initialize prev_gesture here
            def recognize_video():
                nonlocal prev_gesture
                frame_count = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_count % frame_skip == 0:
                        preprocessed_frame = preprocess_image(frame)
                        predicted_class, confidence = predict_gesture(model, preprocessed_frame)

                        if predicted_class != prev_gesture:
                            show_result(predicted_class, confidence)
                            prev_gesture = predicted_class
                    frame_count += 1
                    cv2.imshow("Video Recognition", frame)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                cap.release()
                cv2.destroyAllWindows()
            recognize_btn.configure(command=recognize_video)
            recognize_btn.grid(row=9, column=5)

    def open_camera():
        clear_display_area()
        global cap
        cap = cv2.VideoCapture(0)  # Mở camera
        is_recording = False
        out = None

        def display_frame(frame, target_label, size=(600, 400)):
            """Hiển thị frame lên giao diện Tkinter."""
            frame_resized = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_resized)
            img_tk = ctk.CTkImage(size=size, light_image=img_pil)
            target_label.configure(image=img_tk)
            target_label.image = img_tk  # Lưu tham chiếu để tránh bị thu hồi

        def update_frame():
            """Cập nhật hình ảnh từ camera."""
            ret, frame = cap.read()
            if ret:
                display_frame(frame, camera_label)
            if not is_recording:
                root.after(10, update_frame)

        def capture_image():
            """Chụp ảnh từ camera."""
            result_label.configure(text="")
            history_label.configure(text="")

            ret, frame = cap.read()
            if ret:
                image_path = "captured_image.jpg"
                cv2.imwrite(image_path, frame)  # Lưu ảnh
                display_frame(frame, image_label, size=(400, 300))  # Hiển thị ảnh chụp

                def recognize_gesture():
                    preprocessed_img = preprocess_image(frame)  # Tiền xử lý
                    predicted_class, confidence = predict_gesture(model, preprocessed_img)  # Nhận diện
                    show_result(predicted_class, confidence)  # Hiển thị kết quả

                recognize_btn.configure(command=recognize_gesture)

        def start_video_recording():
            """Bắt đầu quay video."""
            nonlocal is_recording, out
            is_recording = True
            result_label.configure(text="")
            history_label.configure(text="")
            label.configure(text="Đang quay video...")

            # Cấu hình VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('output_video.avi', fourcc, 24.0, (640, 480))

            def record_video():
                ret, frame = cap.read()
                if ret and is_recording:
                    out.write(frame)  # Ghi frame vào video
                    display_frame(frame, camera_label)  # Hiển thị frame
                    root.after(10, record_video)

            record_video()

        def stop_video_recording():
            """Dừng quay video và cung cấp tùy chọn nhận diện từ video."""
            nonlocal is_recording, out
            is_recording = False
            if out:
                out.release()  # Giải phóng VideoWriter
                out = None
            label.configure(text="Đã dừng quay video.")
            video_path = "output_video.avi"
            display_recorded_video(video_path)  # Hiển thị video đã quay

            def recognize_gesture_from_video():
                """Nhận diện cử chỉ từ video đã quay."""
                cap_video = cv2.VideoCapture(video_path)
                frame_count = 0
                prev_gesture = None

                def process_frame():
                    nonlocal frame_count, prev_gesture
                    ret, frame = cap_video.read()
                    if not ret:
                        cap_video.release()  # Đóng video khi kết thúc
                        return

                    # Chỉ nhận diện trên những frame nhất định
                    if frame_count % 10 == 0:  # Mỗi 10 frame xử lý 1 lần
                        preprocessed_frame = preprocess_image(frame)  # Tiền xử lý frame
                        predicted_class, confidence = predict_gesture(model, preprocessed_frame)  # Nhận diện cử chỉ

                        # Hiển thị kết quả nếu độ tin cậy cao và khác cử chỉ trước
                        if confidence > 0.5 and predicted_class != prev_gesture:
                            show_result(predicted_class, confidence)
                            prev_gesture = predicted_class

                    frame_count += 1

                    # Hiển thị frame trên giao diện
                    display_frame(frame, image_label, size=(300, 300))

                    # Lên lịch xử lý frame tiếp theo
                    root.after(30, process_frame)

                process_frame()

            # Gán hàm nhận diện video vào nút "Nhận diện cử chỉ"
            recognize_btn.configure(command=recognize_gesture_from_video)

        def display_recorded_video(video_path):
            """Hiển thị video đã quay."""
            cap_video = cv2.VideoCapture(video_path)

            def process_video():
                ret, frame = cap_video.read()
                if ret:
                    display_frame(frame, image_label, size=(300, 300))  # Hiển thị frame
                    root.after(30, process_video)
                else:
                    cap_video.release()  # Đóng video khi kết thúc

            process_video()

        update_frame()  # Bắt đầu hiển thị từ camera
        image_label.grid(row=2, column=4, columnspan=3, rowspan=6, pady=15, padx=15)
        camera_label.grid(row=2, column=1, columnspan=3, rowspan=6)

        # Gán chức năng cho các nút
        capture_image_btn.configure(command=capture_image)
        capture_image_btn.grid(row=9, column=1)

        start_video_btn.configure(command=start_video_recording)
        start_video_btn.grid(row=9, column=2)

        stop_video_btn.configure(command=stop_video_recording)
        stop_video_btn.grid(row=9, column=3)

        recognize_btn.grid(row=9, column=5)

    def start_live_recognition():
        clear_display_area()
        global cap  # Sử dụng cap toàn cục
        cap = cv2.VideoCapture(0)  # Mở camera trực tiếp

        def update_frame():
            ret, frame = cap.read()
            if ret:
                frame_resized = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(frame_resized)
                img_tk = ctk.CTkImage(size=(600, 400), light_image=img_pil)
                camera_label.configure(image=img_tk)
                camera_label.image = img_tk
                camera_label.grid(row=2, column=1, columnspan=3, rowspan=6, pady=15, padx=15)
                root.after(10, update_frame)  # Tiếp tục cập nhật frame

        def live_recognition():
            global after_event_id_recognition
            ret, frame = cap.read()
            if ret:
                preprocessed_frame = preprocess_image(frame)
                predicted_class, confidence = predict_gesture(model, preprocessed_frame)
                result_label.grid(row=8, column=1, columnspan=3)
                history_label.grid(row=10, column=1, columnspan=3)
                gesture.grid(row=1, column=1, columnspan=3, sticky="e", padx=10)
                show_result(predicted_class, confidence)
                after_event_id_recognition = root.after(500, live_recognition)   # Lặp lại nhận diện sau mỗi 500ms

        update_frame()  # Bắt đầu cập nhật frame camera
        live_recognition()  # Bắt đầu nhận diện cử chỉ


    upload_btn = ctk.CTkButton(root, text="Chọn ảnh/video", font=("Arial", 15, "bold"), text_color="white", corner_radius=10, width=180, height=60,
                               fg_color="#4EA3E2", hover_color="#1DC1FF", command=upload_media)
    upload_btn.grid(row=2, column=0, rowspan=2, padx=10, pady=10)

    camera_btn = ctk.CTkButton(root, text="Chụp ảnh/video", font=("Arial", 15, "bold"), text_color="white", corner_radius=10, width=180, height=60,
                               fg_color="#4EA3E2", hover_color="#1DC1FF",command=open_camera)
    camera_btn.grid(row=4, column=0, rowspan=2, padx=10, pady=10)

    start_live_btn = ctk.CTkButton(root, text="Quay trực tiếp",  font=("Arial", 15, "bold"), text_color="white", corner_radius=10, width=180, height=60,
                                   fg_color="#4EA3E2", hover_color="#1DC1FF",
                                   command=start_live_recognition)
    start_live_btn.grid(row=6, column=0, rowspan=2, padx=10, pady=10)

    root.mainloop()
