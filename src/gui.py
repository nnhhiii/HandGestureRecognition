import os
import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
from preprocess import preprocess_image
from predict import predict_gesture, gesture_history

cap = None  # Khai báo cap toàn cục hoặc ngoài các hàm

def start_gui(model):
    root = tk.Tk()
    root.title("Nhận dạng cử chỉ tay")

    result_label = tk.Label(root, text="")
    result_label.grid(row=2, column=2)

    history_label = tk.Label(root, text="", wraplength=300, justify="left")
    history_label.grid(row=5, column=2, columnspan=2)

    recognize_btn = tk.Button(root, text="Nhận diện cử chỉ")
    recognize_btn.grid_forget()

    capture_image_btn = tk.Button(root, text="Chụp ảnh")
    start_video_btn = tk.Button(root, text="Bắt đầu quay video")
    stop_video_btn = tk.Button(root, text="Dừng quay video")
    back_to_camera_btn = tk.Button(root, text="Quay lại camera")

    camera_label = tk.Label(root)
    camera_label.grid(row=1, column=1)

    image_label = tk.Label(root)
    image_label.grid(row=1, column=2)

    def clear_display_area():
        """Clear the display area and reset buttons specific to each mode."""
        result_label.config(text="")
        history_label.config(text="")
        if hasattr(camera_label, 'image'):
            camera_label.image = None
        camera_label.config(image="")
        if hasattr(image_label, 'image'):
            image_label.image = None
        image_label.config(image="")
        recognize_btn.grid_forget()
        capture_image_btn.grid_forget()
        start_video_btn.grid_forget()
        stop_video_btn.grid_forget()
        back_to_camera_btn.grid_forget()
        gesture_history.clear()

    def show_result(predicted_class, confidence):
        if confidence > 0.5:
            result_label.config(text=f"Cử chỉ được nhận diện: {predicted_class} với độ chính xác {confidence:.2f}")
            gesture_history.append(predicted_class)
            history_label.config(text="Lịch sử nhận diện: " + ", ".join(gesture_history))
        else:
            result_label.config(text="Không nhận diện được cử chỉ. Vui lòng thực hiện lại.")

    def upload_media():
        global cap  # Sử dụng cap toàn cục
        if cap is not None:
            cap.release()  # Đảm bảo đóng camera nếu đang mở
        clear_display_area()
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

            # Xóa ảnh cũ nếu có trước khi hiển thị ảnh mới
            image_label.grid_forget()
            image_label.config(image=img_tk)
            image_label.image = img_tk
            image_label.grid(row=1, column=2)

            def recognize_gesture():
                img_cv = cv2.imread(file_path)
                preprocessed_img = preprocess_image(img_cv)
                predicted_class, confidence = predict_gesture(model, preprocessed_img)
                show_result(predicted_class, confidence)

            recognize_btn.config(command=recognize_gesture)
            recognize_btn.grid(row=3, column=2)


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

                        if confidence > 0.5 and predicted_class != prev_gesture:
                            show_result(predicted_class, confidence)
                            prev_gesture = predicted_class
                    frame_count += 1
                    cv2.imshow("Video Recognition", frame)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                cap.release()
                cv2.destroyAllWindows()

            recognize_btn.config(command=recognize_video)
            recognize_btn.grid(row=3, column=1)

    def open_camera():
        global cap  # Sử dụng cap toàn cục
        clear_display_area()
        cap = cv2.VideoCapture(0)  # Mở camera trực tiếp
        is_recording = False

        def update_frame():
            ret, frame = cap.read()
            if ret:
                frame_resized = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(frame_resized)
                img_tk = ImageTk.PhotoImage(img_pil)
                camera_label.config(image=img_tk)
                camera_label.image = img_tk
                if not is_recording:
                    root.after(10, update_frame)

        def capture_image():
            result_label.config(text="")
            history_label.config(text="")

            ret, frame = cap.read()
            if ret:
                image_path = "captured_image.jpg"
                cv2.imwrite(image_path, frame)

                # Hiển thị ảnh chụp trên giao diện
                img = Image.open(image_path)
                img = img.resize((300, 300))
                img_tk = ImageTk.PhotoImage(img)
                image_label.config(image=img_tk)
                image_label.image = img_tk

                def recognize_gesture():
                    img_cv = cv2.imread(image_path)  # Đọc ảnh đã chụp
                    preprocessed_img = preprocess_image(img_cv)  # Tiền xử lý ảnh
                    predicted_class, confidence = predict_gesture(model, preprocessed_img)  # Nhận diện cử chỉ
                    show_result(predicted_class, confidence)  # Hiển thị kết quả

                recognize_btn.config(command=recognize_gesture)

        def start_video_recording():
            result_label.config(text="")
            history_label.config(text="")

            nonlocal is_recording
            is_recording = True
            result_label.config(text="Đang quay video...")  # Hiển thị thông báo quay video

            # Cấu hình VideoWriter để lưu video
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (640, 480))

            def update_frame1():
                ret, frame = cap.read()
                if ret:
                    # Hiển thị frame lên giao diện
                    frame_resized = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(frame_resized)
                    img_tk = ImageTk.PhotoImage(img_pil)

                    camera_label.config(image=img_tk)
                    camera_label.image = img_tk

                    # Ghi frame vào video
                    if is_recording:
                        out.write(frame)

                    # Cập nhật frame mỗi 10ms
                    root.after(10, update_frame1)

            def record_video():
                nonlocal is_recording
                ret, frame = cap.read()
                if ret:
                    out.write(frame)  # Ghi frame vào video
                    if is_recording:
                        root.after(10, record_video)  # Tiếp tục quay video

            # Bắt đầu quay video và cập nhật hình ảnh từ camera
            update_frame1()

        def stop_video_recording():
            nonlocal is_recording
            is_recording = False
            result_label.config(text="Đã dừng quay video.")  # Hiển thị thông báo dừng quay video

            # Đọc lại video đã quay để hiển thị
            video_path = "output_video.avi"
            cap_video = cv2.VideoCapture(video_path)  # Mở video đã quay

            def display_video():
                ret, frame = cap_video.read()
                if ret:
                    # Chuyển đổi frame từ BGR sang RGB
                    frame_resized = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(frame_resized)
                    img_tk = ImageTk.PhotoImage(img_pil)

                    # Hiển thị frame video trên giao diện
                    image_label.config(image=img_tk)
                    image_label.image = img_tk

                    # Tiếp tục hiển thị video sau mỗi 10ms
                    root.after(30, display_video)

            # Bắt đầu hiển thị video
            display_video()
            update_frame()

            def recognize_gesture():
                cap_video = cv2.VideoCapture(video_path)  # Mở video
                frame_count = 0
                prev_gesture = None  # Biến lưu cử chỉ trước đó để tránh nhận diện lại cùng cử chỉ

                def process_frame():
                    nonlocal frame_count, prev_gesture
                    ret, frame = cap_video.read()
                    if not ret:
                        cap_video.release()  # Đóng video nếu không còn frame
                        return

                    # Chỉ nhận diện trên những frame nhất định
                    if frame_count % 10 == 0:
                        preprocessed_frame = preprocess_image(frame)  # Tiền xử lý frame
                        predicted_class, confidence = predict_gesture(model, preprocessed_frame)  # Nhận diện cử chỉ

                        # Hiển thị kết quả nếu độ tin cậy đủ cao và cử chỉ mới
                        if confidence > 0.5 and predicted_class != prev_gesture:
                            show_result(predicted_class, confidence)
                            prev_gesture = predicted_class

                    frame_count += 1

                    # Chuyển đổi frame từ OpenCV thành hình ảnh Tkinter để hiển thị
                    frame_resized = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(frame_resized)
                    img_tk = ImageTk.PhotoImage(img_pil)

                    # Cập nhật hình ảnh lên giao diện
                    image_label.config(image=img_tk)
                    image_label.image = img_tk  # Giữ tham chiếu tới ảnh để tránh bị thu hồi

                    # Lên lịch gọi lại hàm sau mỗi 30ms để tiếp tục hiển thị video
                    root.after(10, process_frame)

                # Bắt đầu vòng lặp xử lý frame
                process_frame()

            # Gán hàm nhận diện cử chỉ cho nút "Nhận diện cử chỉ"
            recognize_btn.config(command=recognize_gesture)

        update_frame()  # Bắt đầu cập nhật hình ảnh từ camera

        capture_image_btn.config(command=capture_image)
        capture_image_btn.grid(row=3, column=1)
        start_video_btn.config(command=start_video_recording)
        start_video_btn.grid(row=4, column=1)
        stop_video_btn.config(command=stop_video_recording)
        stop_video_btn.grid(row=5, column=1)

        recognize_btn.grid(row=3, column=2)

    def start_live_recognition():
        global cap  # Sử dụng cap toàn cục
        clear_display_area()  # Xóa giao diện trước khi bắt đầu nhận diện
        cap = cv2.VideoCapture(0)  # Mở camera trực tiếp

        def update_frame():
            ret, frame = cap.read()
            if ret:
                frame_resized = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(frame_resized)
                img_tk = ImageTk.PhotoImage(img_pil)
                camera_label.config(image=img_tk)
                camera_label.image = img_tk
                root.after(10, update_frame)  # Tiếp tục cập nhật frame

        def live_recognition():
            ret, frame = cap.read()
            if ret:
                preprocessed_frame = preprocess_image(frame)  # Tiền xử lý frame
                predicted_class, confidence = predict_gesture(model, preprocessed_frame)  # Nhận diện cử chỉ
                show_result(predicted_class, confidence)  # Hiển thị kết quả nhận diện
                root.after(500, live_recognition)  # Lặp lại nhận diện sau mỗi 500ms (có thể điều chỉnh)

        update_frame()  # Bắt đầu cập nhật frame camera
        live_recognition()  # Bắt đầu nhận diện cử chỉ

    upload_btn = tk.Button(root, text="Chọn ảnh/video", command=upload_media)
    upload_btn.grid(row=0, column=0)

    camera_btn = tk.Button(root, text="Chụp ảnh/quay video", command=open_camera)
    camera_btn.grid(row=1, column=0)

    start_live_btn = tk.Button(root, text="Trực tiếp", command=start_live_recognition)
    start_live_btn.grid(row=2, column=0)

    root.mainloop()
