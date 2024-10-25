import cv2
import numpy as np

def preprocess_image(image):
    # Đổi kích thước ảnh về kích thước mong muốn (224x224 cho VGG16)
    image = cv2.resize(image, (224, 224))
    # Chuyển đổi ảnh thành dạng mảng NumPy và chuẩn hóa
    image = np.array(image) / 255.0  # Chuẩn hóa giá trị pixel từ 0-255 về 0-1
    # Thêm chiều cho mảng để phù hợp với định dạng đầu vào của VGG16
    image = np.expand_dims(image, axis=0)  # Thêm chiều batch
    return image
