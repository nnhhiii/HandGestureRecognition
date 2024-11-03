from model import load_train_model
from gui import start_gui
from src.model import train_model
import os


def main():
    # Tải mô hình VGG16 đã huấn luyện
    model = load_train_model()

    # Bắt đầu giao diện người dùng
    start_gui(model)


if __name__ == "__main__":
    model_path = '../models/vgg16_model.keras'

    # Kiểm tra xem mô hình đã được huấn luyện và lưu trữ chưa
    if not os.path.exists(model_path):
        train_data_dir = "../data/train"  # Thư mục chứa dữ liệu huấn luyện
        validation_data_dir = "../data/validation"  # Thư mục chứa dữ liệu xác thực
        train_model(train_data_dir, validation_data_dir, epochs=10)
    else:
        print(f"Mô hình đã tồn tại tại {model_path}. Bỏ qua bước huấn luyện.")

    main()
