import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

def load_model():
    # Tải mô hình VGG16 đã huấn luyện
    model = tf.keras.models.load_model('../models/vgg16_model.h5')  # Đường dẫn tới mô hình đã lưu
    return model

def predict_gesture(model, preprocessed_image):
    # Ds
    class_names = ["Lòng bàn tay", "L", "Năm đâm", "Ngón cái", "Ok"]

    # Dự đoán cử chỉ từ hình ảnh đã được tiền xử lý
    predictions = model.predict(preprocessed_image)

    # Lấy chỉ số của lớp có xác suất cao nhất
    predicted_class_index = tf.argmax(predictions, axis=1).numpy()[0]

    # Lấy tên lớp dựa trên chỉ số
    predicted_class_name = class_names[predicted_class_index]

    return predicted_class_name


def train_model(train_data_dir, validation_data_dir, epochs=10):
    # Sử dụng ImageDataGenerator để tạo các tập dữ liệu cho huấn luyện và xác thực
    train_datagen = ImageDataGenerator(rescale=1.0 / 255.0, rotation_range=20, width_shift_range=0.2,
                                       height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                       horizontal_flip=True, fill_mode='nearest')

    validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    #Convolution layer
    # Tạo mô hình VGG16 đã được huấn luyện trước, không bao gồm lớp đầu ra
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Đóng băng các lớp trong mô hình VGG16
    for layer in base_model.layers:
        layer.trainable = False

    # Thêm các lớp tùy chỉnh cho mô hình
    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(5, activation='softmax')(x)

    # Tạo mô hình hoàn chỉnh
    model = Model(inputs=base_model.input, outputs=predictions)

    # Biên dịch mô hình
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Huấn luyện mô hình
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(train_generator, epochs=epochs, validation_data=validation_generator, callbacks=[early_stopping])

    # Lưu mô hình
    model.save('../models/vgg16_model.h5')
