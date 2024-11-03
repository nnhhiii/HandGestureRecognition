import numpy as np
import tensorflow as tf
from keras.src.layers import BatchNormalization
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

def load_train_model():
    # Tải mô hình VGG16 đã huấn luyện
    model = tf.keras.models.load_model('../models/vgg16_model.keras')  # Đường dẫn tới mô hình đã lưu
    return model

def train_model(train_data_dir, validation_data_dir, epochs):
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
    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    predictions = Dense(5, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.summary()

    # Tạo mô hình hoàn chỉnh
    model = Model(inputs=base_model.input, outputs=predictions)

    # Biên dịch mô hình
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Huấn luyện mô hình
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(train_generator, epochs=epochs, validation_data=validation_generator, callbacks=[early_stopping])

    # Lưu mô hình
    model.save('../models/vgg16_model.keras')
