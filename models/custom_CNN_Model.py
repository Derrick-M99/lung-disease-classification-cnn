import tensorflow as tf
from tensorflow.keras import layers, models, Input

def build_custom_cnn(input_shape=(224, 224, 1), num_classes=4):
    inputs = Input(shape=input_shape)

    # Conv Block 1
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Conv Block 2
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Conv Block 3
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # ➕ Additional Conv Block (better for Grad-CAM)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name="gradcam_target")(x)
    x = layers.BatchNormalization()(x)
    # Don't pool here to preserve feature map spatial resolution for Grad-CAM

    # GAP
    x = layers.GlobalAveragePooling2D()(x)

    # Dense layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

if __name__ == "__main__":
    model = build_custom_cnn()
    model.summary()  

