import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from Custom_CNN_Model import build_custom_cnn
from Densenet121_Model import build_densenet121_model
import os
import pickle
import argparse

# Configuration
BATCH_SIZE = 16
EPOCHS = 50
FINE_TUNE_EPOCHS = 10
IMG_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 1)  # Custom CNN
INPUT_SHAPE_3CH = (224, 224, 3)  # DenseNet
DATA_DIR = "processed_dataset"

# Data generators
train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)

def create_generators(color_mode):
    return (
        train_gen.flow_from_directory(
            os.path.join(DATA_DIR, "train"),
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            color_mode=color_mode
        ),
        val_gen.flow_from_directory(
            os.path.join(DATA_DIR, "val"),
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            color_mode=color_mode
        )
    )

# Callbacks
callbacks_cnn = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint("best_model_cnn.h5", save_best_only=True)
]

callbacks_densenet = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint("best_model_densenet.h5", save_best_only=True)
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models")
    parser.add_argument('--model', choices=['cnn', 'densenet'], required=True, help="Model to train")
    args = parser.parse_args()

    if args.model == "cnn":
        from Custom_CNN_Model import build_custom_cnn
        print("\nTraining Custom CNN...")
        train_cnn, val_cnn = create_generators(color_mode='grayscale')
        model_cnn = build_custom_cnn(input_shape=INPUT_SHAPE)
        history_cnn = model_cnn.fit(
            train_cnn,
            validation_data=val_cnn,
            epochs=EPOCHS,
            callbacks=callbacks_cnn
        )
        with open("history_cnn.pkl", "wb") as f:
            pickle.dump(history_cnn.history, f)

    elif args.model == "densenet":
        print("\nTraining DenseNet121 (frozen base)...")
        train_densenet, val_densenet = create_generators(color_mode='rgb')
        model_densenet, base_model = build_densenet121_model(input_shape=INPUT_SHAPE_3CH)

        # Compile before initial training (Phase 1)
        model_densenet.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        history_densenet = model_densenet.fit(
            train_densenet,
            validation_data=val_densenet,
            epochs=EPOCHS,
            callbacks=callbacks_densenet
        )

        print("\nFine-tuning DenseNet121 (top 50 layers)...")
        base_model.trainable = True
        for layer in base_model.layers[:-50]:
            layer.trainable = False

        # Re-compile with lower LR for fine-tuning (Phase 2)
        model_densenet.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        fine_tune_history = model_densenet.fit(
            train_densenet,
            validation_data=val_densenet,
            epochs=FINE_TUNE_EPOCHS,
            callbacks=callbacks_densenet
        )

        # Combine training histories
        history_densenet.history.update(fine_tune_history.history)
        with open("history_densenet.pkl", "wb") as f:
            pickle.dump(history_densenet.history, f)
