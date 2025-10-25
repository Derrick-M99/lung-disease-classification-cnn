# results.py - Cleaned and Commented Version

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from Custom_CNN_Model import build_custom_cnn
from Densenet121_Model import build_densenet121_model

# Configuration Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
TEST_DIR = "processed_dataset/test"

MODELS = {
    "Custom CNN": "best_model_cnn.h5",
    "DenseNet121": "best_model_densenet.h5"
}

HISTORIES = {
    "Custom CNN": "history_cnn.pkl",
    "DenseNet121": "history_densenet.pkl"
}

# Data Loader
def get_generator(directory, color_mode='rgb'):
    datagen = ImageDataGenerator(rescale=1./255)
    return datagen.flow_from_directory(
        directory,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        color_mode=color_mode
    )

def get_image_from_generator(dataset, image_index):
    batch_size = dataset.batch_size
    batch_index = image_index // batch_size
    within_batch_index = image_index % batch_size
    batch = dataset[batch_index]
    return batch[0][within_batch_index:within_batch_index+1]

# Model Evaluation
def evaluate_model(model, dataset, class_names):
    preds = model.predict(dataset)
    y_pred = np.argmax(preds, axis=1)
    y_true = dataset.classes

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    print(classification_report(y_true, y_pred, target_names=class_names))
    return y_true, y_pred, preds, cm, report

# Visualizations Utilities
def plot_history(histories):
    for name, path in histories.items():
        with open(path, 'rb') as f:
            hist = pickle.load(f)

        plt.figure(figsize=(12, 4))

        # Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(hist['accuracy'], label='Train')
        plt.plot(hist['val_accuracy'], label='Val')
        plt.title(f"Accuracy - {name}")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy")
        plt.legend()

        # Loss
        plt.subplot(1, 2, 2)
        plt.plot(hist['loss'], label='Train')
        plt.plot(hist['val_loss'], label='Val')
        plt.title(f"Loss - {name}")
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.legend()

        plt.tight_layout()
        plt.show()

def plot_confusion_matrix(cm, classes, title):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.show()

def plot_pr_roc(y_true, preds, class_names, title):
    plt.figure(figsize=(12, 5))
    for i in range(len(class_names)):
        # ROC
        fpr, tpr, _ = roc_curve(y_true == i, preds[:, i])
        roc_auc = auc(fpr, tpr)
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.2f})")

        # Precision-Recall
        prec, rec, _ = precision_recall_curve(y_true == i, preds[:, i])
        plt.subplot(1, 2, 2)
        plt.plot(rec, prec, label=class_names[i])

    plt.subplot(1, 2, 1)
    plt.title(f"ROC Curve - {title}"); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title(f"Precision-Recall Curve - {title}"); plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_f1_bar(report, class_names, title):
    f1_scores = [report[name]['f1-score'] for name in class_names]
    plt.figure()
    sns.barplot(x=class_names, y=f1_scores)
    plt.title(f"F1 Score by Class - {title}")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

# Grad-CAM Implementation
def get_all_conv_layers(model):
    convs = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            convs.append(layer)
        elif hasattr(layer, 'layers'):
            convs.extend(get_all_conv_layers(layer))
    return convs

def plot_grad_cams(model, images, labels, class_names, model_name):
    if not model.inputs:
        # Build model if not built
        dummy = tf.convert_to_tensor(images[0], dtype=tf.float32)
        dummy = tf.reshape(dummy, (1, 224, 224, 1 if dummy.shape[-1] == 1 else 3))
        _ = model(dummy)

    conv_layers = get_all_conv_layers(model)
    if not conv_layers:
        print(f"Skipping Grad-CAM: No Conv2D layers found in {model_name}")
        return
    last_conv_layer = conv_layers[-1]

    grad_model = tf.keras.models.Model(inputs=model.input, outputs=[last_conv_layer.output, model.output])

    fig = plt.figure(figsize=(4 * len(images), 5))
    for i, (img, label) in enumerate(zip(images, labels)):
        img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            loss = predictions[:, label]

        grads = tape.gradient(loss, conv_outputs)[0]
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1)
        cam = np.maximum(cam, 0)
        cam = cam / tf.math.reduce_max(cam + tf.keras.backend.epsilon())
        heatmap = tf.image.resize(cam[..., tf.newaxis], (224, 224)).numpy()

        # Display
        ax = fig.add_subplot(1, len(images), i + 1)
        raw_img = np.squeeze(img)
        if raw_img.ndim == 3 and raw_img.shape[-1] == 1:
            raw_img = raw_img[:, :, 0]
        ax.imshow(raw_img, cmap='gray')
        ax.imshow(np.squeeze(heatmap), cmap='jet', alpha=0.5)
        ax.set_title(f"{model_name} - {class_names[label]}")
        ax.axis('off')

    plt.suptitle(f"Grad-CAM Visualizations - {model_name}", fontsize=14)
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(cmap='jet', norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
    fig.colorbar(sm, cax=cbar_ax, label='Activation Intensity')
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.show()


# Main Execution
if __name__ == "__main__":
    for name, model_path in MODELS.items():
        print(f"\n--- {name} ---")

        # Load correct model
        if name == "Custom CNN":
            model = build_custom_cnn(input_shape=(224, 224, 1))
            model.load_weights(model_path)
            model(tf.zeros((1, 224, 224, 1)))  # Ensure model is built
        else:
            model = load_model(model_path)

        test_ds = get_generator(TEST_DIR, color_mode='grayscale' if name == 'Custom CNN' else 'rgb')
        class_names = list(test_ds.class_indices.keys())

        # Run evaluation and visualization
        y_true, y_pred, preds, cm, report = evaluate_model(model, test_ds, class_names)
        plot_confusion_matrix(cm, class_names, f"Test Confusion Matrix - {name}")
        plot_f1_bar(report, class_names, name)
        plot_pr_roc(np.array(y_true), preds, class_names, name)

        try:
            imgs = []
            labels = []
            for i in range(len(class_names)):
                idx = np.where(np.array(y_true) == i)[0][0]
                img = get_image_from_generator(test_ds, idx)
                imgs.append(img)
                labels.append(i)
            plot_grad_cams(model, imgs, labels, class_names, name)
        except Exception as e:
            print(f"Skipping Grad-CAM for {name}: {e}")

    plot_history(HISTORIES)
