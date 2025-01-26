import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Input image size
shape_initializer = (28, 28)

# File directories
TRAINING_DIR = 'models_data/train'
TEST_DIR = 'models_data/test'

# Epochs and batch size
epochs = 10
BATCH_SIZE = 16

# Data collection using TensorFlow utilities
def dataCollection():
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAINING_DIR,
        labels='inferred',
        label_mode='categorical',
        image_size=shape_initializer,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset='training',
        seed=10
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAINING_DIR,
        labels='inferred',
        label_mode='categorical',
        image_size=shape_initializer,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset='validation',
        seed=10
    )

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels='inferred',
        label_mode='categorical',
        image_size=shape_initializer,
        batch_size=BATCH_SIZE
    )

    class_names = train_dataset.class_names

    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_dataset, validation_dataset, test_dataset, class_names

# Model creation
def create_model(num_of_classes):
    model = tf.keras.models.Sequential([
        Input(shape=(shape_initializer[0], shape_initializer[1], 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_of_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

# Plot training/validation accuracy and loss
def plot_model_performance(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    plt.show()

# Predict single image
def predict_single_image(model, img_path):
    img = tf.keras.utils.load_img(img_path, target_size=shape_initializer)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    class_labels = ['apple', 'banana', 'mixed', 'orange']

    print(f"Predicted class: {class_labels[predicted_class[0]]}")
    print("Softmax probability distribution:")
    for i, prob in enumerate(predictions[0]):
        print(f"{class_labels[i]}: {prob:.4f}")

# Evaluate model performance
def evaluate_model_performance(model, test_dataset, train_dataset):
    classes = list(train_dataset.class_names)

    Y_pred = model.predict(test_dataset)
    y_pred = np.argmax(Y_pred, axis=1)

    cm = confusion_matrix(test_dataset.labels, y_pred)

    print("Confusion Matrix:")
    print(cm)

    precision = precision_score(y_true=test_dataset.labels, y_pred=y_pred, average='weighted')
    recall = recall_score(y_true=test_dataset.labels, y_pred=y_pred, average='weighted')
    f1 = f1_score(y_true=test_dataset.labels, y_pred=y_pred, average='weighted')

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print('Classification Report')
    print(classification_report(test_dataset.labels, y_pred, target_names=classes))

# Main function
def main():
    train_dataset, validation_dataset, test_dataset, class_names = dataCollection()

    model_path = 'modelCollector/basicmodel1.keras'

    if os.path.exists(model_path):
        print("Loading existing model...")
        model = load_model(model_path)
    else:
        print("Creating a new model...")
        num_classes = len(class_names)  # Use class_names to get the number of classes
        model = create_model(num_of_classes=num_classes)

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        steps_per_epoch=len(train_dataset),
        validation_steps=len(validation_dataset)
    )

    model.save(model_path, save_format='tf')
    plot_model_performance(history)

    print("Evaluating model performance...")
    evaluate_model_performance(model, test_dataset, train_dataset)

    img_path = os.path.abspath('./models_data/sample_test.jpg')
    predict_single_image(model, img_path)

if __name__ == '__main__':
    main()
