from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

#input image size
shape_initializer = (250, 250)

#File directory
TRAINING_DIR = 'models_data/train'
TEST_DIR = 'models_data/test'

#Epochs
epochs = 80

#Batch size
BATCH_SIZE = 16

def dataCollection():

    seed = 10

    #Image augmentation for training dataset
    data_generator = ImageDataGenerator(
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    
    val_data_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    #Collect data for training dataset combined with image augmentation
    train_generator = data_generator.flow_from_directory(TRAINING_DIR, target_size=shape_initializer, shuffle=True, seed=seed,
                                                     class_mode='categorical', batch_size=BATCH_SIZE, subset="training")
    
    #Collect data for validation dataset without image augmentation
    validation_generator = val_data_generator.flow_from_directory(TRAINING_DIR, target_size=shape_initializer, shuffle=False, seed=seed,
                                                     class_mode='categorical', batch_size=BATCH_SIZE, subset="validation")
    
    #Collect data for test dataset 
    test_generator = ImageDataGenerator(rescale=1./255)
    test_generator = test_generator.flow_from_directory(TEST_DIR, target_size=shape_initializer, shuffle=False, seed=seed,
                                                     class_mode='categorical', batch_size=BATCH_SIZE)
    
    return train_generator, validation_generator, test_generator

#Model creation
def create_model(num_of_classes):
    regularizer = tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)  
    
    cnn = tf.keras.models.Sequential()
    cnn.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(shape_initializer[0], shape_initializer[1], 3), kernel_regularizer=regularizer))
    cnn.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    cnn.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizer))
    cnn.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    cnn.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer=regularizer))
    cnn.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    cnn.add(tf.keras.layers.Dropout(0.25))
    cnn.add(tf.keras.layers.Flatten())
    cnn.add(tf.keras.layers.Dense(units=512, activation='relu', kernel_regularizer=regularizer))
    cnn.add(tf.keras.layers.Dropout(0.5))
    cnn.add(tf.keras.layers.Dense(4, activation='softmax', kernel_regularizer=regularizer))

    cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return cnn

#Method for evluating model performance(Plot Train/Valid accuracy and Train/Valid loss)
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

#Method for evaluating model performance(showcasing how probability distrubtion of Softmax funtion works behind the scene)
def predict_single_image(model, img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(shape_initializer[0], shape_initializer[1]))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    class_labels = ['apple', 'banana', 'mixed', 'orange']  

    print(f"Predicted class: {class_labels[predicted_class[0]]}")
    print("Softmax probability distribution:")
    for i, prob in enumerate(predictions[0]):
        print(f"{class_labels[i]}: {prob:.4f}")


#Method for evaluating model performance (Confusion matrix, F1_score, Precision, f1_score, classification report)
def evaluate_model_performance(model, test_generator, train_generator):
    classes = list(train_generator.class_indices.keys())

    Y_pred = model.predict_generator(test_generator)
    y_pred = np.argmax(Y_pred, axis=1)

    cm = confusion_matrix(test_generator.classes, y_pred)

    print("Confusion Matrix:")
    print(cm)

    precision = precision_score(y_true=test_generator.classes,y_pred= y_pred, average='weighted')
    recall = recall_score(y_true=test_generator.classes,y_pred= y_pred, average='weighted')
    f1 = f1_score(y_true=test_generator.classes,y_pred= y_pred, average='weighted')

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print('Classification Report')
    print(classification_report(test_generator.classes, y_pred, target_names=classes))

#Method for evaluating model performance(loss_accuracy numerical value provider)
def loss_accuracy_numerical_provider(model, test_generator, history):
    test_loss, test_acc = model.evaluate(x= test_generator)
    print(f"Test accuracy: {test_acc}")
    print(f"Test loss: {test_loss}")

    final_train_accuracy = history.history['accuracy'][-1]
    final_val_accuracy = history.history['val_accuracy'][-1]
    print(f"Final Training Accuracy: {final_train_accuracy * 100:.2f}%")
    print(f"Final Validation Accuracy: {final_val_accuracy * 100:.2f}%")

def main():

    #Collect the data
    train_set, valid_set, test_set = dataCollection()

    #Retrieve the samples
    train_samples = train_set.samples
    validation_samples = valid_set.samples
    test_samples = test_set.samples

    #Get the number of classes
    classes = list(train_set.class_indices.keys())
    num_classes  = len(classes) 

    #First check whether trained model exists or not. 
    model_path = 'modelCollector/increaseTargetSizeEpoches'

    #create or retrieve the model
    if os.path.exists(model_path):
        print("Loading existing model...")
        model = load_model(model_path)
    else:
        print("Creating a new model...")
        model = create_model(num_of_classes=num_classes)

    #Compile the model
    model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])

    #Adding early stopping and reduce learning mechanism
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    #Train the model
    history = model.fit(
        train_set,
        steps_per_epoch= train_samples // BATCH_SIZE,
        epochs=epochs,
        validation_data=valid_set,
        verbose = 1,
        validation_steps=validation_samples // BATCH_SIZE,
        callbacks=[early_stopping, reduce_lr])
    
    #Plot the graph
    plot_model_performance(history)

    #loss_accuracy numerical value provider
    loss_accuracy_numerical_provider(model, test_set, history)

    #(Confusion matrix, F1_score, Precision, f1_score, classification report)
    evaluate_model_performance(model, test_generator=test_set, train_generator=train_set)

    #showcasing how probability distrubtion of Softmax funtion works behind the scene
    img_path = os.path.abspath('./models_data/sample_test.jpg') 
    predict_single_image(model, img_path)

if __name__ == '__main__':
    main()

