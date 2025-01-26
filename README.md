# Image Classifier Project

## Overview
This project uses convolutional neural network (CNN) to classify images into predefined categories. It demonstrates image preprocessing, model training, and prediction.

## Features
- Pre-trained or custom model support.
- Real-time or batch image classification.
- Accuracy evaluation and result visualization.

## Requirements
- Python 3.x
- Libraries: TensorFlow/Keras, NumPy, Matplotlib, sklearn

## Installation
1. Clone the repository:
   ```bash
   git clone git@github.com:tns30-dev/image-classifier-project.git
   ```
2. Navigate to the project directory:
   ```bash
   cd image-classifier-project
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Prepare your dataset and place it in the `models_data/` directory.
2. Run the desired script to train and evaluate the model. For example:
   ```bash
   python 01_basicModel.py
   ```
3. Other scripts demonstrate further enhancements such as dropout, increased epochs, image augmentation, and more. Use these scripts to experiment and optimize your model.

## Project Structure
- `01_basicModel.py`: Baseline model implementation.
- `02_addDropoutBasicModel.py`: Adds dropout layers to prevent overfitting.
- `03_increaseEpochs.py`: Increases the training epochs.
- `04_addImageAugmentation.py`: Applies image augmentation techniques.
- `05_changeImageSize.py`: Modifies input image size.
- `06_increaseTargetSizeAgain.py`: Further adjusts image size.
- `07_increaseNoOfEpochs.py`: Increases training epochs further.
- `08_addCnnAndNeuron.py`: Introduces more CNN layers and neurons.
- `09_addLearningStop.py`: Implements early stopping to prevent overtraining.
- `10_addRegularization.py`: Adds regularization to control model complexity.
- `11_increaseTargetSizeEpoches.py`: Combines increased target size and epochs.
- `modelCollector/`: Directory for saved models.
- `models_data/`: Directory for training and testing datasets.


