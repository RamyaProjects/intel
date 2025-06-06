# Intel Image Classification using CNN and Transfer Learning
**Overview**

This project classifies images of natural scenes into 6 categories using a Convolutional Neural Network (CNN) and transfer learning (VGG16). The dataset used is the Intel Image Classification Dataset, containing various landscape and infrastructure images.

**Technologies Used**
- Python
- TensorFlow / Keras
- Matplotlib, Seaborn
- Scikit-learn
  
**Dataset Structure**

The dataset is organized into the following folders:
- seg_train/seg_train: Training images
- seg_test/seg_test: Testing images
- seg_pred/seg_pred: Prediction samples (optional visualization)
  
**Project Workflow**
**1. Data Visualization & Exploration**
- Random samples from each class are displayed.
- Class distribution visualized with bar chart and pie chart.

**2. Data Preparation**
- Images loaded using image_dataset_from_directory.
- Data split into training (80%) and validation (20%) sets.
- Images resized to 128x128 and normalized (rescaled to [0, 1]).

**Model 1: Custom CNN**
**Architecture**
- 3 Convolutional blocks with:
  - Conv2D → BatchNormalization → MaxPooling2D
- Flatten → Dense (128) → Dropout → Dense (256) → Dropout
- Final output layer with 6 units (softmax)

**Compilation**
- Loss: categorical_crossentropy
- Optimizer: Adam (lr=0.0001)
- Metrics: accuracy

**Callbacks**
- EarlyStopping (patience=5)
- ReduceLROnPlateau (lr decay on plateau)

**Results**
- Training stopped at epoch 16
- Test Accuracy: ~82%
- Confusion matrix and classification report visualized.

**Model 2: VGG16 Transfer Learning**
- Used pre-trained VGG16 as the base (frozen convolutional layers).
- Added:
    - Flatten
    - Dense(256) → Dropout
    - Dense(128)
    - Final output: Dense(6, softmax)
 
**Compilation**
- Loss: categorical_crossentropy
- Optimizer: Adam
  
**Callbacks**
- EarlyStopping with best weight restoration
- ModelCheckpoint to save best model

**Results**
- Training stopped at epoch 24 (best at epoch 14)
- Test Accuracy: ~85%
- Significant improvement over the custom CNN
- Confusion matrix and classification report visualized.

**Visualizations**
- Training/Validation Accuracy & Loss over epochs
- Predictions on a random test batch with confidence scores
- Confusion Matrix for test set evaluation  
