# Handwritten Digit Recognition

A deep learning solution for the Kaggle Digit Recognizer competition using TensorFlow and Keras.

## Competition

Kaggle Competition: Digit Recognizer

Notebook Link: https://www.kaggle.com/code/kishankushavaha/handwritten-digit-recognition

## Problem Statement

The goal is to identify handwritten digits (0–9) from grayscale images of size 28×28 pixels. The dataset is based on the MNIST handwritten digit database.

## Dataset

* Training Samples: 42,000
* Test Samples: 28,000
* Features: 784 pixel values per image
* Classes: 10 digits (0–9)

## Approach

### Data Preprocessing

* Loaded train and test datasets
* Separated features and labels
* Normalized pixel values by dividing by 255
* Split training data into training and validation sets

### Model Architecture

```python
Input Layer (784)

Dense(128, activation='relu')

Dense(64, activation='relu')

Dense(10, activation='softmax')
```

### Training Configuration

* Optimizer: Adam
* Loss Function: Sparse Categorical Crossentropy
* Epochs: 10
* Validation Split: 20%

## Results

| Metric              | Value  |
| ------------------- | ------ |
| Validation Accuracy | 97.07% |
| Validation Loss     | 0.1212 |

## Submission

Predictions were generated on the test dataset and saved in the required Kaggle submission format:

```text
ImageId,Label
1,2
2,0
3,9
...
```

## Libraries Used

* NumPy
* Pandas
* Scikit-learn
* TensorFlow
* Keras

## Run the Notebook

1. Clone the repository

```bash
git clone <repository-url>
```

2. Install dependencies

```bash
pip install numpy pandas scikit-learn tensorflow
```

3. Open the notebook and run all cells.

## Kaggle Notebook

https://www.kaggle.com/code/kishankushavaha/handwritten-digit-recognition
