# Handwritten Digit Recognition using Deep Learning

## Overview

This project solves the Kaggle Digit Recognizer competition using a Deep Neural Network (DNN) built with TensorFlow and Keras.

The model is trained on the MNIST dataset, which contains 28×28 grayscale images of handwritten digits (0–9). The goal is to correctly classify each image into its corresponding digit.

---

## Dataset

The dataset consists of:

* **train.csv**

  * 42,000 training examples
  * 784 pixel features
  * 1 target column (`label`)

* **test.csv**

  * 28,000 testing examples
  * 784 pixel features

Each image contains:

* Size: 28 × 28 pixels
* Total Features: 784
* Pixel values range from 0 to 255

---

## Technologies Used

* Python
* NumPy
* Pandas
* TensorFlow
* Keras
* Scikit-learn

---

## Project Workflow

### 1. Load Dataset

```python
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
```

### 2. Separate Features and Labels

```python
X = train.drop("label", axis=1)
y = train["label"]
```

### 3. Normalize Data

Pixel values are scaled between 0 and 1.

```python
X = X / 255.0
test = test / 255.0
```

### 4. Train-Validation Split

```python
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 5. Build Neural Network

Architecture:

Input Layer (784 neurons)
↓
Dense Layer (128 neurons, ReLU)
↓
Dense Layer (64 neurons, ReLU)
↓
Output Layer (10 neurons, Softmax)

```python
model = Sequential()

model.add(Dense(128, activation="relu", input_shape=(784,)))
model.add(Dense(64, activation="relu"))
model.add(Dense(10, activation="softmax"))
```

### 6. Compile Model

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

### 7. Train Model

```python
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    validation_data=(X_valid, y_valid)
)
```

### 8. Evaluate Model

```python
loss, accuracy = model.evaluate(X_valid, y_valid)
```

Validation Accuracy:

**97.07%**

---

## Generate Predictions

```python
pred = model.predict(test)
pred = np.argmax(pred, axis=1)
```

---

## Create Submission File

```python
submission = pd.DataFrame({
    "ImageId": range(1, len(pred) + 1),
    "Label": pred
})

submission.to_csv("submission.csv", index=False)
```

Generated file:

```text
submission.csv
```

Format:

| ImageId | Label |
| ------- | ----- |
| 1       | 2     |
| 2       | 0     |
| 3       | 9     |
| ...     | ...   |

---

## Results

| Metric              | Value                           |
| ------------------- | ------------------------------- |
| Training Samples    | 42,000                          |
| Test Samples        | 28,000                          |
| Epochs              | 10                              |
| Optimizer           | Adam                            |
| Loss Function       | Sparse Categorical Crossentropy |
| Validation Accuracy | 97.07%                          |

---

## Future Improvements

* Add Dropout layers to reduce overfitting.
* Use Convolutional Neural Networks (CNNs).
* Implement Data Augmentation.
* Apply Learning Rate Scheduling.
* Perform Hyperparameter Tuning.

---

## Author

**Kishan**

M.Tech Student
Subject Matter Expert (Mathematics)
Interested in Machine Learning, Deep Learning, Generative AI, and Cyber Security.
