#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build the model
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flatten the input
    Dense(128, activation='relu'),  # First hidden layer
    Dense(64, activation='relu'),   # Second hidden layer
    Dense(10, activation='softmax') # Output layer
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')


# In[2]:


# Save the model
model.save('mnist_model.h5')


# In[ ]:


import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('mnist_model.h5')

# Function to preprocess the captured image
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    normalized = resized.astype('float32') / 255.0
    reshaped = np.reshape(normalized, (1, 28, 28))
    return reshaped

# Function to predict the digit
def predict_digit(img):
    processed = preprocess_image(img)
    prediction = model.predict(processed)
    return np.argmax(prediction)

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define the region of interest (ROI) for capturing the digit
    x, y, w, h = 300, 100, 200, 200
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi = frame[y:y + h, x:x + w]

    # Predict the digit in the ROI
    digit = predict_digit(roi)

    # Display the prediction on the frame
    cv2.putText(frame, str(digit), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow('Digit Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

