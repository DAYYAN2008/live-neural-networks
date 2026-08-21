import numpy as np
import tensorflow as tf
import random

model = tf.keras.models.load_model('mnist_cnn_model.keras')

with np.load('mnist.npz') as data:
   x_test = data['x_test']
   y_test = data['y_test']

sample_indices = random.sample(range(len(x_test)), 5)

for idx in sample_indices:
   raw_image = x_test[idx]
   actual_label = y_test[idx]
   input_tensor = raw_image.reshape((1, 28, 28, 1)).astype("float32")/ 255.0
   probabilities = model.predict(input_tensor, verbose=0)
   predicted_label = np.argmax(probabilities)
   confidence = np.max(probabilities) * 100

   print(f"Sample #{idx:04d} -> True Label: {actual_label} | Predicted: {predicted_label} (Confidence: {confidence:.2f}%)")