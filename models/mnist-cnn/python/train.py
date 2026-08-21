import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Load data without trailing commas
with np.load('mnist.npz') as data:
   x_train = data['x_train']
   y_train = data['y_train']
   x_test = data['x_test']
   y_test = data['y_test']

# 3. Add the single channel dimension (Height, Width, Channel = 1)
X_train = x_train.reshape((60000, 28, 28, 1))
X_test = x_test.reshape((10000, 28, 28, 1))

# 4. Normalize pixel values from [0, 255] to [0.0, 1.0]
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

model = models.Sequential([
   layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
   layers.MaxPooling2D((2, 2)),
   layers.Conv2D(64, (3, 3), activation='relu'),
   layers.MaxPooling2D((2, 2)),
   layers.Flatten(),
   layers.Dense(64, activation='relu'),
   layers.Dense(10, activation='softmax')
])

model.summary()

model.compile(
   optimizer='adam',
   loss='sparse_categorical_crossentropy',
   metrics=['accuracy']
)

print("Training the model...")

history = model.fit(
   X_train, y_train,
   epochs=5,
   batch_size=64,
   validation_data = (X_test, y_test)
)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n==========================================")
print(f"Final Accuracy on 10,000 Test Set: {test_acc * 100:.2f}%")
print(f"==========================================")

model.save('mnist_cnn_model.keras')