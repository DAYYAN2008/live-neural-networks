# MNIST CNN

A Custom Convolutional Neural Network trained on the MNIST handwritten digit dataset.

This directory isolates all code related to this specific model.

## Directory Layout

- `index.html`: The interactive model page.
- `js/app.js`: TensorFlow.js canvas interaction and inference logic.
- `model/`: Contains `model.json` and weight binaries exported from Keras.
- `python/`: Contains `train.py`, `evaluate.py`, the original dataset `mnist.npz`, and the keras model file `mnist_cnn_model.keras`.

## Performance

- **Test Accuracy**: 98.6%
- **Framework**: TensorFlow (Python) & TFJS (Browser)
- **Optimizer**: Adam
