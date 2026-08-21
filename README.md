# Neural Lab

A personal collection of machine learning models and experiments. This lab serves as a technical portfolio where I can continuously add models that I train and deploy, demonstrating both the underlying architecture and the interactive browser-based deployment.

## Structure

```text
/
├── index.html        # Main landing page for the Model Lab
├── assets/           # Global assets (images, icons)
├── css/              # Global styling (main.css, components.css)
├── js/               # Global JavaScript
└── models/           # Individual model directories
    ├── mnist-cnn/    # Custom CNN trained on MNIST
    └── _template/    # Template for adding new models
```

## Adding a New Model

To add a new model to this portfolio, follow these steps:

1. Create a new folder inside `/models` (e.g., `my-new-model/`). Alternatively, copy the `_template/` folder.
2. Put the browser inference code inside `/js/app.js`.
3. Put the trained browser-compatible model files inside `/model/`.
4. Put all Python training/evaluation scripts and requirements inside `/python/`.
5. Update `index.html` within the new model directory to describe the architecture and provide an interactive demo.
6. Add the new model as a card on the global `/index.html` landing page.
7. Add a `README.md` in the model folder explaining the model.

## Technology Stack

- **Frontend**: HTML5, CSS3 (Custom Variables), Vanilla JavaScript.
- **Inference Engine**: TensorFlow.js (for browser-side ML execution).
- **Model Training**: Python, TensorFlow/Keras.
