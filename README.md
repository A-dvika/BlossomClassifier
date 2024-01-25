## AI Programming with Python -Udacity
# BlossomClassifier-Flower Image Classification Project

```markdown
# Flower Image Classification Project

## Overview

This project focuses on training a deep neural network to classify flower images. It includes two main components: training the model using a provided dataset and making predictions on new flower images using the trained model.

## Project Structure

The project is organized into two main parts:

1. **Training the Model (`train.py`):**
   - The `train.py` script is a command-line interface for training a deep neural network on an image dataset.
   - It uses PyTorch and torchvision to define and train the model.
   - Command-line arguments allow customization of the data directory, model architecture, learning rate, number of hidden units, epochs, and GPU usage.
   - The trained model is saved, and a checkpoint with relevant information is created for later use.

2. **Making Predictions (`predict.py`):**
   - The `predict.py` script is designed for making predictions using a trained model checkpoint.
   - It takes an image file, a model checkpoint, and additional parameters like the number of top classes to display.
   - The script loads the model checkpoint, modifies the classifier, processes the image, and outputs the top predicted classes with their probabilities.
   - It also supports GPU inference and provides an option to specify a mapping of category names.

## How to Use

### Training the Model

```bash
python train.py data_directory --save_dir save_directory --arch resnet18 --learning_rate 0.001 --hidden_units 512 --epochs 10 --gpu
```

- `data_directory`: Path to the dataset.
- `--save_dir`: Directory to save checkpoints (default: current directory).
- `--arch`: Model architecture (default: 'resnet18').
- `--learning_rate`: Learning rate for the optimizer (default: 0.001).
- `--hidden_units`: Number of hidden units in the classifier (default: 512).
- `--epochs`: Number of training epochs (default: 10).
- `--gpu`: Use GPU for training (optional).

### Making Predictions

```bash
python predict.py image_path checkpoint --top_k 5 --category_names cat_to_name.json --gpu
```

- `image_path`: Path to the image for prediction.
- `checkpoint`: Path to the model checkpoint.
- `--top_k`: Number of top classes to display (default: 5).
- `--category_names`: Path to the mapping of categories to names (default: cat_to_name.json).
- `--gpu`: Use GPU for inference (optional).
## Notebook Training (`train.ipynb`)

The project also provides a Jupyter notebook for training the deep neural network. The notebook provides an interactive environment where you can execute code cells step by step.

### How to Use

1. Open the `train.ipynb` notebook in Jupyter environment.
2. Execute each code cell sequentially.
3. Customize hyperparameters, model architecture, and other settings as needed.
4. Follow the instructions and comments within the notebook for guidance.
5. Save the trained model checkpoint generated by the notebook for later use.

### Dependencies

Make sure to install the required dependencies before running the notebook. You can install them using the following command:

```bash
pip install -r requirements.txt

## Dependencies

- Python 3.x
- PyTorch
- torchvision
- NumPy
- Matplotlib
- PIL (Pillow)



