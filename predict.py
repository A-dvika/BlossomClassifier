import argparse
import torch
from torchvision import models, transforms
from torch import nn
from PIL import Image
import json
import numpy as np
import matplotlib.pyplot as plt

# Function to process the image
def process_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image = np.array(image) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std

    image = image.transpose((2, 0, 1))
    return torch.from_numpy(image).type(torch.FloatTensor)

# Function to predict the class of an image
def predict(image_path, model, topk=5):
    # Process the image
    img = process_image(image_path)

    # Add batch of size 1 to image
    img = img.unsqueeze(0)

    # Set model to evaluation mode
    model.eval()

    # Predict the class probabilities
    with torch.no_grad():
        output = model(img)
        probabilities = torch.exp(output)
        top_probabilities, top_indices = probabilities.topk(topk)
        top_probabilities = top_probabilities.numpy().flatten()
        top_indices = top_indices.numpy().flatten()

    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [class_names[str(idx + 1)] for idx in top_indices]

    return top_probabilities, top_classes

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Make predictions using a trained deep learning model.")
    parser.add_argument("image_path", help="Path to the image for prediction")
    parser.add_argument("checkpoint", help="Path to the model checkpoint")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top classes to display")
    parser.add_argument("--category_names", default="cat_to_name.json", help="Path to the mapping of categories to names")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")

    args = parser.parse_args()

    # Load the model checkpoint
    checkpoint = torch.load(args.checkpoint)
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    # Modify the classifier
    num_features = model.fc.in_features
    classifier = nn.Sequential(
        nn.Linear(num_features, checkpoint['hidden_units']),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(checkpoint['hidden_units'], len(checkpoint['class_to_idx'])),
        nn.LogSoftmax(dim=1)
    )
    model.fc = classifier

    model.load_state_dict(checkpoint['model_state_dict'])

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    model.to(device)

    # Process the image
    img = process_image(args.image_path)

    # Add batch of size 1 to image
    img = img.unsqueeze(0)

    # Set model to evaluation mode
    model.eval()

    # Predict the class probabilities
    with torch.no_grad():
        output = model(img.to(device))
        probabilities = torch.exp(output)
        top_probabilities, top_indices = probabilities.topk(args.top_k)
        top_probabilities = top_probabilities.cpu().numpy().flatten()
        top_indices = top_indices.cpu().numpy().flatten()

    # Load the category names mapping
    class_names = None
    with open(args.category_names, 'r') as f:
        class_names = json.load(f)

    # Convert indices to class labels
    top_classes = [class_names[str(idx + 1)] for idx in top_indices]


    # Display the results
    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)

    # Display the image
    image = Image.open(args.image_path)
    plt.imshow(image)
    plt.axis('off')

    # Plot the probabilities as a bar chart
    plt.subplot(2, 1, 2)
    plt.barh(top_classes, top_probabilities)
    plt.xlabel('Probability')
    plt.gca().invert_yaxis()  # Invert y-axis to display the highest probability at the top
    plt.show()
