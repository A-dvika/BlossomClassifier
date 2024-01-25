import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

# Function to print a summary of the model architecture
def print_model_summary(model, input_size=(3, 224, 224)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    summary_str = []
    summary_str.append(f"{'Layer (type)':<25} {'Output Shape':<25} {'Param #':<15}")
    summary_str.append("="*75)
    
    total_params = 0
    trainable_params = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            total_params += sum(p.numel() for p in module.parameters())
            if module.weight.requires_grad:
                trainable_params += sum(p.numel() for p in module.parameters())
            summary_str.append(f"{name:<25} {str(module):<25} {sum(p.numel() for p in module.parameters()):<15}")
    
    summary_str.append("="*75)
    summary_str.append(f"Total params: {total_params}")
    summary_str.append(f"Trainable params: {trainable_params}")
    summary_str.append(f"Non-trainable params: {total_params - trainable_params}")
    
    for line in summary_str:
        print(line)
def save_checkpoint(model, save_dir, arch, hidden_units, image_datasets, optimizer, epochs):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'class_to_idx': image_datasets['train'].class_to_idx,
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': epochs,
        'hidden_units': hidden_units  # Add this line to include the hidden_units information
    }

    torch.save(checkpoint, f'{save_dir}/Checkpoint.pth')



    torch.save(checkpoint, f'{save_dir}/Checkpoint.pth')
def train_model(data_dir, save_dir, arch, learning_rate, hidden_units, epochs, gpu):
    # Define data directories
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define data transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load datasets
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }

    # Create data loaders
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': DataLoader(image_datasets['valid'], batch_size=32),
        'test': DataLoader(image_datasets['test'], batch_size=32)
    }

    

    # Define a new classifier
    num_classes = 102  # Assuming 102 classes for the flower dataset
    # Load a pre-trained model
    model = getattr(models, arch)(pretrained=True)

    # Modify the classifier
    num_features = model.fc.in_features
    classifier = nn.Sequential(
        nn.Linear(num_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_units, num_classes),
        nn.LogSoftmax(dim=1)
    )
    model.fc = classifier  # Modify the classifier in the model

    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
    model.to(device)

    # Define loss criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()
        train_loss = 0.0

        for inputs, labels in tqdm(dataloaders['train'], desc="Training", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(dataloaders['train'].dataset)

        # Validation loop
        model.eval()
        valid_loss = 0.0
        correct_predictions = 0

        with torch.no_grad():
            for inputs, labels in tqdm(dataloaders['valid'], desc="Validation", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_predictions += torch.sum(preds == labels.data)

        valid_loss /= len(dataloaders['valid'].dataset)
        valid_acc = correct_predictions.double() / len(dataloaders['valid'].dataset)

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f} - Valid Loss: {valid_loss:.4f} - Valid Acc: {valid_acc:.4f}")

    best_model_state = model.state_dict()
    torch.save(best_model_state, f'{save_dir}/best_model.pth')
    # Save checkpoint
    save_checkpoint(model, save_dir, arch, hidden_units, image_datasets, optimizer, epochs)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a deep neural network on an image dataset")
    parser.add_argument("data_directory", help="Path to the dataset")
    parser.add_argument("--save_dir", help="Directory to save checkpoints", default=".")
    parser.add_argument("--arch", help="Architecture (e.g., 'mobilenet_v2')", default="resnet18")
    parser.add_argument("--learning_rate", type=float, help="Learning rate", default=0.001)
    parser.add_argument("--hidden_units", type=int, help="Number of hidden units", default=512)
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=10)
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")

    args = parser.parse_args()

    # Train the model
    train_model(args.data_directory, args.save_dir, args.arch, args.learning_rate, args.hidden_units, args.epochs, args.gpu)