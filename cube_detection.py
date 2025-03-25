import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import numpy as np

# Configuration
MODEL_DIR = "./models"
MODEL_NAME = "cube-detector-01.pth"
IMAGE_SIZE = 224  # Standard size for ResNet models
BATCH_SIZE = 32
NUM_EPOCHS = 20
PATIENCE = 5  # Early stopping patience
LEARNING_RATE = 0.001
CLASSES = ["not_cube", "cube"]  # Class names

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define data preprocessing pipeline
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def load_or_create_model():
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    
    # Check if model exists
    if os.path.exists(model_path):
        print(f"Found existing model, loading from {model_path}...")
        model = torch.load(model_path, map_location=device)
        return model
    
    print("No existing model found, creating new model...")
    
    # Load pre-trained ResNet50
    model = models.resnet50(pretrained=True)
    
    # Freeze all parameters in the pre-trained model
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(CLASSES))
    
    # Move model to the appropriate device
    model = model.to(device)
    
    return model

def train_model(model, train_dataloader, val_dataloader=None):
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # Only optimize parameters of the final layer (which are not frozen)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    best_accuracy = 0.0
    no_improvement_count = 0
    
    print("Starting model training...")
    
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_dataloader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_dataloader.dataset)
        epoch_acc = correct / total
        print(f"Training Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        
        # Validation phase
        if val_dataloader:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in tqdm(val_dataloader, desc="Validating"):
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # Statistics
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_epoch_loss = val_loss / len(val_dataloader.dataset)
            val_epoch_acc = val_correct / val_total
            print(f"Validation Loss: {val_epoch_loss:.4f}, Accuracy: {val_epoch_acc:.4f}")
            
            # Early stopping logic
            if val_epoch_acc > best_accuracy:
                best_accuracy = val_epoch_acc
                no_improvement_count = 0
                # Save best model
                torch.save(model, os.path.join(MODEL_DIR, "cube-detector-01-best.pth"))
                print(f"New best model saved with accuracy: {best_accuracy:.4f}")
            else:
                no_improvement_count += 1
                if no_improvement_count >= PATIENCE:
                    print(f"Early stopping triggered after {PATIENCE} epochs without improvement!")
                    break
    
    # Save final model
    torch.save(model, os.path.join(MODEL_DIR, MODEL_NAME))
    print(f"Model saved to: {os.path.abspath(os.path.join(MODEL_DIR, MODEL_NAME))}")
    
    return model

def predict_image(model, image_path):
    # Load and preprocess the image
    from PIL import Image
    
    if not os.path.exists(image_path):
        print(f"Test image does not exist: {image_path}")
        return
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Apply transformations
    img_tensor = data_transforms['val'](img).unsqueeze(0).to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
    # Get the predicted class
    _, predicted_idx = torch.max(outputs, 1)
    predicted_class = CLASSES[predicted_idx.item()]
    probability = probabilities[predicted_idx.item()].item()
    
    print("Prediction results:")
    print(f"Class: {predicted_class}, Probability: {probability:.4f}")
    print(f"Image is most likely: {predicted_class}, probability: {probability*100:.2f}%")
    
    # Return all class probabilities for reference
    class_probabilities = {CLASSES[i]: prob.item() for i, prob in enumerate(probabilities)}
    return predicted_class, probability, class_probabilities

def main():
    # Load or create model
    model = load_or_create_model()
    
    # Check if we need to train the model
    train_model_flag = True
    if os.path.exists(os.path.join(MODEL_DIR, MODEL_NAME)):
        response = input("Model already exists. Do you want to train it again? (y/n): ")
        train_model_flag = response.lower() == 'y'
    
    if train_model_flag:
        # Load datasets
        print("Loading training dataset...")
        try:
            train_dataset = datasets.ImageFolder(
                root="dataset",
                transform=data_transforms['train']
            )
            train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
            print(f"Training dataset size: {len(train_dataset)}")
            print(f"Classes: {train_dataset.classes}")
        except Exception as e:
            print(f"Error loading training dataset: {e}")
            return
        
        # Load validation dataset if it exists
        val_dataloader = None
        try:
            if os.path.exists("dataset_val") and os.path.isdir("dataset_val"):
                print("Loading validation dataset...")
                val_dataset = datasets.ImageFolder(
                    root="dataset_val",
                    transform=data_transforms['val']
                )
                val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
                print(f"Validation dataset size: {len(val_dataset)}")
            else:
                print("Warning: No separate validation dataset found at dataset_val")
                print("Will use training dataset for validation (not recommended for production)")
                val_dataloader = train_dataloader
        except Exception as e:
            print(f"Error loading validation dataset: {e}")
            val_dataloader = train_dataloader
        
        # Train the model
        model = train_model(model, train_dataloader, val_dataloader)
    
    # Test the model
    test_image_path = "test.jpeg"
    predict_image(model, test_image_path)

if __name__ == "__main__":
    main()