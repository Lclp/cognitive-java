import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
# import cv2
from sklearn.model_selection import train_test_split

# Configuration
MODEL_DIR = "./models"
MODEL_NAME = "cube-detector-improved.pth"
IMAGE_SIZE = 512  # Standard size for ResNet models
BATCH_SIZE = 16
NUM_EPOCHS = 30
PATIENCE = 7  # Early stopping patience
LEARNING_RATE = 0.0001
CLASSES = ["not_cube", "cube"]  # Class names
RANDOM_SEED = 42

# Set random seeds for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define data preprocessing pipeline with more augmentation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def visualize_augmentations(dataset, num_samples=5):
    """Visualize augmentations applied to images"""
    fig, axes = plt.subplots(num_samples, 5, figsize=(15, 3*num_samples))
    
    for i in range(num_samples):
        # Get a random image
        idx = random.randint(0, len(dataset)-1)
        img, label = dataset[idx]
        
        # Original image (denormalize)
        img_np = img.numpy().transpose(1, 2, 0)
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)
        
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f"Class: {dataset.classes[label]}")
        
        # Show 4 more augmented versions
        for j in range(1, 5):
            aug_img, _ = dataset[idx]
            aug_np = aug_img.numpy().transpose(1, 2, 0)
            aug_np = aug_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            aug_np = np.clip(aug_np, 0, 1)
            axes[i, j].imshow(aug_np)
            axes[i, j].set_title(f"Augmentation {j}")
    
    plt.tight_layout()
    plt.savefig("augmentation_examples.png")
    print("Saved augmentation examples to augmentation_examples.png")


def load_or_create_model():
    """Create or load a model with improved architecture"""
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)

    # Create a new model instance - try EfficientNet instead of ResNet
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Freeze early layers but unfreeze later layers for fine-tuning
    # This allows the model to adapt better to the specific task
    layers_to_unfreeze = 30  # Unfreeze the last few layers
    ct = 0
    for child in model.features.children():
        ct += 1
        if ct < len(list(model.features.children())) - layers_to_unfreeze:
            for param in child.parameters():
                param.requires_grad = False

    # Replace the classifier
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_ftrs, len(CLASSES))
    )

    # Check if model exists and load state dict if it does
    if os.path.exists(model_path):
        print(f"Found existing model, loading from {model_path}...")
        try:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using newly initialized model instead...")
    else:
        print("No existing model found, using new model...")

    # Move model to the appropriate device
    model = model.to(device)
    return model


def save_model(model, path):
    """Save model state dict only to avoid pickle issues"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model state dict saved to: {os.path.abspath(path)}")


def create_weighted_sampler(dataset):
    """Create a weighted sampler to handle class imbalance"""
    targets = [label for _, label in dataset.samples]
    class_counts = np.bincount(targets)
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = class_weights[targets]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler


def train_model(model, train_dataloader, val_dataloader=None):
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Use AdamW optimizer with weight decay for regularization
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )
    
    # Learning rate scheduler to reduce LR when training plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=3, 
        verbose=True
    )

    best_accuracy = 0.0
    no_improvement_count = 0

    print("Starting model training...")
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

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
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        print(f"Training Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        # Validation phase
        if val_dataloader:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            # Track predictions for confusion matrix
            all_preds = []
            all_labels = []

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
                    
                    # Save predictions and labels
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            val_epoch_loss = val_loss / len(val_dataloader.dataset)
            val_epoch_acc = val_correct / val_total
            history['val_loss'].append(val_epoch_loss)
            history['val_acc'].append(val_epoch_acc)
            print(f"Validation Loss: {val_epoch_loss:.4f}, Accuracy: {val_epoch_acc:.4f}")
            
            # Print per-class accuracy
            from sklearn.metrics import classification_report
            print("\nClassification Report:")
            print(classification_report(all_labels, all_preds, target_names=CLASSES))

            # Update learning rate based on validation accuracy
            scheduler.step(val_epoch_acc)

            # Early stopping logic
            if val_epoch_acc > best_accuracy:
                best_accuracy = val_epoch_acc
                no_improvement_count = 0
                # Save best model
                save_model(model, os.path.join(MODEL_DIR, "cube-detector-best.pth"))
                print(f"New best model saved with accuracy: {best_accuracy:.4f}")
            else:
                no_improvement_count += 1
                if no_improvement_count >= PATIENCE:
                    print(f"Early stopping triggered after {PATIENCE} epochs without improvement!")
                    break

    # Save final model
    save_model(model, os.path.join(MODEL_DIR, MODEL_NAME))
    
    # Plot training history
    plot_training_history(history)

    return model


def plot_training_history(history):
    """Plot training and validation metrics"""
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Saved training history plot to training_history.png")


def predict_image(model, image_path):
    """Predict class for a single image with visualization"""
    if not os.path.exists(image_path):
        print(f"Test image does not exist: {image_path}")
        return

    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Create a copy for visualization
    img_display = img.copy()
    
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
    print(f"Image is most likely: {predicted_class}, probability: {probability * 100:.2f}%")
    
    # Show all class probabilities
    for i, cls in enumerate(CLASSES):
        print(f"{cls}: {probabilities[i].item()*100:.2f}%")

    # Visualize the prediction
    plt.figure(figsize=(8, 8))
    plt.imshow(np.array(img_display))
    plt.title(f"Prediction: {predicted_class} ({probability*100:.2f}%)")
    plt.axis('off')
    
    # Save the visualization
    plt.savefig(f"prediction_{os.path.basename(image_path)}")
    print(f"Saved prediction visualization to prediction_{os.path.basename(image_path)}")

    # Return all class probabilities for reference
    class_probabilities = {CLASSES[i]: prob.item() for i, prob in enumerate(probabilities)}
    return predicted_class, probability, class_probabilities


def prepare_datasets():
    """Prepare datasets with proper validation split"""
    print("Preparing datasets...")
    
    # Check if dataset exists
    if not os.path.exists("dataset"):
        print("Error: dataset directory not found!")
        return None, None
    
    # Load all images and their labels
    full_dataset = datasets.ImageFolder(
        root="dataset",
        transform=data_transforms['train']
    )
    
    # Print class distribution
    class_counts = {}
    for _, label in full_dataset.samples:
        class_name = full_dataset.classes[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print("Class distribution in dataset:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} images")
    
    # Create train/val split if no separate validation set exists
    if not os.path.exists("dataset_val") or not os.path.isdir("dataset_val"):
        print("Creating train/validation split from main dataset...")
        
        # Get indices for each class
        class_indices = {i: [] for i in range(len(full_dataset.classes))}
        for idx, (_, label) in enumerate(full_dataset.samples):
            class_indices[label].append(idx)
        
        # Split each class separately to maintain class distribution
        train_indices = []
        val_indices = []
        for class_idx, indices in class_indices.items():
            train_idx, val_idx = train_test_split(
                indices, 
                test_size=0.2,  # 20% for validation
                random_state=RANDOM_SEED
            )
            train_indices.extend(train_idx)
            val_indices.extend(val_idx)
        
        # Create train and validation datasets
        from torch.utils.data import Subset
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(
            datasets.ImageFolder(root="dataset", transform=data_transforms['val']), 
            val_indices
        )
        
        # Create data loaders
        train_sampler = WeightedRandomSampler(
            weights=[1.0] * len(train_indices),
            num_samples=len(train_indices),
            replacement=True
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE,
            sampler=train_sampler,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
        
        print(f"Training set: {len(train_indices)} images")
        print(f"Validation set: {len(val_indices)} images")
        
    else:
        # Use separate validation directory
        print("Using separate validation directory...")
        
        train_dataset = datasets.ImageFolder(
            root="dataset",
            transform=data_transforms['train']
        )
        
        val_dataset = datasets.ImageFolder(
            root="dataset_val",
            transform=data_transforms['val']
        )
        
        # Create a weighted sampler to handle class imbalance
        train_sampler = create_weighted_sampler(train_dataset)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=train_sampler,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
        
        print(f"Training set: {len(train_dataset)} images")
        print(f"Validation set: {len(val_dataset)} images")
    
    # Visualize some augmented images
    try:
        visualize_augmentations(full_dataset)
    except Exception as e:
        print(f"Could not visualize augmentations: {e}")
    
    return train_loader, val_loader


def main():
    # Prepare datasets
    train_dataloader, val_dataloader = prepare_datasets()
    if train_dataloader is None:
        return
    
    # Load or create model
    model = load_or_create_model()

    # Check if we need to train the model
    train_model_flag = True
    if os.path.exists(os.path.join(MODEL_DIR, MODEL_NAME)):
        response = input("Model already exists. Do you want to train it again? (y/n): ")
        train_model_flag = response.lower() == 'y'

    if train_model_flag:
        # Train the model
        model = train_model(model, train_dataloader, val_dataloader)

    # Test the model on a specific image
    test_image_path = input("Enter path to test image (default: ./pics/yes.jpg): ").strip() or "./pics/yes.jpg"
    predict_image(model, test_image_path)


if __name__ == "__main__":
    main()