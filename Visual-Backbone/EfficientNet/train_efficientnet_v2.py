#!/usr/bin/env python3
"""
EfficientNetV2-S Model Training for DermNet Dataset
23-class skin disease classification task

This implementation uses the actual EfficientNetV2-S model with:
- Input size: 384x384 (optimized for EfficientNetV2-S)
- Batch size: 16 (adjusted for larger input size)
- Enhanced data augmentation pipeline
"""

import argparse
import json
import os
import time
import warnings
from pathlib import Path
from typing import Dict, Tuple, List
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import timm
from tqdm import tqdm

warnings.filterwarnings('ignore')


class DermNetDataset(Dataset):
    """Custom dataset for DermNet skin disease images"""
    
    def __init__(self, dataframe: pd.DataFrame, data_root: str, transform=None, is_training: bool = True):
        self.dataframe = dataframe
        self.data_root = Path(data_root)
        self.transform = transform
        self.is_training = is_training
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = self.data_root / row['image_path']
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image if loading fails
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(row['label'], dtype=torch.long)
        
        return image, label


class EfficientNetV2Classifier(nn.Module):
    """EfficientNetV2-S based classifier for skin disease classification"""
    
    def __init__(self, num_classes: int = 23, freeze_layers: int = 150):
        super(EfficientNetV2Classifier, self).__init__()
        
        # Load pre-trained EfficientNetV2-S using timm
        self.backbone = timm.create_model(
            "tf_efficientnetv2_s.in21k_ft_in1k", 
            pretrained=True,
            num_classes=num_classes
        )

        # Freeze early layers for transfer learning
        self._freeze_layers(freeze_layers)
        
        # Replace the classifier head with a custom one
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def _freeze_layers(self, freeze_layers: int):
        """Freeze the first N layers of the backbone"""
        if freeze_layers > 0:
            # Count total parameters
            total_params = sum(1 for _ in self.backbone.parameters())
            
            # Freeze feature extraction layers
            params = list(self.backbone.parameters())
            for i, param in enumerate(params[:freeze_layers]):
                param.requires_grad = False
            print(f"Frozen first {freeze_layers} layers out of {total_params} total parameters")
    
    def forward(self, x):
        return self.backbone(x)


def get_transforms(input_size: int = 384, is_training: bool = True):
    """Get data transforms for training and validation"""
    
    if is_training:
        return transforms.Compose([
            transforms.Resize((input_size + 32, input_size + 32)),
            transforms.RandomCrop((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


class CosineAnnealingWarmupRestarts(optim.lr_scheduler._LRScheduler):
    """Cosine annealing with warm restarts and warmup"""
    
    def __init__(self, optimizer, first_cycle_steps, cycle_mult=1., max_lr=0.1, min_lr=0.001, 
                 warmup_steps=0, gamma=1., last_epoch=-1):
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set lr of each param group according to the specified schedule
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + np.cos(np.pi * (self.step_in_cycle-self.warmup_steps) \
                                  / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(np.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def load_data(data_root: str, processed_data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Load training and test data"""
    
    # Load processed data
    train_df = pd.read_csv(os.path.join(processed_data_path, 'train.csv'))
    test_df = pd.read_csv(os.path.join(processed_data_path, 'test.csv'))
    
    # Load class weights
    with open(os.path.join(processed_data_path, 'class_weights.json'), 'r') as f:
        class_weights = json.load(f)
    
    # Convert string keys to int
    class_weights = {int(k): v for k, v in class_weights.items()}
    
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Number of classes: {len(class_weights)}")
    
    return train_df, test_df, class_weights


def train_epoch(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, 
                optimizer: optim.Optimizer, device: torch.device, epoch: int) -> Tuple[float, float]:
    """Train for one epoch"""
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} Training')
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, 
                  device: torch.device, epoch: int) -> Tuple[float, float]:
    """Validate for one epoch"""
    
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} Validation')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device, 
                  num_classes: int) -> Dict:
    """Evaluate model on test set"""
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': all_predictions,
        'labels': all_labels,
        'confusion_matrix': cm.tolist()
    }
    
    return results


def plot_training_history(history: Dict, save_path: str):
    """Plot and save training history"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Training Loss', marker='o')
    ax1.plot(history['val_loss'], label='Validation Loss', marker='s')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Training Accuracy', marker='o')
    ax2.plot(history['val_acc'], label='Validation Accuracy', marker='s')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: str):
    """Plot and save confusion matrix"""
    
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='EfficientNetV2-S Training for DermNet Dataset')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, default='.', help='Root directory of the dataset')
    parser.add_argument('--processed_data_path', type=str, default='processed_data', help='Path to processed data directory')
    parser.add_argument('--output_dir', type=str, default='models', help='Directory to save models and results')

    # Model arguments
    parser.add_argument('--num_classes', type=int, default=23, help='Number of classes')
    parser.add_argument('--freeze_layers', type=int, default=150, help='Number of layers to freeze from the beginning')
    parser.add_argument('--input_size', type=int, default=384, help='Input image size')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate for cosine annealing')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs')

    # Other arguments
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loader workers')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')

    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.output_dir = args.output_dir + '/' + f'EfficientNetV2-S_{timestamp}'

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Checkpoint directory for loading existing models
    checkpoint_dir = 'models/EfficientNetV2-S'
    
    # Load data
    print("Loading data...")
    train_df, test_df, class_weights = load_data(args.data_root, args.processed_data_path)
    
    # Load category mapping
    with open(os.path.join(args.processed_data_path, 'category_mapping.json'), 'r') as f:
        category_mapping = json.load(f)
    class_names = category_mapping['categories']
    
    # Create datasets and data loaders
    print("Creating datasets...")
    train_transform = get_transforms(args.input_size, is_training=True)
    val_transform = get_transforms(args.input_size, is_training=False)
    
    train_dataset = DermNetDataset(train_df, args.data_root, transform=train_transform, is_training=True)
    test_dataset = DermNetDataset(test_df, args.data_root, transform=val_transform, is_training=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True)
    
    # Create model
    print("Creating model...")
    model = EfficientNetV2Classifier(num_classes=args.num_classes, freeze_layers=args.freeze_layers)
    model = model.to(device)
    
    # Print model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    
    # Create loss function with class weights
    class_weights_tensor = torch.tensor([class_weights[i] for i in range(args.num_classes)], 
                                       dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # Create optimizer (AdamW with weight decay)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Create learning rate scheduler with warmup and cosine annealing
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=total_steps,
        max_lr=args.learning_rate,
        min_lr=args.min_lr,
        warmup_steps=warmup_steps
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    # Load existing model weights if available
    best_model_path = os.path.join(checkpoint_dir, 'best_model_efficientnet_v2.pth')
    start_epoch = 0
    
    if os.path.exists(best_model_path):
        print("Loading existing model weights...")
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if available
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint['val_acc']
        print(f"Successfully loaded model from epoch {start_epoch} with validation accuracy {best_val_acc:.2f}%")
    else:
        print("No existing model found, starting from scratch")
        best_val_acc = 0.0
    
    best_epoch = 0
    patience_counter = 0
    
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 50)
        
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validation
        val_loss, val_acc = validate_epoch(model, test_loader, criterion, device, epoch)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'args': args
            }, os.path.join(args.output_dir, 'best_model_efficientnet_v2.pth'))
            print(f"New best model saved! Val Acc: {val_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            print("Early stopping triggered")
            break
        
        # Save checkpoint every N epochs
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'args': args
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}_efficientnet_v2.pth'))
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
        'args': args
    }, os.path.join(args.output_dir, 'final_model_efficientnet_v2.pth'))
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time/3600:.2f} hours")
    print(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
    
    # Save training history
    with open(os.path.join(args.output_dir, 'training_history_efficientnet_v2.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training history
    plot_training_history(history, os.path.join(args.output_dir, 'training_history_efficientnet_v2.png'))
    
    # Load best model for final evaluation
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model_efficientnet_v2.pth'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    print("Evaluating on test set...")
    test_results = evaluate_model(model, test_loader, device, args.num_classes)
    
    print(f"\nFinal Test Results:")
    print(f"Accuracy: {test_results['accuracy']:.4f}")
    print(f"Precision: {test_results['precision']:.4f}")
    print(f"Recall: {test_results['recall']:.4f}")
    print(f"F1-Score: {test_results['f1_score']:.4f}")
    
    # Save test results
    with open(os.path.join(args.output_dir, 'test_results_efficientnet_v2.json'), 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        test_results_json = {
            'accuracy': test_results['accuracy'],
            'precision': test_results['precision'],
            'recall': test_results['recall'],
            'f1_score': test_results['f1_score'],
            'confusion_matrix': test_results['confusion_matrix']
        }
        json.dump(test_results_json, f, indent=2)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        np.array(test_results['confusion_matrix']), 
        class_names, 
        os.path.join(args.output_dir, 'confusion_matrix_efficientnet_v2.png')
    )
    
    print(f"\nAll results saved to {args.output_dir}")
    print("\n=== Training completed! ===")


if __name__ == "__main__":
    main()
