#!/usr/bin/env python3
"""
Model Evaluation Script for EfficientNetV2-S
Evaluates trained model on test set and generates performance reports
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import timm
from tqdm import tqdm


class DermNetDataset(Dataset):
    """Custom dataset for DermNet skin disease images"""
    
    def __init__(self, dataframe: pd.DataFrame, data_root: str, transform=None):
        self.dataframe = dataframe
        self.data_root = Path(data_root)
        self.transform = transform
        
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
        
        # Replace the classifier head with a custom one
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


def get_transforms(input_size: int = 384):
    """Get data transforms for evaluation"""
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


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
    plt.show()
    plt.close()


def find_latest_model_dir(base_dir: str = 'models') -> str:
    """Find the latest EfficientNetV2-S model directory"""
    model_dirs = []
    if os.path.exists(base_dir):
        for item in os.listdir(base_dir):
            if item.startswith('EfficientNetV2-S_') and os.path.isdir(os.path.join(base_dir, item)):
                model_dirs.append(item)
    
    if not model_dirs:
        raise ValueError(f"No EfficientNetV2-S model directories found in {base_dir}")
    
    # Sort by timestamp (newest first)
    model_dirs.sort(reverse=True)
    latest_dir = os.path.join(base_dir, model_dirs[0])
    print(f"Found latest model directory: {latest_dir}")
    return latest_dir


def main():
    parser = argparse.ArgumentParser(description='Evaluate EfficientNetV2-S Model')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, default='.', help='Root directory of the dataset')
    parser.add_argument('--processed_data_path', type=str, default='processed_data', help='Path to processed data directory')
    parser.add_argument('--model_dir', type=str, default=None, help='Directory containing the trained model (if None, will find latest)')
    parser.add_argument('--model_file', type=str, default='best_model_efficientnet_v2.pth', help='Model file name')
    
    # Model arguments
    parser.add_argument('--num_classes', type=int, default=23, help='Number of classes')
    parser.add_argument('--input_size', type=int, default=384, help='Input image size')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    
    # Other arguments
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loader workers')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cuda, cpu)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Find model directory if not specified
    if args.model_dir is None:
        args.model_dir = find_latest_model_dir()
    
    model_path = os.path.join(args.model_dir, args.model_file)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from: {model_path}")
    
    # Load test data
    print("Loading test data...")
    test_df = pd.read_csv(os.path.join(args.processed_data_path, 'test.csv'))
    print(f"Test samples: {len(test_df)}")
    
    # Load category mapping
    with open(os.path.join(args.processed_data_path, 'category_mapping.json'), 'r') as f:
        category_mapping = json.load(f)
    class_names = category_mapping['categories']
    print(f"Number of classes: {len(class_names)}")
    
    # Create test dataset and data loader
    test_transform = get_transforms(args.input_size)
    test_dataset = DermNetDataset(test_df, args.data_root, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True)
    
    # Create and load model
    print("Creating and loading model...")
    model = EfficientNetV2Classifier(num_classes=args.num_classes)
    
    # Load model weights with weights_only=False to handle PyTorch 2.6+ compatibility
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Model loaded successfully from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Previous validation accuracy: {checkpoint.get('val_acc', 'unknown'):.2f}%")
    
    # Evaluate model
    print("Evaluating model on test set...")
    test_results = evaluate_model(model, test_loader, device, args.num_classes)
    
    print(f"\n=== Final Test Results ===")
    print(f"Accuracy: {test_results['accuracy']:.4f} ({test_results['accuracy']*100:.2f}%)")
    print(f"Precision: {test_results['precision']:.4f}")
    print(f"Recall: {test_results['recall']:.4f}")
    print(f"F1-Score: {test_results['f1_score']:.4f}")
    
    # Save test results
    results_path = os.path.join(args.model_dir, 'test_results_efficientnet_v2.json')
    with open(results_path, 'w') as f:
        test_results_json = {
            'accuracy': test_results['accuracy'],
            'precision': test_results['precision'],
            'recall': test_results['recall'],
            'f1_score': test_results['f1_score'],
            'confusion_matrix': test_results['confusion_matrix']
        }
        json.dump(test_results_json, f, indent=2)
    print(f"Test results saved to: {results_path}")
    
    # Plot and save confusion matrix
    cm_path = os.path.join(args.model_dir, 'confusion_matrix_efficientnet_v2.png')
    plot_confusion_matrix(
        np.array(test_results['confusion_matrix']), 
        class_names, 
        cm_path
    )
    print(f"Confusion matrix saved to: {cm_path}")
    
    # Calculate and display per-class metrics
    print("\n=== Per-Class Performance ===")
    cm = np.array(test_results['confusion_matrix'])
    
    # Calculate per-class precision, recall, and F1
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []
    
    for i in range(len(class_names)):
        # True positives, false positives, false negatives
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1)
        
        print(f"{class_names[i][:30]:30} - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    
    # Save per-class results
    per_class_results = {
        'class_names': class_names,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class
    }
    
    per_class_path = os.path.join(args.model_dir, 'per_class_results.json')
    with open(per_class_path, 'w') as f:
        json.dump(per_class_results, f, indent=2)
    print(f"\nPer-class results saved to: {per_class_path}")
    
    print(f"\n=== Evaluation completed! ===")
    print(f"All results saved to: {args.model_dir}")


if __name__ == "__main__":
    main()
