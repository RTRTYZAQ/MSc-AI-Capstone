#!/usr/bin/env python3
"""
Single Image Inference Script for EfficientNetV2-S
Performs inference on a single image and returns top-k predictions with confidence scores
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm


class EfficientNetV2Classifier(nn.Module):
    """EfficientNetV2-M based classifier for skin disease classification"""
    
    def __init__(self, num_classes: int = 23, freeze_layers: int = 150):
        super(EfficientNetV2Classifier, self).__init__()
        
        # Load pre-trained EfficientNetV2-M using timm
        self.backbone = timm.create_model(
            "tf_efficientnetv2_m.in21k_ft_in1k",
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
    """Get data transforms for inference"""
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def predict_single_image(model: nn.Module, image_path: str, transform, device: torch.device, 
                        class_names: List[str], top_k: int = 5) -> Dict:
    """
    Predict a single image and return top-k predictions with confidence scores
    
    Args:
        model: Trained model
        image_path: Path to the image file
        transform: Image preprocessing transform
        device: Device to run inference on
        class_names: List of class names
        top_k: Number of top predictions to return
    
    Returns:
        Dictionary containing predictions and confidence scores
    """
    model.eval()
    
    # Load and preprocess image
    try:
        image = Image.open(image_path).convert('RGB')
        print(f"Image loaded successfully: {image.size}")
    except Exception as e:
        raise ValueError(f"Error loading image {image_path}: {e}")
    
    # Apply transforms and add batch dimension
    input_tensor = transform(image).unsqueeze(0).to(device)
    print(f"Input tensor shape: {input_tensor.shape}")
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0)  # Remove batch dimension
    
    # Get top-k predictions
    top_k = min(top_k, len(class_names))  # Don't exceed number of classes
    topk_probs, topk_indices = torch.topk(probs, top_k)
    
    # Convert to numpy and create results
    topk_probs = topk_probs.cpu().numpy()
    topk_indices = topk_indices.cpu().numpy()
    
    # Create confidence dictionary (for all top-k classes)
    confidence_dict = {}
    predictions_list = []
    
    for i, (idx, prob) in enumerate(zip(topk_indices, topk_probs)):
        class_name = class_names[idx]
        confidence = float(prob)
        
        confidence_dict[class_name] = confidence
        predictions_list.append({
            'rank': i + 1,
            'class_index': int(idx),
            'class_name': class_name,
            'confidence': confidence
        })
    
    # Create comprehensive results dictionary
    results = {
        'image_path': image_path,
        'image_size': list(image.size),
        'top_prediction': {
            'class_name': class_names[topk_indices[0]],
            'class_index': int(topk_indices[0]),
            'confidence': float(topk_probs[0])
        },
        f'top_{top_k}_predictions': predictions_list,
        'confidence_dict': confidence_dict,  # This is what you requested for further selection
        'inference_summary': {
            'most_confident_class': class_names[topk_indices[0]],
            'confidence_score': float(topk_probs[0]),
            'second_choice': class_names[topk_indices[1]] if len(topk_indices) > 1 else None,
            'second_confidence': float(topk_probs[1]) if len(topk_probs) > 1 else None,
            'confidence_gap': float(topk_probs[0] - topk_probs[1]) if len(topk_probs) > 1 else None
        }
    }
    
    return results


# def find_latest_model_dir(base_dir: str = 'models') -> str:
#     """Find the latest EfficientNetV2-S model directory"""
#     model_dirs = []
#     if os.path.exists(base_dir):
#         for item in os.listdir(base_dir):
#             if item.startswith('EfficientNetV2-S_') and os.path.isdir(os.path.join(base_dir, item)):
#                 model_dirs.append(item)
    
#     if not model_dirs:
#         raise ValueError(f"No EfficientNetV2-S model directories found in {base_dir}")
    
#     # Sort by timestamp (newest first)
#     model_dirs.sort(reverse=True)
#     latest_dir = os.path.join(base_dir, model_dirs[0])
#     print(f"Found latest model directory: {latest_dir}")
#     return latest_dir


# def main():
#     parser = argparse.ArgumentParser(description='Single Image Inference with EfficientNetV2-S')
    
#     # Required arguments
#     parser.add_argument('--image_path', type=str, help='Path to the image file for inference')
    
#     # Model arguments
#     parser.add_argument('--model_dir', type=str, default=None, help='Directory containing the trained model (if None, will find latest)')
#     parser.add_argument('--model_file', type=str, default='best_model_efficientnet_v2.pth', help='Model file name')
#     parser.add_argument('--processed_data_path', type=str, default='processed_data', help='Path to processed data directory (for class names)')
    
#     # Inference arguments
#     parser.add_argument('--num_classes', type=int, default=23, help='Number of classes')
#     parser.add_argument('--input_size', type=int, default=384, help='Input image size')
#     parser.add_argument('--top_k', type=int, default=5, help='Number of top predictions to return')
#     parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cuda, cpu)')
    
#     # Output arguments
#     parser.add_argument('--save_result', type=str, default=None, help='Path to save prediction result (JSON file)')
#     parser.add_argument('--verbose', action='store_true', help='Show detailed output')
    
#     args = parser.parse_args()
    
#     # Set device
#     if args.device == 'auto':
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     else:
#         device = torch.device(args.device)
    
#     print(f"Using device: {device}")
    
#     # Check if image exists
#     if not os.path.exists(args.image_path):
#         raise FileNotFoundError(f"Image file not found: {args.image_path}")
    
#     # Find model directory if not specified
#     if args.model_dir is None:
#         args.model_dir = find_latest_model_dir()
    
#     model_path = os.path.join(args.model_dir, args.model_file)
    
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model file not found: {model_path}")
    
#     print(f"Loading model from: {model_path}")
    
#     # Load category mapping
#     category_mapping_path = os.path.join(args.processed_data_path, 'category_mapping.json')
#     if not os.path.exists(category_mapping_path):
#         raise FileNotFoundError(f"Category mapping not found: {category_mapping_path}")
    
#     with open(category_mapping_path, 'r') as f:
#         category_mapping = json.load(f)
#     class_names = category_mapping['categories']
#     print(f"Loaded {len(class_names)} class names")
    
#     # Create model and load weights
#     print("Creating and loading model...")
#     model = EfficientNetV2Classifier(num_classes=args.num_classes)
    
#     # Load model weights with weights_only=False to handle PyTorch 2.6+ compatibility
#     checkpoint = torch.load(model_path, map_location=device, weights_only=False)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model = model.to(device)
    
#     if args.verbose:
#         print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
#         print(f"Previous validation accuracy: {checkpoint.get('val_acc', 'unknown'):.2f}%")
    
#     # Create transform
#     transform = get_transforms(args.input_size)
    
#     # Perform inference
#     print(f"\n=== Performing inference on: {args.image_path} ===")
    
#     try:
#         result = predict_single_image(
#             model, args.image_path, transform, device, 
#             class_names, args.top_k
#         )
        
#         # Display results
#         print(f"\n🎯 Top Prediction:")
#         print(f"   Class: {result['top_prediction']['class_name']}")
#         print(f"   Confidence: {result['top_prediction']['confidence']:.4f} ({result['top_prediction']['confidence']*100:.2f}%)")
        
#         print(f"\n📊 Top-{args.top_k} Predictions:")
#         for pred in result[f'top_{args.top_k}_predictions']:
#             print(f"   {pred['rank']}. {pred['class_name']}: {pred['confidence']:.4f} ({pred['confidence']*100:.2f}%)")
        
#         print(f"\n📋 Confidence Dictionary (for further selection):")
#         for class_name, confidence in result['confidence_dict'].items():
#             print(f"   '{class_name}': {confidence:.4f}")
        
#         # Show inference summary
#         summary = result['inference_summary']
#         print(f"\n📈 Inference Summary:")
#         print(f"   Most Confident: {summary['most_confident_class']} ({summary['confidence_score']*100:.2f}%)")
#         if summary['second_choice']:
#             print(f"   Second Choice: {summary['second_choice']} ({summary['second_confidence']*100:.2f}%)")
#             print(f"   Confidence Gap: {summary['confidence_gap']*100:.2f}%")
        
#         # Save results if requested
#         if args.save_result:
#             with open(args.save_result, 'w') as f:
#                 json.dump(result, f, indent=2)
#             print(f"\n💾 Results saved to: {args.save_result}")
        
#         print(f"\n✅ Inference completed successfully!")
        
#         # Return the confidence dictionary for programmatic use
#         return result['confidence_dict']
        
#     except Exception as e:
#         print(f"❌ Error during inference: {e}")
#         return None


# if __name__ == "__main__":
#     main()
