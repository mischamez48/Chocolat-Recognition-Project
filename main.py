# main.py - Executable script for IAPR 2025 Final Project: Chocolate Detector
# Produces the exact submission file uploaded to Kaggle

import os
import argparse
import torch
import numpy as np
import random
import pandas as pd

# Import custom modules
from src.dataset import ChocolateTestDataset, get_segmentation_transforms
from src.model import AttentionUNet, FeatureExtractor
from src.pipeline import run_test_inference, convert_to_submission_format, save_submission_file

def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed} for reproducibility")

def main():
    parser = argparse.ArgumentParser(description="Chocolate Detector Inference Script")
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Path to model checkpoints')
    parser.add_argument('--output_file', type=str, default='test_submission.csv', help='Name of output submission file')
    args = parser.parse_args()

    print("IAPR 2025 Final Project: Chocolate Detector")
    print("Running inference pipeline to produce submission file...")

    # Set random seed
    set_seed(42)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    TEST_IMG_PATH = os.path.join(args.dataset_path, 'test')

    # Class names in training order
    class_names = [
        'Amandina', 'Arabia', 'Comtesse', 'Crème brulée', 'Jelly Black',
        'Jelly Milk', 'Jelly White', 'Noblesse', 'Noir authentique',
        'Passion au lait', 'Stracciatella', 'Tentation noir', 'Triangolo'
    ]
    print(f"Using {len(class_names)} classes")

    # Load segmentation model
    attention_unet_model = AttentionUNet(n_channels=3, n_classes=1, base_filters=32).to(device)
    checkpoint = torch.load(
        os.path.join(args.checkpoint_dir, 'attention_unet_model3.pth'),
        map_location=device
    )
    attention_unet_model.load_state_dict(checkpoint['model_state_dict'])
    attention_unet_model.eval()
    print(f"Loaded segmentation model: {attention_unet_model.count_parameters():,} parameters")

    # Load feature extractor model
    feature_extractor = FeatureExtractor(num_classes=len(class_names), class_names=class_names).to(device)
    feature_extractor.load_state_dict(
        torch.load(
            os.path.join(args.checkpoint_dir, 'feature_extractor_best_f1.pth'),
            map_location=device
        )
    )
    feature_extractor.eval()
    print(f"Loaded feature extractor model: {feature_extractor.count_parameters():,} parameters")

    # Create test dataset
    val_transform = get_segmentation_transforms(train=False)
    test_dataset = ChocolateTestDataset(TEST_IMG_PATH, transform=val_transform)
    print(f"Created test dataset with {len(test_dataset)} images")

    # Inference
    print("Running inference on test set...")
    test_results = run_test_inference(
        test_dataset=test_dataset,
        segmentation_model=attention_unet_model,
        feature_model=feature_extractor,
        class_names=class_names,
        device=device,
        show_visualizations=False
    )

    # Convert to submission
    print("Converting results to submission format...")
    submission_df = convert_to_submission_format(test_results, class_names)

    # Verify column names
    sample_submission = pd.read_csv(os.path.join(args.dataset_path, 'sample_submission.csv'))
    print("\nVerifying column names:")
    print(f"Sample submission columns: {list(sample_submission.columns)}")
    print(f"Our submission columns: {list(submission_df.columns)}")

    # Save submission
    print(f"Saving submission to {args.output_file}...")
    save_submission_file(submission_df, args.output_file)

    total_predictions = submission_df.iloc[:, 1:].sum().sum()
    print(f"Submission contains predictions for {len(submission_df)} images with {total_predictions} total chocolates")
    print("Done! Submission file has been created.")

if __name__ == "__main__":
    main()
