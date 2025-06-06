# Chocolate Recognition Project

A computer vision project for detecting and classifying different types of chocolates using deep learning techniques. This project was developed as part of the IAPR 2025 (Image Analysis and Pattern Recognition) course.

## Overview

This project implements a two-stage approach for chocolate recognition:
1. **Segmentation**: Uses an Attention U-Net model to segment chocolate objects from images
2. **Classification**: Employs a feature extractor to classify the segmented chocolates into 13 different categories

## Chocolate Classes

The model can recognize 13 different types of chocolates:
- Amandina
- Arabia  
- Comtesse
- Crème brulée
- Jelly Black
- Jelly Milk
- Jelly White
- Noblesse
- Noir authentique
- Passion au lait
- Stracciatella
- Tentation noir
- Triangolo

## Project Structure

- `main.py` - Main inference script for generating submissions
- `check.py` - Validation and checking utilities
- `chocolate_detection.ipynb` - Main development notebook
- `report_Group2.ipynb` - Project report and analysis
- `presentation_IAPR2025.pdf` - Project presentation
- `my_submission.csv` - Generated submission file
- `src/` - Source code modules:
  - `model.py` - Neural network architectures (Attention U-Net, Feature Extractor)
  - `dataset.py` - Data loading and preprocessing
  - `pipeline.py` - Inference pipeline
  - `training.py` - Model training utilities
  - `loss.py` - Custom loss functions
  - `utils.py` - Helper utilities
  - `checkpoints/` - Trained model weights

## Usage

### Requirements
- Python 3.7+
- PyTorch
- OpenCV
- NumPy
- Pandas
- Other dependencies as specified in the notebooks

### Running Inference

```bash
python main.py --dataset_path /path/to/dataset --checkpoint_dir src/checkpoints --output_file submission.csv
```

### Arguments:
- `--dataset_path`: Path to the dataset directory containing test images
- `--checkpoint_dir`: Path to directory with trained model checkpoints
- `--output_file`: Name of the output CSV submission file (default: test_submission.csv)

## Model Architecture

- **Segmentation Model**: Attention U-Net with 3 input channels and 1 output class
- **Classification Model**: Custom Feature Extractor for multi-class chocolate recognition
- Both models use reproducible training with fixed random seeds for consistency

## Results

The project generates predictions in CSV format suitable for Kaggle submissions, with confidence scores for each chocolate class per image.

## Authors

EPFL - Master in Robotics

Mischa Mez,
Mathieu Sanchez,
Zaynab Hajroun,

Image Analysis and Pattern Recognition Course 2025 
