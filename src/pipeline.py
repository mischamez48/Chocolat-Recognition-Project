import torch
import torch.nn.functional as F
import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from skimage.measure import regionprops
from tqdm import tqdm
import pandas as pd
import os

def separate_chocolates_watershed(mask, min_distance=10, min_size=300):
    """
    Separate connected chocolates using watershed segmentation

    Args:
        mask: Binary mask (numpy array or torch tensor)
        min_distance: Minimum distance between peaks (controls separation sensitivity)
        min_size: Minimum size of objects to keep

    Returns:
        Separated mask with individual chocolates labeled
    """
    # Convert to numpy if tensor
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    # Ensure mask is binary
    if mask.max() <= 1.0:
        mask_binary = (mask > 0.5).astype(bool)
    else:
        mask_binary = (mask > 127).astype(bool)

    # Compute the distance transform: each pixel contains distance to nearest background
    distance = ndi.distance_transform_edt(mask_binary)

    # Find local maxima (centers of chocolates)
    # min_distance controls how far apart chocolates must be
    coords = peak_local_max(distance, min_distance=min_distance, labels=mask_binary)

    # Create markers for watershed
    markers = np.zeros_like(mask_binary, dtype=int)
    markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)

    # Apply watershed segmentation
    labels = watershed(-distance, markers, mask=mask_binary)

    # Remove small objects if needed
    if min_size > 0:
        for i in range(1, labels.max() + 1):
            if np.sum(labels == i) < min_size:
                labels[labels == i] = 0

    return labels

def extract_chocolate_regions_improved(image, segmentation_model, device, min_distance=15, min_size=5):
    """
    Improved function to extract individual chocolate regions from an image

    Args:
        image: Input image tensor [C, H, W]
        segmentation_model: Binary segmentation model
        device: Device to run inference on
        min_distance: Minimum distance for watershed separation
        min_size: Minimum size of chocolate regions

    Returns:
        List of extracted chocolate regions with their bounding boxes
    """
    # Add batch dimension
    image_batch = image.unsqueeze(0).to(device)

    # Get segmentation mask using the UNet model
    with torch.no_grad():
        mask_logits = segmentation_model(image_batch)
        pred_mask = (torch.sigmoid(mask_logits) > 0.5).float().squeeze().cpu().numpy()

    # Apply watershed segmentation to separate connected chocolates
    labeled_mask = separate_chocolates_watershed(
        mask=pred_mask,
        min_distance=min_distance,
        min_size=min_size
    )

    # Extract each chocolate region
    regions = []
    original_image = image.cpu()

    # For each labeled region
    for label_idx in range(1, labeled_mask.max() + 1):
        # Get binary mask for this region
        region_mask = (labeled_mask == label_idx)

        # Find bounding box
        props = regionprops(region_mask.astype(int))
        if not props:
            continue

        y1, x1, y2, x2 = props[0].bbox

        # Calculate center of mass and area
        center_y, center_x = props[0].centroid
        area = props[0].area

        # Ensure minimum size and expand slightly
        h, w = (y2 - y1), (x2 - x1)

        # Skip extremely small regions
        if h < 10 or w < 10:
            continue

        # Use balanced padding based on aspect ratio
        aspect_ratio = w / h
        if aspect_ratio > 1.5:
            # Wide chocolate - add more padding vertically
            padding_y = max(5, int(h * 0.2))
            padding_x = max(5, int(w * 0.1))
        elif aspect_ratio < 0.67:
            # Tall chocolate - add more padding horizontally
            padding_y = max(5, int(h * 0.1))
            padding_x = max(5, int(w * 0.2))
        else:
            # Balanced chocolate - add equal padding
            padding_y = max(5, int(h * 0.15))
            padding_x = max(5, int(w * 0.15))

        # Add padding and clip to image boundaries
        y1 = max(0, y1 - padding_y)
        x1 = max(0, x1 - padding_x)
        y2 = min(original_image.shape[1], y2 + padding_y)
        x2 = min(original_image.shape[2], x2 + padding_x)

        # Crop region from original image
        region = original_image[:, y1:y2, x1:x2]

        # Skip if dimension is zero (rare edge case)
        if region.shape[1] == 0 or region.shape[2] == 0:
            continue

        # Resize to match training size (224x224)
        region = F.interpolate(
            region.unsqueeze(0),
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        regions.append({
            'image': region,
            'bbox': [x1, y1, x2, y2],
            'mask': region_mask,
            'area': area,
            'center': (center_x, center_y)
        })

    return regions

def chocolate_detection_pipeline(image_tensor, segmentation_model, feature_model, class_names, device):
    """
    Complete chocolate detection pipeline that:
    1) Segments chocolates from background
    2) Separates connected chocolates using watershed
    3) Classifies each chocolate
    4) Returns counts for each chocolate type

    Args:
        image_tensor: Input image tensor [C, H, W]
        segmentation_model: Binary segmentation model
        feature_model: Feature extraction and classification model
        class_names: List of chocolate class names
        device: Device to run inference on

    Returns:
        counts: Dictionary mapping chocolate types to counts
        visualization_data: Dict with data for visualization
    """
    # Step 1: Extract chocolate regions using segmentation model
    chocolate_regions = extract_chocolate_regions_improved(
        image=image_tensor,
        segmentation_model=segmentation_model,
        device=device,
        min_distance=10,  # Parameter tuned based on validation experiments
        min_size=300
    )

    # Initialize counts for each class
    counts = {class_name: 0 for class_name in class_names}

    # Collect visualization data
    vis_data = {
        'original_image': image_tensor,
        'regions': chocolate_regions,
        'classes': [],
        'scores': []
    }

    # If no chocolates detected, return zeros
    if len(chocolate_regions) == 0:
        return counts, vis_data

    # Step 2: Classify each region
    feature_model.eval()
    with torch.no_grad():
        for region in chocolate_regions:
            # Get the chocolate image
            choc_img = region['image']

            # Add batch dimension
            choc_img = choc_img.unsqueeze(0).to(device)

            # Get classification prediction
            logits, _ = feature_model(choc_img)
            probabilities = F.softmax(logits, dim=1)

            # Get the class with highest probability
            score, class_idx = torch.max(probabilities, dim=1)
            class_name = class_names[class_idx.item()]

            # Increment count for this class
            counts[class_name] += 1

            # Store for visualization
            vis_data['classes'].append(class_name)
            vis_data['scores'].append(score.item())

    return counts, vis_data

def convert_to_submission_format(test_results, class_names):
    """
    Convert test results to submission format with corrected column names

    Args:
        test_results: Dictionary with test results
        class_names: List of chocolate class names

    Returns:
        Pandas DataFrame in submission format
    """
    # Load sample submission to get exact column names
    sample_submission = pd.read_csv('dataset_project_iapr2025/sample_submission.csv')
    expected_columns = list(sample_submission.columns)

    # Create submission dataframe with ID column
    submission_data = {'id': []}

    # Add columns for each class, using the exact names from sample submission
    class_columns = expected_columns[1:]  # Skip 'id'
    for col in class_columns:
        submission_data[col] = []

    # Create mapping from our class names to expected column names
    # The issue is with "Crème brulée" vs "Creme brulee"
    name_mapping = {}
    for our_name in class_names:
        for expected_name in class_columns:
            # Check if names match when accents are removed
            if our_name.lower().replace('è', 'e').replace('é', 'e') == expected_name.lower().replace('è', 'e').replace('é', 'e'):
                name_mapping[our_name] = expected_name
                break

        # If no match found, use the original name
        if our_name not in name_mapping:
            name_mapping[our_name] = our_name

    # Fill data
    for img_id, result in test_results.items():
        submission_data['id'].append(int(img_id))

        # Initialize all classes to 0
        counts = {col: 0 for col in class_columns}

        # Add detected counts
        for class_name, count in result['counts'].items():
            # Map our class name to expected column name
            column_name = name_mapping.get(class_name, class_name)

            # Extra safety check
            if column_name in counts:
                counts[column_name] = int(count)
            else:
                print(f"Warning: Class {class_name} maps to {column_name} which is not in expected columns")

        # Add all counts to submission
        for col in class_columns:
            submission_data[col].append(counts[col])

    # Create DataFrame
    submission_df = pd.DataFrame(submission_data)

    # Sort by ID for consistency
    submission_df = submission_df.sort_values('id').reset_index(drop=True)

    return submission_df

def run_test_inference(test_dataset, segmentation_model, feature_model, class_names, device, show_visualizations=True):
    """
    Run inference on test dataset

    Args:
        test_dataset: Test dataset
        segmentation_model: Binary segmentation model
        feature_model: Feature extraction and classification model
        class_names: List of chocolate class names
        device: Device to run inference on
        show_visualizations: Whether to show visualization plots (default: True)

    Returns:
        Dictionary with results for each image
    """

    from src.utils import visualize_pipeline_results

    # Set models to evaluation mode
    segmentation_model.eval()
    feature_model.eval()

    test_results = {}

    print(f"Running inference on {len(test_dataset)} test images...")

    for idx in tqdm(range(len(test_dataset))):
        # Get test image and ID
        image, img_id = test_dataset[idx]

        # Run the pipeline
        counts, vis_data = chocolate_detection_pipeline(
            image_tensor=image,
            segmentation_model=segmentation_model,
            feature_model=feature_model,
            class_names=class_names,
            device=device
        )

        # Store results
        test_results[img_id] = {
            'counts': counts,
            'num_detected': len(vis_data['regions']),
            'vis_data': vis_data
        }

        # Optional: Visualize some examples (e.g. every 36th)
        if show_visualizations and idx % 36 == 0:  # Added check for show_visualizations
            visualize_pipeline_results(image, counts, vis_data)

    print(f"Completed inference on {len(test_results)} test images")
    return test_results

def save_submission_file(submission_df, filename='submission.csv'):
    """
    Save submission DataFrame to CSV file

    Args:
        submission_df: DataFrame in submission format
        filename: Output filename
    """
    # Ensure all values are integers
    for col in submission_df.columns:
        if col != 'id':
            submission_df[col] = submission_df[col].astype(int)

    # Save to CSV
    submission_df.to_csv(filename, index=False)
    print(f"Saved submission to {filename}")

    # Verify file exists and is readable
    try:
        test_read = pd.read_csv(filename)
        print(f"Verified file can be read successfully")
        print(f"  - File size: {os.path.getsize(filename) / 1024:.2f} KB")
        print(f"  - Row count: {len(test_read)}")
    except Exception as e:
        print(f"Error verifying file: {e}")