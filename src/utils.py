import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import label
from scipy import ndimage as ndi
from torchvision.utils import draw_bounding_boxes
import random
from torchvision.transforms import functional as TF
import pandas as pd
from src.pipeline import extract_chocolate_regions_improved

def inverse_transform(tensor, mean=[0.6841, 0.6594, 0.6527], std=[0.1504, 0.1537, 0.1760]):
    """
    Convert normalized tensor to image for visualization.
    
    Args:
        tensor: PyTorch tensor of shape [C, H, W]
        mean: Mean used for normalization
        std: Std used for normalization
        
    Returns:
        numpy array of shape [H, W, C] in range [0, 1]
    """
    # Convert to CPU if on GPU
    if tensor.is_cuda:
        tensor = tensor.cpu()
        
    # Convert to numpy and transpose
    img = tensor.clone().detach().numpy()
    
    # Inverse of normalization
    mean = np.array(mean).reshape(3, 1, 1)
    std = np.array(std).reshape(3, 1, 1)
    img = img * std + mean
    
    # Clamp values to [0, 1]
    img = np.clip(img, 0, 1)
    
    # Transpose from [C, H, W] to [H, W, C]
    img = img.transpose(1, 2, 0)
    
    return img

def visualize_segmentation_sample(dataset, idx=None):
    """
    Visualize a single sample from the dataset

    Args:
        dataset: The segmentation dataset
        idx: Index of the sample to visualize (random if None)
    """
    if idx is None:
        idx = random.randint(0, len(dataset)-1)

    # Get a single sample (no batching)
    image, target = dataset[idx]

    # Display image shape to confirm rectangular dimensions
    print(f"Image shape: {image.shape}")  # Should show something like [3, 224, 336]

    # Denormalize the image
    mean = torch.tensor([0.6841, 0.6594, 0.6527])
    std = torch.tensor([0.1504, 0.1537, 0.1760])
    image_denorm = image * std[:, None, None] + mean[:, None, None]

    # Convert to uint8 for visualization
    image_uint8 = (image_denorm * 255).to(torch.uint8)

    # Get target components
    boxes = target['boxes']
    labels = target['labels']
    masks = target['masks'].bool()

    # Create class labels and colors
    class_labels = [dataset.class_names[label.item()] for label in labels]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
              (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
              (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
              (192, 92, 0)]

    # Draw bounding boxes
    result_with_boxes = draw_bounding_boxes(
        image_uint8,
        boxes=boxes,
        labels=class_labels,
        colors=colors,
        width=2
    )

    # Display original and boxed images
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    axes[0].imshow(TF.to_pil_image(image_denorm))
    axes[0].set_title(f"Original Image {image.shape[-2]}x{image.shape[-1]}")
    axes[0].axis('off')

    axes[1].imshow(TF.to_pil_image(result_with_boxes))
    axes[1].set_title("Bounding Boxes")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

    # Also display masks individually
    if masks.shape[0] > 0:
        num_masks = masks.shape[0]
        fig, axes = plt.subplots(1, num_masks, figsize=(4*num_masks, 4))

        # Handle case with only one mask
        if num_masks == 1:
            axes = [axes]

        for i, (mask, label) in enumerate(zip(masks, labels)):
            class_name = dataset.class_names[label.item()]
            axes[i].imshow(mask, cmap='gray')
            axes[i].set_title(f"Mask: {class_name}")
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

def get_binary_seg_backgrounds(binary_seg_dataset, id_to_background):
    binary_seg_backgrounds = []

    for img_id in binary_seg_dataset.valid_image_ids:
        try:
            # Handle string IDs that start with 'L'
            if isinstance(img_id, str) and img_id.startswith('L'):
                base_id = img_id.split('_')[0][1:]  # e.g., '1000756' from 'L1000756_1'
                numeric_id = int(base_id)
            else:
                numeric_id = int(img_id)

            # Append background or "unknown" if missing
            background = id_to_background.get(numeric_id, "unknown")
            binary_seg_backgrounds.append(background)
        except:
            binary_seg_backgrounds.append("unknown")

    return binary_seg_backgrounds

def show_background_distribution(binary_seg_train_indices, binary_seg_val_indices, binary_seg_backgrounds):
    def count_distribution(indices):
        counts = {}
        for idx in indices:
            bg = binary_seg_backgrounds[idx]
            counts[bg] = counts.get(bg, 0) + 1
        return counts

    train_counts = count_distribution(binary_seg_train_indices)
    val_counts = count_distribution(binary_seg_val_indices)
    total_train = len(binary_seg_train_indices)
    total_val = len(binary_seg_val_indices)

    # Sort consistently
    all_bgs = sorted(set(binary_seg_backgrounds), key=lambda bg: -train_counts.get(bg, 0))

    print("\nBackground distribution in binary segmentation training set:")
    for bg in all_bgs:
        count = train_counts.get(bg, 0)
        print(f"  - {bg}: {count} ({count/total_train*100:.1f}%)")

    print("\nBackground distribution in binary segmentation validation set:")
    for bg in all_bgs:
        count = val_counts.get(bg, 0)
        print(f"  - {bg}: {count} ({count/total_val*100:.1f}%)")

    # Plotting
    df = pd.DataFrame({
        'Train': {bg: train_counts.get(bg, 0) / total_train * 100 for bg in all_bgs},
        'Validation': {bg: val_counts.get(bg, 0) / total_val * 100 for bg in all_bgs}
    })

    df.plot(kind='bar', figsize=(6, 4), rot=45)
    plt.title("Background Distribution (%)")
    plt.ylabel("Percentage")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def visualize_binary_samples(dataset, num_samples=3):
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    for idx in indices:
        image, mask = dataset[idx]

        # Convert to numpy for visualization
        image_np = inverse_transform(image)
        mask_np = mask.squeeze().numpy()

        # Visualize
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        ax[0].imshow(image_np)
        ax[0].set_title('Image')
        ax[0].axis('off')

        ax[1].imshow(mask_np, cmap='gray')
        ax[1].set_title('Binary Mask (All Chocolates)')
        ax[1].axis('off')

        plt.tight_layout()
        plt.show()

def visualize_predictions(model, dataset, device, num_samples=5):
    """
    Visualize model predictions on random samples, handling rectangular images
    """
    # Set model to evaluation mode
    model.eval()

    # Randomly select samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    with torch.no_grad():
        for idx in indices:
            # Get sample
            image, true_mask = dataset[idx]

            # Add batch dimension and move to device
            image_batch = image.unsqueeze(0).to(device)

            # Get prediction
            output = model(image_batch)

            # Resize output to match input dimensions if they differ
            if output.shape[-2:] != image.shape[-2:]:
                output = F.interpolate(
                    output,
                    size=image.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )

            # Get prediction probability and binary mask
            pred_prob = torch.sigmoid(output).squeeze().cpu()
            pred_mask = (pred_prob > 0.5).float()

            # Convert to numpy for visualization (maintain original dimensions)
            image_np = inverse_transform(image)
            true_mask_np = true_mask.squeeze().numpy()
            pred_prob_np = pred_prob.numpy()
            pred_mask_np = pred_mask.numpy()

            # Visualize
            fig, ax = plt.subplots(1, 4, figsize=(20, 5))

            ax[0].imshow(image_np)
            ax[0].set_title(f'Original Image {image_np.shape[0]}x{image_np.shape[1]}')
            ax[0].axis('off')

            ax[1].imshow(true_mask_np, cmap='gray')
            ax[1].set_title(f'True Mask {true_mask_np.shape[0]}x{true_mask_np.shape[1]}')
            ax[1].axis('off')

            # For heatmap visualization
            ax[2].imshow(pred_prob_np, cmap='jet')
            ax[2].set_title(f'Prediction Probability {pred_prob_np.shape[0]}x{pred_prob_np.shape[1]}')
            ax[2].axis('off')

            ax[3].imshow(pred_mask_np, cmap='gray')
            ax[3].set_title(f'Binary Prediction {pred_mask_np.shape[0]}x{pred_mask_np.shape[1]}')
            ax[3].axis('off')

            plt.tight_layout()
            plt.show()

def separate_chocolates_watershed(mask, min_distance=15, min_size=50):
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

def visualize_watershed_separation(original_mask, watershed_labels):
    """
    Visualize the watershed separation results

    Args:
        original_mask: Original binary mask
        watershed_labels: Labeled mask from watershed segmentation
    """
    # Create colored visualization of labels
    def create_colored_labels(labeled_mask):
        colored_mask = np.zeros((labeled_mask.shape[0], labeled_mask.shape[1], 3), dtype=np.uint8)
        for i in range(1, labeled_mask.max() + 1):
            # Generate a unique color for each label
            color = np.random.randint(50, 255, size=3)
            colored_mask[labeled_mask == i] = color
        return colored_mask

    # Create visualizations
    colored_labels = create_colored_labels(watershed_labels)

    # Set up the figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original mask
    axes[0].imshow(original_mask, cmap='gray')
    axes[0].set_title(f'Original Mask\n({label(original_mask).max()} components)')
    axes[0].axis('off')

    # Watershed separation
    axes[1].imshow(watershed_labels, cmap='nipy_spectral')
    axes[1].set_title(f'Watershed Separation\n({watershed_labels.max()} components)')
    axes[1].axis('off')

    # Colored components
    axes[2].imshow(colored_labels)
    axes[2].set_title('Separated Chocolates')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

def process_and_visualize_with_watershed(image, segmentation_model, device, min_distance=10, min_size=50, crop_padding=5):
    """
    Process an image, extract chocolate regions with watershed, and visualize with tighter crops

    Args:
        image: Image tensor
        segmentation_model: Segmentation model
        device: Device to run on
        min_distance: Watershed min_distance parameter
        min_size: Minimum component size (reduced to detect smaller chocolates)
        crop_padding: Padding for chocolate crops (smaller = tighter crops)
    """
    # Get segmentation mask
    with torch.no_grad():
        image_batch = image.unsqueeze(0).to(device)
        output = segmentation_model(image_batch)

        # Make sure output is resized to match input dimensions
        if output.shape[-2:] != image.shape[-2:]:
            output = F.interpolate(
                output,
                size=image.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        pred_mask = (torch.sigmoid(output) > 0.5).float().squeeze().cpu().numpy()

    # Apply watershed segmentation
    watershed_labels = separate_chocolates_watershed(
        mask=pred_mask,
        min_distance=min_distance,
        min_size=min_size
    )

    print(f"Watershed found {watershed_labels.max()} chocolate components")

    # Extract chocolate regions
    regions = extract_chocolate_regions_improved(
        image=image,
        segmentation_model=segmentation_model,
        device=device,
        min_distance=min_distance,
        min_size=min_size
    )

    print(f"Extracted {len(regions)} chocolate regions")

    # Visualize with tighter crops and watershed mask
    visualize_chocolate_extraction(
        original_image=image,
        binary_mask=pred_mask,
        watershed_mask=watershed_labels,
        extraction_results=regions,
        crop_padding=crop_padding  # Use smaller padding for tighter crops
    ) 

def visualize_chocolate_extraction(original_image, binary_mask, watershed_mask, extraction_results, crop_padding=5):
    """
    Enhanced visualization function to show all extracted chocolate regions
    
    Args:
        original_image: Original image (tensor or numpy array)
        binary_mask: Binary mask (tensor or numpy array)
        watershed_mask: Watershed segmentation mask (tensor or numpy array)
        extraction_results: Results from extract_chocolate_regions
        crop_padding: Padding for chocolate crops (smaller = tighter crops)
    """
    # Convert original image if tensor
    if isinstance(original_image, torch.Tensor):
        if original_image.shape[0] == 3:  # [C, H, W]
            original_image = inverse_transform(original_image)
        else:
            raise ValueError("Expected image shape [3, H, W] for tensor input.")
    elif original_image.max() > 1.0:
        original_image = original_image.astype(np.uint8)
    else:
        original_image = np.clip(original_image, 0, 1)

    # Convert masks if tensors
    if isinstance(binary_mask, torch.Tensor):
        binary_mask = binary_mask.cpu().numpy()
    
    if isinstance(watershed_mask, torch.Tensor):
        watershed_mask = watershed_mask.cpu().numpy()

    num_chocolates = len(extraction_results)
    
    # Create figure with appropriate size
    plt.figure(figsize=(20, 4))

    # Original image
    plt.subplot(1, 4, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')

    # Binary mask
    plt.subplot(1, 4, 2)
    plt.imshow(binary_mask, cmap='gray')
    plt.title('Binary Mask')
    plt.axis('off')
    
    # Watershed mask
    plt.subplot(1, 4, 3)
    plt.imshow(watershed_mask, cmap='nipy_spectral')
    plt.title(f'Watershed Mask\n({np.max(watershed_mask)} components)')
    plt.axis('off')

    # Bounding boxes
    plt.subplot(1, 4, 4)
    plt.imshow(original_image)
    for chocolate in extraction_results:
        x_min, y_min, x_max, y_max = chocolate['bbox']
        plt.gca().add_patch(plt.Rectangle((x_min, y_min),
                                        x_max - x_min,
                                        y_max - y_min,
                                        fill=False,
                                        edgecolor='r',
                                        linewidth=2))
    plt.title(f'Detected Chocolates: {num_chocolates}')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Calculate grid size for displaying the extracted chocolates
    if num_chocolates > 0:
        cols = min(5, num_chocolates)
        rows = (num_chocolates + cols - 1) // cols
        
        # Create a new figure for the chocolate regions
        plt.figure(figsize=(4*cols, 4*rows))
        
        # Display each extracted chocolate
        for i, choco_result in enumerate(extraction_results):
            plt.subplot(rows, cols, i+1)
            
            # Get the region
            if isinstance(choco_result['image'], torch.Tensor):
                region_img = inverse_transform(choco_result['image'])
            else:
                region_img = choco_result['image']
                
            plt.imshow(region_img)
            plt.title(f'Chocolate {i+1}')
            plt.axis('off')
            
        plt.tight_layout()
        plt.show()
    else:
        print("No chocolate regions extracted!")

def visualize_aligned_dataset(region_images, correct_labels, class_names=None,
                             samples_per_class=5, figsize=(15, 15), seed=None):
    """
    Visualize the aligned training dataset with samples organized by class.

    Args:
        region_images: List of chocolate region image tensors
        correct_labels: List of corresponding class labels
        class_names: Optional list of class names (if None, use unique labels)
        samples_per_class: Number of samples to display per class
        figsize: Figure size for the plot
        seed: Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)

    # Get unique class names if not provided
    if class_names is None:
        class_names = sorted(list(set(correct_labels)))

    # Group images by class using plain dict
    class_to_images = {}
    for img, label in zip(region_images, correct_labels):
        if label not in class_to_images:
            class_to_images[label] = []
        class_to_images[label].append(img)

    # Count class distribution manually
    class_counts = {}
    for label in correct_labels:
        class_counts[label] = class_counts.get(label, 0) + 1

    total_images = len(correct_labels)
    num_classes = len(class_names)
    cols = min(5, samples_per_class)
    rows = num_classes

    fig = plt.figure(figsize=figsize)

    for i, class_name in enumerate(class_names):
        class_images = class_to_images.get(class_name, [])
        count = len(class_images)

        if count == 0:
            continue

        samples = min(samples_per_class, count)
        indices = random.sample(range(count), samples)
        selected_images = [class_images[j] for j in indices]

        for j, img in enumerate(selected_images):
            ax = fig.add_subplot(rows, cols, i * cols + j + 1)

            if isinstance(img, torch.Tensor):
                h, w = img.shape[1], img.shape[2]
                aspect = w / h
                img_np = inverse_transform(img)
                ax.imshow(img_np)

                if j == 0:
                    ax.set_title(f"{class_name}\n({count} samples, {count/total_images:.1%})\nAR: {aspect:.2f}")
                else:
                    ax.set_title(f"AR: {aspect:.2f}")
            else:
                ax.imshow(img)
                if j == 0:
                    ax.set_title(f"{class_name}\n({count} samples, {count/total_images:.1%})")

            ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.suptitle(f"Aligned Dataset: {total_images} Chocolate Samples", fontsize=16)
    plt.show()

    # Plot class distribution
    plt.figure(figsize=(12, 6))
    counts = [class_counts.get(cls, 0) for cls in class_names]
    plt.bar(class_names, counts)
    plt.xlabel('Chocolate Class')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution in Aligned Dataset')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    print(f"Total samples: {total_images}")
    print("\nClass distribution:")
    for cls in class_names:
        count = class_counts.get(cls, 0)
        print(f"  - {cls}: {count} samples ({count/total_images:.1%})")

def visualize_feature_extractor_batch(dataloader, class_names, num_images=8):
    """
    Visualize a batch of images from the feature extractor's dataloader

    Args:
        dataloader: DataLoader for feature extractor
        class_names: List of class names
        num_images: Maximum number of images to display
    """
    # Get a batch
    images, labels = next(iter(dataloader))

    # Limit the number of images to display
    num_images = min(num_images, len(images))

    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_images)))

    # Create figure
    plt.figure(figsize=(6, 6))

    for i in range(num_images):
        # Get image and label
        image = images[i]
        label = labels[i]
        class_name = class_names[label]

        # Calculate aspect ratio
        h, w = image.shape[1], image.shape[2]
        aspect_ratio = w/h

        # Convert tensor to numpy for visualization
        image_np = inverse_transform(image)

        # Display image
        plt.subplot(grid_size, grid_size, i+1)
        plt.imshow(image_np)
        plt.title(f"{class_name}\nShape: {h}x{w}\nAR: {aspect_ratio:.2f}")
        plt.axis('off')

    plt.tight_layout()
    plt.suptitle("Feature Extractor Input Samples", fontsize=16, y=1.05)
    plt.show()

    # Also show the batch shape information
    print(f"Batch shape: {images.shape}")

def visualize_pipeline_results(image_tensor, counts, vis_data, max_regions=15):
    """
    Visualize the results of the chocolate detection pipeline

    Args:
        image_tensor: Original image tensor
        counts: Dictionary of chocolate counts
        vis_data: Visualization data from pipeline
        max_regions: Maximum number of regions to display
    """
    # Ensure we have results to display
    if not vis_data['regions']:
        print("No chocolates detected in this image.")
        return

    # Convert image to numpy for visualization
    img_np = inverse_transform(image_tensor)

    # Setup figure for main results
    fig1 = plt.figure(figsize=(18, 6))

    # Original image
    ax1 = fig1.add_subplot(1, 3, 1)
    ax1.imshow(img_np)
    ax1.set_title("Original Image")
    ax1.axis('off')

    # Segmentation results with bounding boxes
    ax2 = fig1.add_subplot(1, 3, 2)
    ax2.imshow(img_np)
    for region in vis_data['regions']:
        x1, y1, x2, y2 = region['bbox']
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='r', linewidth=2)
        ax2.add_patch(rect)
    ax2.set_title(f"Detected Chocolates ({len(vis_data['regions'])} total)")
    ax2.axis('off')

    # Counts barplot
    ax3 = fig1.add_subplot(1, 3, 3)
    non_zero_classes = {k: v for k, v in counts.items() if v > 0}
    if non_zero_classes:
        y_pos = range(len(non_zero_classes))
        bars = ax3.barh(y_pos, list(non_zero_classes.values()))
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(list(non_zero_classes.keys()))
        ax3.invert_yaxis()  # Display top class at the top

        # Add count numbers inside bars
        for i, bar in enumerate(bars):
            count = list(non_zero_classes.values())[i]
            ax3.text(max(0.3, count/2), i, str(count), ha='center', va='center')

        ax3.set_title("Chocolate Counts")
    else:
        ax3.text(0.5, 0.5, "No chocolates detected", ha='center', va='center')
        ax3.set_title("Chocolate Counts (None)")
        ax3.axis('off')

    plt.tight_layout()
    plt.show()

    # Create a separate figure for individual chocolates
    regions_to_show = min(max_regions, len(vis_data['regions']))
    if regions_to_show > 0:
        # Calculate grid dimensions
        cols = min(6, regions_to_show)
        rows = (regions_to_show + cols - 1) // cols  # Ceiling division

        # Create figure
        fig2 = plt.figure(figsize=(3*cols, 3*rows))

        for i in range(regions_to_show):
            # Create subplot
            ax = fig2.add_subplot(rows, cols, i+1)

            # Get region image
            region_img = vis_data['regions'][i]['image']
            if isinstance(region_img, torch.Tensor):
                region_img = inverse_transform(region_img)

            # Show image with classification
            ax.imshow(region_img)
            class_name = vis_data['classes'][i]
            score = vis_data['scores'][i]
            ax.set_title(f"{class_name}\n(conf: {score:.2f})")
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    # Print total counts
    print("\nChocolate Counts:")
    for class_name, count in counts.items():
        if count > 0:
            print(f"  - {class_name}: {count}")
    print(f"Total: {sum(counts.values())}")
