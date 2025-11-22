#!/usr/bin/env python3
"""
green_pixel_visualization.py - Visual demonstration of green pixel detection
Shows how the HSV color analysis identifies healthy plant tissue (green pixels)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path


def apply_shape_filtering(mask):
    """
    Apply shape-based filtering to remove excess pixels and clean up the plant mask
    Uses morphological operations and connected component analysis
    """
    print("🔍 Applying shape-based filtering to remove excess pixels...")
    
    # Step 1: Remove small noise with opening (erosion followed by dilation)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    
    # Step 2: Fill small holes with closing (dilation followed by erosion)
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel_medium)
    
    # Step 3: Connected component analysis to remove isolated small regions
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_closed, connectivity=8)
    
    # Calculate minimum area threshold (0.1% of image area)
    height, width = mask.shape
    min_area = (height * width) * 0.001
    
    # Create final mask keeping only significant regions
    final_mask = np.zeros((height, width), dtype=np.uint8)
    removed_components = 0
    
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area > min_area:
            component_mask = (labels == i).astype(np.uint8) * 255
            final_mask = cv2.bitwise_or(final_mask, component_mask)
        else:
            removed_components += 1
    
    # Step 4: Final smoothing to create more natural plant shapes
    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_smooth)
    
    pixels_removed = np.count_nonzero(mask) - np.count_nonzero(final_mask)
    print(f"   ✂️ Removed {pixels_removed:,} excess pixels ({removed_components} small components)")
    print(f"   🌿 Shape-filtered mask: {np.count_nonzero(final_mask):,} pixels")
    
    return final_mask


def detect_plant_boundaries(image):
    """
    Automatically detect optimal ROI boundaries based on plant content
    Returns: (exclude_above_y, roi_top_y) boundaries
    """
    height, width = image.shape[:2]
    
    # Convert to HSV and detect green pixels
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 20, 20])
    upper_green = np.array([60, 255, 255])
    green_mask = cv2.inRange(image_hsv, lower_green, upper_green)
    
    # Find rows with green pixels
    row_sums = np.sum(green_mask, axis=1)
    green_rows = np.where(row_sums > 0)[0]
    
    if len(green_rows) == 0:
        print("⚠️ No green pixels found, using default boundaries")
        return int(height * 0.4), int(height * 0.9)
    
    # Find plant extent
    first_green_row = green_rows[0]
    last_green_row = green_rows[-1]
    plant_height = last_green_row - first_green_row
    
    # Add adaptive padding
    top_padding = max(20, int(plant_height * 0.1))  # At least 20px or 10% of plant height
    bottom_padding = max(30, int(plant_height * 0.05)) # At least 30px or 5% of plant height
    
    # Calculate boundaries
    exclude_above_y = max(0, first_green_row - top_padding)
    roi_top_y = min(height, last_green_row + bottom_padding)
    
    print(f"🔍 Auto-detected plant boundaries:")
    print(f"   Plant foliage span: y={first_green_row} to y={last_green_row} ({plant_height}px tall)")
    print(f"   Adaptive ROI: y={exclude_above_y} to y={roi_top_y} (padding: top={top_padding}px, bottom={bottom_padding}px)")
    
    return exclude_above_y, roi_top_y


def create_pot_exclusion_mask(image, method='roi_coordinates', crop_ratio=0.7, roi_top_y=950, exclude_above_y=400, exclude_center_stick=True, stick_width=40, auto_detect=False, known_stick_point=None):
    """
    Create a mask to exclude pot/soil area from analysis
    Methods:
    - 'crop_top': Only analyze top portion of image
    - 'roi_coordinates': Use specific y-coordinate as ROI boundary
    - 'color_based': Exclude brown/dark soil colors
    - 'combined': Use both methods
    - 'auto_boundaries': Automatically detect optimal boundaries
    """
    height, width = image.shape[:2]
    mask = np.ones((height, width), dtype=np.uint8) * 255  # Start with all white (include all)
    
    # Auto-detect boundaries if requested
    if method == 'auto_boundaries' or auto_detect:
        exclude_above_y, roi_top_y = detect_plant_boundaries(image)
        method = 'roi_coordinates'  # Use detected boundaries with roi_coordinates method
    
    if method in ['crop_top', 'combined']:
        # Exclude bottom portion (pot area)
        crop_height = int(height * crop_ratio)
        mask[crop_height:, :] = 0  # Set bottom area to black (exclude)
        print(f"🪴 Cropping to top {crop_ratio*100:.0f}% of image (excluding pot)")
    
    if method in ['roi_coordinates', 'combined']:
        # Use specific y-coordinate as ROI boundary
        # Analyze everything above y=902 (plant area), exclude below (pot area)
        if roi_top_y < height:
            mask[roi_top_y:, :] = 0  # Exclude area below y=902 (pot area)
            print(f"📍 ROI set: analyzing from y=0 to y={roi_top_y} (plant area above pot)")
        else:
            print(f"⚠️ ROI y-coordinate ({roi_top_y}) exceeds image height ({height})")
        
        # Also exclude everything above y=497 (background/setup area)
        if exclude_above_y > 0:
            mask[:exclude_above_y, :] = 0  # Exclude area above y=497
            print(f"📍 Additional exclusion: removing pixels above y={exclude_above_y} (background area)")
    
    if method in ['color_based', 'combined']:
        # Enhanced pot detection for green-tinted pots
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Brown/soil color ranges
        lower_brown1 = np.array([8, 50, 20])   # Light brown
        upper_brown1 = np.array([25, 255, 200])
        
        lower_brown2 = np.array([0, 0, 0])     # Dark colors (soil/shadows)
        upper_brown2 = np.array([180, 255, 50])
        
        # Green-tinted pot colors (algae, moss, or lighting effects)
        lower_green_pot = np.array([35, 30, 30])   # Dull green on pot surfaces
        upper_green_pot = np.array([85, 120, 150])  # Not bright healthy plant green
        
        # Create masks for pot colors
        brown_mask1 = cv2.inRange(image_hsv, lower_brown1, upper_brown1)
        brown_mask2 = cv2.inRange(image_hsv, lower_brown2, upper_brown2)
        
        # Only apply green pot exclusion in the pot region (bottom portion)
        pot_region_start = int(height * 0.75)  # Bottom 25% of image
        green_pot_mask = np.zeros((height, width), dtype=np.uint8)
        green_pot_region = cv2.inRange(image_hsv[pot_region_start:, :], lower_green_pot, upper_green_pot)
        green_pot_mask[pot_region_start:, :] = green_pot_region
        
        # Combine all pot masks
        pot_mask = cv2.bitwise_or(brown_mask1, brown_mask2)
        pot_mask = cv2.bitwise_or(pot_mask, green_pot_mask)
        
        # Remove pot colors from analysis mask
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(pot_mask))
        
        excluded_pot_pixels = np.count_nonzero(pot_mask)
        print(f"🎨 Excluding brown/soil colored pixels")
        print(f"🪴 Excluding {excluded_pot_pixels} green-tinted pot pixels")
    
    # Exclude center stick if requested
    if exclude_center_stick and stick_width > 0:
        # Create vertical mask for center stick
        center_x = width // 2
        stick_left = max(0, center_x - stick_width // 2)
        stick_right = min(width, center_x + stick_width // 2)
        
        # Exclude the stick area
        mask[:, stick_left:stick_right] = 0
        print(f"🌿 Excluding center stick: {stick_width}px wide at x={center_x}")
    
    return mask


def visualize_green_pixels(image_path, output_path=None, exclude_pot=True, pot_method='roi_coordinates', crop_ratio=0.7, roi_top_y=950, exclude_above_y=400, exclude_center_stick=True, stick_width=40, auto_detect=False, known_stick_point=None):
    """
    Load an image and create a visualization showing detected green pixels
    with optional pot exclusion and center stick exclusion
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return None
    
    print(f"📸 Processing image: {os.path.basename(image_path)}")
    
    # Get original dimensions
    original_height, original_width = image.shape[:2]
    print(f"📏 Original size: {original_width}x{original_height} pixels")
    
    # Resize for processing (same as in main script)
    if original_width > 800:
        scale = 800 / original_width
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        image = cv2.resize(image, (new_width, new_height))
        print(f"📏 Resized to: {new_width}x{new_height} pixels")
    
    # Convert to HSV color space
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # HSV ranges for green pixels (same as main script)
    lower_green = np.array([30, 20, 20])
    upper_green = np.array([60, 255, 255])
    
    # Create green mask
    green_mask = cv2.inRange(image_hsv, lower_green, upper_green)
    
    # Create pot exclusion mask if requested
    if exclude_pot:
        pot_exclusion_mask = create_pot_exclusion_mask(image, method=pot_method, crop_ratio=crop_ratio, roi_top_y=roi_top_y, exclude_above_y=exclude_above_y, exclude_center_stick=exclude_center_stick, stick_width=stick_width, auto_detect=auto_detect, known_stick_point=known_stick_point)
    else:
        pot_exclusion_mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255
    
    # Apply pot exclusion to green mask
    green_mask_filtered = cv2.bitwise_and(green_mask, pot_exclusion_mask)
    
    # Apply shape-based filtering to remove excess pixels
    green_mask_shape_filtered = apply_shape_filtering(green_mask_filtered)
    
    # Count green pixels (filtered vs unfiltered vs shape-filtered)
    green_pixel_count_original = np.count_nonzero(green_mask)
    green_pixel_count_filtered = np.count_nonzero(green_mask_filtered)
    green_pixel_count_shape_filtered = np.count_nonzero(green_mask_shape_filtered)
    
    # Calculate percentages based on analysis area (not total image)
    analysis_area = np.count_nonzero(pot_exclusion_mask)
    total_pixels = image.shape[0] * image.shape[1]
    
    green_percentage_original = (green_pixel_count_original / total_pixels) * 100
    green_percentage_filtered = (green_pixel_count_filtered / analysis_area) * 100 if analysis_area > 0 else 0
    green_percentage_shape_filtered = (green_pixel_count_shape_filtered / analysis_area) * 100 if analysis_area > 0 else 0
    
    print(f"🌿 Green pixels detected (original): {green_pixel_count_original:,}")
    print(f"🌿 Green pixels detected (ROI filtered): {green_pixel_count_filtered:,}")
    print(f"🌿 Green pixels detected (shape filtered): {green_pixel_count_shape_filtered:,}")
    print(f"📊 Green percentage (original): {green_percentage_original:.2f}%")
    print(f"📊 Green percentage (ROI filtered): {green_percentage_filtered:.2f}%")
    print(f"📊 Green percentage (shape filtered): {green_percentage_shape_filtered:.2f}%")
    print(f"📏 Analysis area: {analysis_area:,} pixels ({(analysis_area/total_pixels)*100:.1f}% of image)")
    
    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create visualization with subplots showing the shape filtering process
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    # Original image
    axes[0].imshow(image_rgb)
    axes[0].set_title('Original Image', fontweight='bold', fontsize=14)
    axes[0].axis('off')
    
    # ROI filtered green mask
    axes[1].imshow(green_mask_filtered, cmap='gray')
    axes[1].set_title(f'ROI Filtered Green\n{green_pixel_count_filtered:,} pixels', fontweight='bold', fontsize=14)
    axes[1].axis('off')
    
    # Shape filtered green mask (final result)
    axes[2].imshow(green_mask_shape_filtered, cmap='gray')
    axes[2].set_title(f'Shape Filtered Green\n{green_pixel_count_shape_filtered:,} pixels', fontweight='bold', fontsize=14)
    axes[2].axis('off')
    
    # Create overlay showing only the shape-filtered pixels in green
    overlay = np.zeros_like(image_rgb)
    overlay[green_mask_shape_filtered > 0] = [0, 255, 0]  # Pure green for plant pixels
    overlay[green_mask_shape_filtered == 0] = [50, 50, 50]  # Dark gray for background
    
    axes[3].imshow(overlay)
    axes[3].set_title(f'Final Plant Shape\n{green_pixel_count_shape_filtered:,} pixels ({green_percentage_shape_filtered:.2f}%)', 
                     fontweight='bold', fontsize=14)
    axes[3].axis('off')
    
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Visualization saved to: {output_path}")
    
    plt.show()
    
    return {
        'green_pixels_original': green_pixel_count_original,
        'green_pixels_filtered': green_pixel_count_filtered,
        'green_pixels_shape_filtered': green_pixel_count_shape_filtered,
        'total_pixels': total_pixels,
        'analysis_area': analysis_area,
        'green_percentage_original': green_percentage_original,
        'green_percentage_filtered': green_percentage_filtered,
        'green_percentage_shape_filtered': green_percentage_shape_filtered,
        'image_size': (image.shape[1], image.shape[0])
    }


def find_example_image(image_index=1):
    """
    Find an example image from the dataset
    """
    # Try the main data folder
    data_folder = "/Users/olivia/Library/CloudStorage/GoogleDrive-olivia.simon@temporalag.com/My Drive/Purps Phenotypes/Viserion Heat Test 1 - For Olivia"
    
    # Look for JPG files
    all_images = []
    if os.path.exists(data_folder):
        for root, dirs, files in os.walk(data_folder):
            for file in files:
                if file.lower().endswith('.jpg'):
                    all_images.append(os.path.join(root, file))
    
    if len(all_images) > image_index:
        example_image = all_images[image_index]
        print(f"🎯 Found example image #{image_index + 1}: {example_image}")
        return example_image
    elif len(all_images) > 0:
        # Fallback to first image if index is out of range
        example_image = all_images[0]
        print(f"🎯 Using first available image (index {image_index} not found): {example_image}")
        return example_image
    
    print("❌ No example images found")
    return None


def main():
    """
    Main function to demonstrate green pixel detection
    """
    print("🔍 GREEN PIXEL VISUALIZATION DEMO")
    print("=" * 50)
    
    # Find a different example image (15th image in the dataset)
    example_image_path = find_example_image(image_index=14)
    
    if example_image_path is None:
        print("❌ Could not find example image. Please check the data folder path.")
        return
    
    # Create output path
    output_folder = "/Users/olivia/Git"
    output_filename = f"green_pixel_demo_{os.path.basename(example_image_path).replace('.jpg', '.png')}"
    output_path = os.path.join(output_folder, output_filename)
    
    # Analyze and visualize with automatic boundary detection
    print("\n🤖 Using automatic boundary detection for optimal ROI...")
    results = visualize_green_pixels(example_image_path, output_path, exclude_pot=True, pot_method='auto_boundaries', auto_detect=True, exclude_center_stick=False)
    
    if results:
        print("\n✅ GREEN PIXEL ANALYSIS COMPLETE!")
        print(f"📊 Original results: {results['green_pixels_original']:,} green pixels ({results['green_percentage_original']:.2f}%)")
        print(f"🔍 ROI filtered: {results['green_pixels_filtered']:,} green pixels ({results['green_percentage_filtered']:.2f}%)")
        print(f"🌿 Shape filtered: {results['green_pixels_shape_filtered']:,} green pixels ({results['green_percentage_shape_filtered']:.2f}%)")
        print(f"📏 Analysis area: {results['analysis_area']:,} pixels ({(results['analysis_area']/results['total_pixels'])*100:.1f}% of image)")
        print(f"📏 Processed image size: {results['image_size'][0]}x{results['image_size'][1]} pixels")


if __name__ == "__main__":
    main()
