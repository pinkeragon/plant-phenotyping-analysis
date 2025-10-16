#!/usr/bin/env python3
"""
viserion_peterpettigrew_FAST.py - FAST version for new Google Drive dataset
Processes only first 5 images per folder with automatic green pixel visualization
"""

# ============================================================================
# IMPORTS
# ============================================================================
import numpy, pandas, matplotlib, seaborn, scipy, cv2
import os
import re
import shutil
import subprocess
import time
import glob
import datetime
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import label
import matplotlib.pyplot as plt

print("All good! (FAST TEST VERSION)")
print("Skipping file renaming, going straight to analysis...")

# ============================================================================
# FUNCTION DEFINITIONS
# ============================================================================

def apply_shape_filtering(mask):
    """
    Apply shape-based filtering to remove excess pixels and clean up the plant mask
    """
    # Remove small noise with opening
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    
    # Fill small holes with closing
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel_medium)
    
    # Connected component analysis to remove isolated small regions
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_closed, connectivity=8)
    
    # Calculate minimum area threshold (0.1% of image area)
    height, width = mask.shape
    min_area = (height * width) * 0.001
    
    # Create final mask keeping only significant regions
    final_mask = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area > min_area:
            component_mask = (labels == i).astype(np.uint8) * 255
            final_mask = cv2.bitwise_or(final_mask, component_mask)
    
    # Final smoothing
    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_smooth)
    
    return final_mask


def detect_plant_boundaries(image):
    """
    Automatically detect optimal ROI boundaries based on plant content
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
        return int(height * 0.4), int(height * 0.9)
    
    # Find plant extent
    first_green_row = green_rows[0]
    last_green_row = green_rows[-1]
    plant_height = last_green_row - first_green_row
    
    # Add adaptive padding
    top_padding = max(20, int(plant_height * 0.1))
    bottom_padding = max(30, int(plant_height * 0.05))
    
    # Calculate boundaries
    exclude_above_y = max(0, first_green_row - top_padding)
    roi_top_y = min(height, last_green_row + bottom_padding)
    
    return exclude_above_y, roi_top_y


def create_pot_exclusion_mask(image, exclude_above_y, roi_top_y):
    """
    Create a mask to exclude pot/soil area from analysis with automatic boundaries
    """
    height, width = image.shape[:2]
    mask = np.ones((height, width), dtype=np.uint8) * 255
    
    # Apply adaptive ROI boundaries
    if roi_top_y < height:
        mask[roi_top_y:, :] = 0  # Exclude area below plant
    
    if exclude_above_y > 0:
        mask[:exclude_above_y, :] = 0  # Exclude area above plant
    
    # Enhanced pot detection for green-tinted pots
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Brown/soil color ranges
    lower_brown1 = np.array([8, 50, 20])
    upper_brown1 = np.array([25, 255, 200])
    lower_brown2 = np.array([0, 0, 0])
    upper_brown2 = np.array([180, 255, 50])
    
    # Green-tinted pot colors
    lower_green_pot = np.array([35, 30, 30])
    upper_green_pot = np.array([85, 120, 150])
    
    # Create masks for pot colors
    brown_mask1 = cv2.inRange(image_hsv, lower_brown1, upper_brown1)
    brown_mask2 = cv2.inRange(image_hsv, lower_brown2, upper_brown2)
    
    # Only apply green pot exclusion in the pot region (bottom portion)
    pot_region_start = int(height * 0.75)
    green_pot_mask = np.zeros((height, width), dtype=np.uint8)
    green_pot_region = cv2.inRange(image_hsv[pot_region_start:, :], lower_green_pot, upper_green_pot)
    green_pot_mask[pot_region_start:, :] = green_pot_region
    
    # Combine all pot masks
    pot_mask = cv2.bitwise_or(brown_mask1, brown_mask2)
    pot_mask = cv2.bitwise_or(pot_mask, green_pot_mask)
    
    # Remove pot colors from analysis mask
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(pot_mask))
    
    return mask


def create_visualization(image, green_mask_filtered, green_mask_final, filename, output_dir):
    """
    Create a 4-panel visualization showing the processing steps
    """
    import matplotlib.pyplot as plt
    
    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create visualization with subplots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original image
    axes[0].imshow(image_rgb)
    axes[0].set_title('Original Image', fontweight='bold', fontsize=12)
    axes[0].axis('off')
    
    # ROI filtered green mask
    axes[1].imshow(green_mask_filtered, cmap='gray')
    axes[1].set_title(f'ROI Filtered\n{np.count_nonzero(green_mask_filtered):,} pixels', fontweight='bold', fontsize=12)
    axes[1].axis('off')
    
    # Shape filtered green mask
    axes[2].imshow(green_mask_final, cmap='gray')
    axes[2].set_title(f'Shape Filtered\n{np.count_nonzero(green_mask_final):,} pixels', fontweight='bold', fontsize=12)
    axes[2].axis('off')
    
    # Final overlay
    overlay = np.zeros_like(image_rgb)
    overlay[green_mask_final > 0] = [0, 255, 0]  # Green for plant
    overlay[green_mask_final == 0] = [50, 50, 50]  # Dark gray for background
    
    axes[3].imshow(overlay)
    axes[3].set_title(f'Final Plant Shape\n{np.count_nonzero(green_mask_final):,} pixels', fontweight='bold', fontsize=12)
    axes[3].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    output_path = os.path.join(output_dir, f"{filename}_visualization.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def calculate_height_width(mask, min_cluster_size=50):
    """
    Calculate plant height and width from green pixel mask
    """
    labeled_array, num_features = label(mask)
    green_coords = []

    for i in range(1, num_features + 1):
        cluster_coords = np.column_stack(np.where(labeled_array == i))
        if len(cluster_coords) >= min_cluster_size:
            green_coords.append(cluster_coords)

    if not green_coords:
        return 0, 0, 0, 0, 0, 0

    green_coords = np.vstack(green_coords)
    min_y, max_y = np.min(green_coords[:, 0]), np.max(green_coords[:, 0])
    min_x, max_x = np.min(green_coords[:, 1]), np.max(green_coords[:, 1])

    height = int(max_y - min_y)
    width = int(max_x - min_x)

    return height, width, int(min_x), int(max_x), int(min_y), int(max_y)


def process_single_image(image_path, filename, folder_name, pattern, create_visualization_flag, top_level_folder_path, created_visualizations):
    """
    Process a single image and return analysis results
    """
    filename_without_ext = os.path.splitext(filename)[0]
    
    m = pattern.match(filename_without_ext)
    if not m:
        return None, True, False  # malformed, unreadable
    
    # Extract components from NEF filename format: A_1_0dpi_2.NEF
    prefix = m.group("prefix").upper()
    well = int(m.group("well"))
    angle_code = int(m.group("angle"))
    
    # Convert angle code to degrees: 1=0°, 2=90°, 3=180°, 4=270°
    angle_to_degrees = {1: 0, 2: 90, 3: 180, 4: 270}
    rotation = angle_to_degrees.get(angle_code, angle_code)  # fallback to original if not found
    
    # Use folder name as date_str since there's no date in filename
    date_str = folder_name
    
    # Try to read NEF file with OpenCV (may not work directly)
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"   ⚠️ Could not read NEF file: {filename} (OpenCV may not support NEF format)")
        return None, False, True  # malformed, unreadable
    
    # Resize for faster processing
    height, width = image.shape[:2]
    if width > 500:  # Smaller resize for speed
        scale = 500 / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height))

    total_pixel_count = int(image.shape[0] * image.shape[1])
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Step 1: Detect adaptive ROI boundaries based on plant content
    exclude_above_y, roi_top_y = detect_plant_boundaries(image)
    
    # Step 2: Create pot exclusion mask with adaptive boundaries
    pot_exclusion_mask = create_pot_exclusion_mask(image, exclude_above_y, roi_top_y)
    analysis_area = int(np.count_nonzero(pot_exclusion_mask))

    # HSV ranges
    lower_yellow = np.array([25, 50, 50]); upper_yellow = np.array([30, 255, 255])
    lower_green = np.array([30, 20, 20]); upper_green = np.array([60, 255, 255])
    lower_red1 = np.array([0, 50, 50]); upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50]); upper_red2 = np.array([180, 255, 255])

    # Create color masks
    yellow_mask = cv2.inRange(image_hsv, lower_yellow, upper_yellow)
    green_mask = cv2.inRange(image_hsv, lower_green, upper_green)
    red_mask = cv2.bitwise_or(cv2.inRange(image_hsv, lower_red1, upper_red1),
                             cv2.inRange(image_hsv, lower_red2, upper_red2))

    # Step 3: Apply pot exclusion to all color masks
    yellow_mask_filtered = cv2.bitwise_and(yellow_mask, pot_exclusion_mask)
    green_mask_filtered = cv2.bitwise_and(green_mask, pot_exclusion_mask)
    red_mask_filtered = cv2.bitwise_and(red_mask, pot_exclusion_mask)

    # Step 4: Apply shape-based filtering to clean up masks
    yellow_mask_final = apply_shape_filtering(yellow_mask_filtered)
    green_mask_final = apply_shape_filtering(green_mask_filtered)
    red_mask_final = apply_shape_filtering(red_mask_filtered)

    # Calculate dimensions using the filtered green mask
    plant_height, plant_width, min_x, max_x, min_y, max_y = calculate_height_width(green_mask_final)

    # Count pixels from filtered and shape-cleaned masks
    yellow_count = int(np.count_nonzero(yellow_mask_final))
    green_count = int(np.count_nonzero(green_mask_final))
    red_count = int(np.count_nonzero(red_mask_final))
    
    # Calculate normalized yellow ratio (yellow / (yellow + green))
    plant_pixels = yellow_count + green_count
    normalized_yellow = yellow_count / plant_pixels if plant_pixels > 0 else 0
    
    # Calculate analysis coverage metrics
    analysis_coverage = (analysis_area / total_pixel_count) * 100
    plant_density = (plant_pixels / analysis_area) * 100 if analysis_area > 0 else 0

    # Create visualization for suitable images
    if create_visualization_flag and green_count > 1000:  # Only if significant plant detected
        print(f"   🎨 Creating visualization for {filename_without_ext} (randomly selected)")
        viz_path = create_visualization(image, green_mask_filtered, green_mask_final, 
                                      filename_without_ext, top_level_folder_path)
        print(f"   💾 Visualization saved: {os.path.basename(viz_path)}")
        
        # Immediately open the visualization
        try:
            print(f"   🖼️ Opening visualization: {os.path.basename(viz_path)}")
            subprocess.run(['open', viz_path], check=True)
            time.sleep(1)  # Brief pause to ensure it opens
        except Exception as e:
            print(f"   ⚠️ Failed to open visualization: {e}")
            try:
                os.system(f'open "{viz_path}"')
            except:
                pass
        
        # Add to cleanup list
        created_visualizations.append(viz_path)

    # Return analysis results
    result = {
        "Date": date_str,
        "Flat": folder_name,  # Using folder name as identifier
        "Prefix": prefix,
        "Well": well,
        "Rotation": rotation,
        "Filename": filename_without_ext,
        "Plant Height (pixels)": plant_height,
        "Plant Width (pixels)": plant_width,
        "Total Pixel Count": total_pixel_count,
        "Analysis Area": analysis_area,
        "Analysis Coverage (%)": round(analysis_coverage, 2),
        "Yellow Pixel Count": yellow_count,
        "Green Pixel Count": green_count,
        "Red Pixel Count": red_count,
        "Total Plant Pixels": plant_pixels,
        "Plant Density (%)": round(plant_density, 2),
        "Normalized Yellow Ratio": round(normalized_yellow, 4),
        "ROI Top Y": exclude_above_y,
        "ROI Bottom Y": roi_top_y,
    }
    
    return result, False, False  # not malformed, not unreadable


def create_statistical_analysis(df, csv_file_path):
    """
    Create comprehensive statistical analysis and plots
    """
    print("\n📊 Creating foliage visualization...")
    
    # Calculate foliage (green) pixel percentage
    df['Foliage_Percentage'] = (df['Green Pixel Count'] / df['Total Pixel Count']) * 100
    
    # Extract group information from prefix and create pot identifier
    df['Group'] = df['Prefix']
    df['Pot_ID'] = df['Prefix'] + df['Well'].astype(str)
    
    # Group data by Group and calculate mean foliage percentage
    import matplotlib.pyplot as plt
    
    group_stats = df.groupby('Group')['Foliage_Percentage'].agg(['mean', 'std', 'count']).reset_index()
    
    # Print summary to console
    print("\n🌿 FOLIAGE SUMMARY:")
    print(group_stats.round(2))
    
    # 📈 CREATE LINE PLOTS (Time Series Analysis)
    print("\n📈 Creating line plots for foliage analysis over time...")
    
    # Since we don't have actual dates, create a simple sequence based on folder order
    # Create a sequential day number for each unique folder
    unique_folders = df['Date'].unique()
    folder_to_day = {folder: i for i, folder in enumerate(sorted(unique_folders))}
    df['Days_Post_Treatment'] = df['Date'].map(folder_to_day)
    
    print(f"📅 Processing {len(unique_folders)} folder(s)")
    print(f"📅 Folder sequence: {df['Days_Post_Treatment'].min()} to {df['Days_Post_Treatment'].max()}")
    
    # Group data for time series plotting
    time_data = df.groupby(['Group', 'Days_Post_Treatment']).agg({
        'Foliage_Percentage': ['mean', 'std', 'count'],
        'Green Pixel Count': ['mean', 'std'],
        'Yellow Pixel Count': ['mean', 'std'],
        'Plant Height (pixels)': ['mean', 'std'],
        'Normalized Yellow Ratio': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    time_data.columns = ['Group', 'Days_Post_Treatment', 
                        'Foliage_Mean', 'Foliage_Std', 'Foliage_Count',
                        'Green_Mean', 'Green_Std', 
                        'Yellow_Mean', 'Yellow_Std',
                        'Height_Mean', 'Height_Std',
                        'Normalized_Yellow_Mean', 'Normalized_Yellow_Std']
    
    # Create focused visualization with core metrics - Optimized for macOS screen
    import seaborn as sns
    from scipy import stats
    from scipy.stats import f_oneway, ttest_ind
    
    plt.figure(figsize=(18, 8))
    
    # Unified color scheme: A=Pink, B=Blue, C=Green, D=Orange
    group_colors = {'A': '#E691A8', 'B': '#5A9FD4', 'C': '#7AE582', 'D': '#F4A460'}
    
    # STATIC ANALYSIS PLOTS (Row 1)
    
    # Plot 1: Box plot of foliage percentage distribution with statistical analysis
    plt.subplot(2, 3, 1)
    groups = df['Group'].unique()
    foliage_data = [df[df['Group'] == group]['Foliage_Percentage'] for group in sorted(groups)]
    box_plot = plt.boxplot(foliage_data, tick_labels=sorted(groups), patch_artist=True)
    
    # Color the boxes with group colors
    group_names = sorted(groups)
    for i, patch in enumerate(box_plot['boxes']):
        if i < len(group_names) and group_names[i] in group_colors:
            patch.set_facecolor(group_colors[group_names[i]])
            patch.set_alpha(0.7)
    
    # Perform ANOVA
    if len(foliage_data) > 1:
        f_stat, p_value = f_oneway(*foliage_data)
        
        # Add statistical annotations
        plt.title(f'Foliage % Distribution\nANOVA: F={f_stat:.2f}, p={p_value:.3f}', fontweight='bold', fontsize=10)
        
        # Add pairwise comparisons (simplified approach)
        if p_value < 0.05 and len(foliage_data) >= 2:
            max_y = max([max(data) if len(data) > 0 else 0 for data in foliage_data])
            # Compare first two groups if available
            if len(foliage_data[0]) > 0 and len(foliage_data[-1]) > 0:
                t_stat, p_val_comp = ttest_ind(foliage_data[0], foliage_data[-1])
                
                if p_val_comp < 0.05:
                    significance = '***' if p_val_comp < 0.001 else '**' if p_val_comp < 0.01 else '*'
                    plt.plot([1, len(group_names)], [max_y + 0.5, max_y + 0.5], 'k-', linewidth=1)
                    plt.text((1 + len(group_names))/2, max_y + 0.7, significance, ha='center', fontsize=12, fontweight='bold')
    else:
        plt.title('Foliage % Distribution', fontweight='bold', fontsize=10)
    
    plt.xlabel('Group')
    plt.ylabel('Foliage %')
    plt.grid(True, alpha=0.3)
    
    # TIME SERIES PLOTS - STANDARDIZED COLORS AND SCALES
    
    # Plot 2: Foliage percentage over time by group
    plt.subplot(2, 3, 2)
    for group in sorted(time_data['Group'].unique()):
        group_data = time_data[time_data['Group'] == group]
        color = group_colors.get(group, '#888888')
        plt.errorbar(group_data['Days_Post_Treatment'], group_data['Foliage_Mean'], 
                    yerr=group_data['Foliage_Std'], 
                    label=f'{group}', marker='o', linewidth=2, 
                    capsize=5, color=color)
    
    plt.title('Foliage % Over Time by Group', fontweight='bold', fontsize=10)
    plt.xlabel('Time Point')
    plt.ylabel('Foliage %')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Green pixel count over time
    plt.subplot(2, 3, 3)
    for group in sorted(time_data['Group'].unique()):
        group_data = time_data[time_data['Group'] == group]
        color = group_colors.get(group, '#888888')
        plt.errorbar(group_data['Days_Post_Treatment'], group_data['Green_Mean']/1000, 
                    yerr=group_data['Green_Std']/1000, 
                    label=f'{group}', marker='o', linewidth=2, 
                    capsize=5, color=color)
    
    plt.title('Green Pixel Count Over Time', fontweight='bold', fontsize=10)
    plt.xlabel('Time Point')
    plt.ylabel('Green Pixels (k)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Plant height over time
    plt.subplot(2, 3, 4)
    for group in sorted(time_data['Group'].unique()):
        group_data = time_data[time_data['Group'] == group]
        color = group_colors.get(group, '#888888')
        plt.errorbar(group_data['Days_Post_Treatment'], group_data['Height_Mean'], 
                    yerr=group_data['Height_Std'], 
                    label=f'{group}', marker='o', linewidth=2, 
                    capsize=5, color=color)
    
    plt.title('Plant Height Over Time', fontweight='bold', fontsize=10)
    plt.xlabel('Time Point')
    plt.ylabel('Height (px)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Foliage change from baseline
    plt.subplot(2, 3, 5)
    for group in sorted(time_data['Group'].unique()):
        group_data = time_data[time_data['Group'] == group].sort_values('Days_Post_Treatment')
        if len(group_data) > 1:
            baseline = group_data['Foliage_Mean'].iloc[0]
            change = group_data['Foliage_Mean'] - baseline
            color = group_colors.get(group, '#888888')
            plt.plot(group_data['Days_Post_Treatment'], change, 
                    label=f'{group}', marker='o', linewidth=2, 
                    color=color)
    
    plt.title('Foliage Change from Baseline', fontweight='bold', fontsize=10)
    plt.xlabel('Time Point')
    plt.ylabel('Foliage % Change')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Plant height distribution by group with statistical analysis
    plt.subplot(2, 3, 6)
    groups = df['Group'].unique()
    height_data = [df[df['Group'] == group]['Plant Height (pixels)'] for group in sorted(groups)]
    box_plot = plt.boxplot(height_data, tick_labels=sorted(groups), patch_artist=True)
    
    # Color the boxes with group colors
    group_names = sorted(groups)
    for i, patch in enumerate(box_plot['boxes']):
        if i < len(group_names) and group_names[i] in group_colors:
            patch.set_facecolor(group_colors[group_names[i]])
            patch.set_alpha(0.7)
    
    # Perform ANOVA for height
    if len(height_data) > 1:
        f_stat_h, p_value_h = f_oneway(*height_data)
        
        # Add statistical annotations
        plt.title(f'Plant Height Distribution\nANOVA: F={f_stat_h:.2f}, p={p_value_h:.3f}', fontweight='bold', fontsize=10)
        
        # Add pairwise comparisons for height
        if p_value_h < 0.05 and len(height_data) >= 2:
            max_y_h = max([max(data) if len(data) > 0 else 0 for data in height_data])
            # Compare first and last groups if available
            if len(height_data[0]) > 0 and len(height_data[-1]) > 0:
                t_stat_h, p_val_h_comp = ttest_ind(height_data[0], height_data[-1])
                
                if p_val_h_comp < 0.05:
                    significance_h = '***' if p_val_h_comp < 0.001 else '**' if p_val_h_comp < 0.01 else '*'
                    plt.plot([1, len(group_names)], [max_y_h + 20, max_y_h + 20], 'k-', linewidth=1)
                    plt.text((1 + len(group_names))/2, max_y_h + 30, significance_h, ha='center', fontsize=12, fontweight='bold')
    else:
        plt.title('Plant Height Distribution', fontweight='bold', fontsize=10)
    
    plt.xlabel('Group')
    plt.ylabel('Height (px)')
    plt.grid(True, alpha=0.3)
    
    # Adjust spacing between subplots for better readability and wider plots
    plt.subplots_adjust(hspace=0.4, wspace=0.3, left=0.05, right=0.98, top=0.95, bottom=0.08)
    
    # Save the comprehensive combined plot
    p = Path(csv_file_path)
    combined_plot_path = p.parent / f"{p.stem}_COMBINED_ANALYSIS.png"
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Combined analysis plots saved to: {combined_plot_path}")
    
    plt.show()


def cleanup_visualizations(created_visualizations):
    """
    Clean up visualization files after analysis
    """
    if created_visualizations:
        print(f"\n🗑️ Auto-deleting {len(created_visualizations)} visualization files...")
        print("⏳ Waiting 15 seconds for you to review the visualizations...")
        
        # Wait 15 seconds for user to view the files
        time.sleep(15)
        
        # Delete the visualization files
        for viz_file in created_visualizations:
            filename = os.path.basename(viz_file)
            try:
                os.remove(viz_file)
                print(f"   ✅ Deleted: {filename}")
            except Exception as e:
                print(f"   ⚠️ Failed to delete: {filename} - {e}")
        
        print("🧹 Cleanup complete!")
    else:
        print("\n⚠️ No visualization files were created to clean up")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function
    """
    print("🔍 FAST MODE: Processing only first 5 images per folder...")
    
    # Configuration - Updated to use new Google Drive shortcut folder
    top_level_folder_path = r"/Users/olivia/Library/CloudStorage/GoogleDrive-olivia.simon@temporalag.com/.shortcut-targets-by-id/1wfQj5eKPBwlq4-7faNtD1x6s-9sh7X69/t0_0dpt"
    folder_name = os.path.basename(top_level_folder_path)
    csv_file_path = os.path.join(top_level_folder_path, f"{folder_name}_FAST_TEST.csv")

    # Pattern for NEF files: A_1_0dpi_2.NEF (flat_plant_dpi_angle.NEF)
    PATTERN = re.compile(r"^(?P<prefix>[A-Z])_(?P<well>\d+)_0dpi_(?P<angle>\d+)$", re.IGNORECASE)
    
    results = []
    malformed = 0
    unreadable = 0
    processed_count = 0

    MAX_IMAGES_PER_FOLDER = 5  # FAST VERSION: Only process 5 images per folder for testing

    # Counter to track folder processing for visualization selection
    folder_counter = 0

    # List to track created visualization files for cleanup
    created_visualizations = []

    # Process images organized by folders
    for root, dirs, files in os.walk(top_level_folder_path):
        # Skip the root directory itself and only process subfolders
        if root == top_level_folder_path:
            continue
        
        folder_name = os.path.basename(root)
        
        # Process all folders (not just date folders since NEF dataset may have different structure)
        # Skip hidden folders and system folders
        if folder_name.startswith('.') or folder_name.startswith('__'):
            continue
        
        print(f"📁 Processing folder: {folder_name}")
        
        # Increment folder counter
        folder_counter += 1
        
        nef_files = [f for f in files if f.lower().endswith('.nef')]
        images_processed_in_folder = 0
        
        # Only create visualization for every 2nd folder (5 total from 10 folders)
        create_visualization_for_folder = (folder_counter % 2 == 0)  # Every 2nd folder
        
        # Randomly shuffle the files so we get different images each run
        import random
        random.shuffle(nef_files)
        
        for filename in nef_files:
            if images_processed_in_folder >= MAX_IMAGES_PER_FOLDER:
                print(f"   ⚡ Stopping at {MAX_IMAGES_PER_FOLDER} images for {folder_name}")
                break
            
            image_path = os.path.join(root, filename)
            
            # Process single image
            result, is_malformed, is_unreadable = process_single_image(
                image_path, filename, folder_name, PATTERN, 
                create_visualization_for_folder, top_level_folder_path, created_visualizations
            )
            
            if is_malformed:
                malformed += 1
                continue
            
            if is_unreadable:
                unreadable += 1
                continue
            
            if result:
                results.append(result)
                processed_count += 1
                images_processed_in_folder += 1
                create_visualization_for_folder = False  # Only create one per folder
                
                # Progress tracking for large datasets
                if processed_count % 50 == 0:
                    print(f"   ⚡ Progress: {processed_count} images processed...")
                elif processed_count % 10 == 0:
                    print(f"   ✅ {result['Filename']} - Height: {result['Plant Height (pixels)']}px, Green pixels: {result['Green Pixel Count']}")
            
        print(f"   📊 Completed {folder_name}: {images_processed_in_folder} images processed")

    print(f"\n🎯 FAST TEST COMPLETE!")
    print(f"Total images processed: {processed_count}")
    print(f"Malformed: {malformed}, Unreadable: {unreadable}")

    if results:
        df = pd.DataFrame(results)
        p = Path(csv_file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
        print(f"✔ Results saved to: {p}")
        
        # Create statistical analysis
        create_statistical_analysis(df, csv_file_path)
        
        # Clean up visualization files
        cleanup_visualizations(created_visualizations)
        
    else:
        print("❌ No results to save")


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
