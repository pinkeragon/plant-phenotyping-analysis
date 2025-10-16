#!/usr/bin/env python3
"""
Viserion_Tweaked_2_FAST.py - Test version that processes only first few images per flat
"""

import numpy, pandas, matplotlib, seaborn, scipy, cv2
print("All good! (FAST TEST VERSION)")

#Block One - FAST VERSION (Skip file renaming for testing)
import os
import re
import shutil
import subprocess
import time
import glob
import datetime
from pathlib import Path

print("Skipping file renaming, going straight to analysis...")

#Block Two - Enhanced Image Analysis (FAST VERSION)
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import label

def apply_shape_filtering(mask):
    """
    Appl    # Save the comprehensive combined plot
         # Save the comprehensive combined plo    # Save the comprehensive combined plot
    combined_plot_path = p.parent / f"{p.stem}_COMBINED_ANALYSIS.png"
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Combined analysis plots saved to: {combined_plot_path}")
    
    plt.show()
    
    # 🗑️ AUTO-DELETE VISUALIZATION FILES AFTER ANALYSIS
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
    
else:
    print("❌ No results to save")bined_plot_path = p.parent / f"{p.stem}_COMBINED_ANALYSIS.png"
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Combined analysis plots saved to: {combined_plot_path}")
    
    plt.show()
    
    # 🎨 AUTOMATICALLY OPEN GREEN PIXEL VISUALIZATIONS
    print("\n🎨 Opening green pixel visualizations...")
    
    # Find all visualization files in the output directory
    viz_pattern = os.path.join(top_level_folder_path, "*_visualization.png")
    viz_files = glob.glob(viz_pattern)
    
    if viz_files:
        print(f"📂 Found {len(viz_files)} visualization files")
        
        # Sort by filename to show in chronological order
        viz_files.sort()
        
        # Open each visualization file with a small delay
        for i, viz_file in enumerate(viz_files):
            filename = os.path.basename(viz_file)
            print(f"   🖼️ Opening: {filename}")
            
            try:
                # Use subprocess for more reliable file opening
                subprocess.run(['open', viz_file], check=True)
                
                # Small delay to prevent overwhelming the system
                if i < len(viz_files) - 1:  # Don't delay after the last file
                    time.sleep(0.5)
                    
            except subprocess.CalledProcessError as e:
                print(f"   ⚠️ Failed to open: {filename} - {e}")
        
        print("✅ All green pixel visualizations opened!")
    else:
        print("⚠️ No visualization files found to open")
    
else:
    print("❌ No results to save")he comprehensive combined plot
    combined_plot_path = p.parent / f"{p.stem}_COMBINED_ANALYSIS.png"
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Combined analysis plots saved to: {combined_plot_path}")
    
    plt.show()
    
    # Automatically open all visualization files
    viz_pattern = os.path.join(top_level_folder_path, "*_visualization.png")
    viz_files = glob.glob(viz_pattern)
    
    if viz_files:
        print(f"📂 Found {len(viz_files)} visualization files")
        
        # Sort by filename to show in chronological order
        viz_files.sort()
        
        # Open each visualization file with a small delay
        for i, viz_file in enumerate(viz_files):
            filename = os.path.basename(viz_file)
            print(f"   🖼️ Opening: {filename}")
            
            try:
                # Use subprocess for more reliable file opening
                subprocess.run(['open', viz_file], check=True)
                
                # Small delay to prevent overwhelming the system
                if i < len(viz_files) - 1:  # Don't delay after the last file
                    time.sleep(0.5)
                    
            except subprocess.CalledProcessError:
                print(f"   ⚠️ Failed to open: {filename}")
        
        print("✅ All green pixel visualizations opened!")
    else:
        print("⚠️ No visualization files found to open")
    
else:
    print("❌ No results to save")d_plot_path = p.parent / f"{p.stem}_COMBINED_ANALYSIS.png"
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
    print(f"🎯 Combined analysis plots saved to: {combined_plot_path}")
    
    plt.show()
    
    # 🎨 AUTOMATICALLY OPEN GREEN PIXEL VISUALIZATIONS
    print("\n🎨 Opening green pixel visualizations...")
    
    # Find all visualization files in the output directory
    import glob
    import subprocess
    import time
    
    viz_pattern = os.path.join(top_level_folder_path, "*_visualization.png")
    viz_files = glob.glob(viz_pattern)
    
    if viz_files:
        print(f"📂 Found {len(viz_files)} visualization files")
        
        # Sort by filename to show in chronological order
        viz_files.sort()
        
        # Open each visualization file with a small delay
        for i, viz_file in enumerate(viz_files):
            filename = os.path.basename(viz_file)
            print(f"   🖼️ Opening: {filename}")
            
            try:
                # Use subprocess for more reliable file opening
                subprocess.run(['open', viz_file], check=True)
                
                # Small delay to prevent overwhelming the system
                if i < len(viz_files) - 1:  # Don't delay after the last file
                    time.sleep(0.5)
                    
            except subprocess.CalledProcessError:
                print(f"   ⚠️ Failed to open: {filename}")
        
        print("✅ All green pixel visualizations opened!")
    else:
        print("⚠️ No visualization files found to open")
    
else:
    print("❌ No results to save")ased filtering to remove excess pixels and clean up the plant mask
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

# Configuration - Updated to use Viserion Heat Test folder
top_level_folder_path = r"/Users/olivia/Library/CloudStorage/GoogleDrive-olivia.simon@temporalag.com/My Drive/Purps Phenotypes/Viserion Heat Test 1 - For Olivia"
folder_name = os.path.basename(top_level_folder_path)
csv_file_path = os.path.join(top_level_folder_path, f"{folder_name}_FAST_TEST.csv")

PATTERN = re.compile(r"^(?P<date>\d{8})_(?P<prefix>[A-Z])(?P<well>\d{1,2})_(?P<rot>\d+)$", re.IGNORECASE)
results = []
malformed = 0
unreadable = 0
processed_count = 0

MAX_IMAGES_PER_FOLDER = 5  # FAST VERSION: Only process 5 images per folder for testing

print("🔍 FAST MODE: Processing only first 5 images per date folder...")

# Counter to track folder processing for visualization selection
folder_counter = 0

# List to track created visualization files for cleanup
created_visualizations = []

# Process images organized by date folders
for root, dirs, files in os.walk(top_level_folder_path):
    # Skip the root directory itself and only process date folders
    if root == top_level_folder_path:
        continue
    
    folder_name = os.path.basename(root)
    
    # Only process folders that look like dates (8 digits)
    if not folder_name.isdigit() or len(folder_name) != 8:
        continue
    
    print(f"📁 Processing date folder: {folder_name}")
    
    # Increment folder counter
    folder_counter += 1
    
    jpg_files = [f for f in files if f.lower().endswith('.jpg')]
    images_processed_in_folder = 0
    
    # Only create visualization for every 2nd folder (5 total from 10 folders)
    create_visualization_for_folder = (folder_counter % 2 == 0)  # Every 2nd folder
    
    # Randomly shuffle the files so we get different images each run
    import random
    random.shuffle(jpg_files)
    
    for filename in jpg_files:
        if images_processed_in_folder >= MAX_IMAGES_PER_FOLDER:
            print(f"   ⚡ Stopping at {MAX_IMAGES_PER_FOLDER} images for {folder_name}")
            break
            
        filename_without_ext = os.path.splitext(filename)[0]
        
        m = PATTERN.match(filename_without_ext)
        if not m:
            malformed += 1
            continue

        date_str = m.group("date")
        prefix = m.group("prefix").upper()
        well = int(m.group("well"))
        rotation = m.group("rot")

        image_path = os.path.join(root, filename)
        image = cv2.imread(image_path)

        if image is None:
            unreadable += 1
            continue

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

        # Create visualization for first suitable image in shuffled folder (random each run)
        if create_visualization_for_folder and green_count > 1000:  # Only if significant plant detected
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
            
            create_visualization_for_folder = False  # Only create one per folder

        results.append({
            "Date": date_str,
            "Flat": folder_name,  # Using date folder as identifier
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
        })
        
        processed_count += 1
        images_processed_in_folder += 1
        
        # Progress tracking for large datasets
        if processed_count % 50 == 0:
            print(f"   ⚡ Progress: {processed_count} images processed...")
        elif processed_count % 10 == 0:
            print(f"   ✅ {filename_without_ext} - Height: {plant_height}px, Green pixels: {np.count_nonzero(green_mask)}")
        
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
    
    # 📊 CREATE FOLIAGE VISUALIZATION
    print("\n📊 Creating foliage visualization...")
    
    # Calculate foliage (green) pixel percentage
    df['Foliage_Percentage'] = (df['Green Pixel Count'] / df['Total Pixel Count']) * 100
    
    # Extract group information from prefix and create pot identifier
    df['Group'] = df['Prefix']
    df['Pot_ID'] = df['Prefix'] + df['Well'].astype(str)
    
    # Create a simple plot showing foliage percentage by group
    import matplotlib.pyplot as plt
    
    # Group data by Group and calculate mean foliage percentage
    group_stats = df.groupby('Group')['Foliage_Percentage'].agg(['mean', 'std', 'count']).reset_index()
    

    
    # Print summary to console
    print("\n🌿 FOLIAGE SUMMARY:")
    print(group_stats.round(2))
    
    # 📈 CREATE LINE PLOTS (Time Series Analysis)
    print("\n📈 Creating line plots for foliage analysis over time...")
    
    # Add days post-treatment calculation
    df['Date_Extracted'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    
    # Use earliest date as day 0 (baseline)
    baseline_date = df['Date_Extracted'].min()
    df['Days_Post_Treatment'] = (df['Date_Extracted'] - baseline_date).dt.days
    
    print(f"📅 Baseline date: {baseline_date.strftime('%Y-%m-%d')}")
    print(f"📅 Date range: {df['Days_Post_Treatment'].min()} to {df['Days_Post_Treatment'].max()} days")
    
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
    import matplotlib.pyplot as plt
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
        patch.set_facecolor(group_colors[group_names[i]])
        patch.set_alpha(0.7)
    
    # Perform ANOVA
    f_stat, p_value = f_oneway(*foliage_data)
    
    # Add statistical annotations
    plt.title(f'Foliage % Distribution\nANOVA: F={f_stat:.2f}, p={p_value:.3f}', fontweight='bold', fontsize=10)
    
    # Add pairwise comparisons (simplified approach)
    if p_value < 0.05:
        max_y = max([max(data) for data in foliage_data])
        # Compare A vs D (typically most different)
        group_a_data = df[df['Group'] == 'A']['Foliage_Percentage']
        group_d_data = df[df['Group'] == 'D']['Foliage_Percentage']
        t_stat, p_val_ad = ttest_ind(group_a_data, group_d_data)
        
        if p_val_ad < 0.05:
            significance = '***' if p_val_ad < 0.001 else '**' if p_val_ad < 0.01 else '*'
            plt.plot([1, 4], [max_y + 0.5, max_y + 0.5], 'k-', linewidth=1)
            plt.text(2.5, max_y + 0.7, significance, ha='center', fontsize=12, fontweight='bold')
    
    plt.xlabel('Group')
    plt.ylabel('Foliage %')
    plt.grid(True, alpha=0.3)
    
    # TIME SERIES PLOTS - STANDARDIZED COLORS AND SCALES
    
    # Plot 2: Foliage percentage over time by group
    plt.subplot(2, 3, 2)
    for group in sorted(time_data['Group'].unique()):
        group_data = time_data[time_data['Group'] == group]
        plt.errorbar(group_data['Days_Post_Treatment'], group_data['Foliage_Mean'], 
                    yerr=group_data['Foliage_Std'], 
                    label=f'{group}', marker='o', linewidth=2, 
                    capsize=5, color=group_colors[group])
    
    plt.title('Foliage % Over Time by Group', fontweight='bold', fontsize=10)
    plt.xlabel('Days Post-Treatment')
    plt.ylabel('Foliage %')
    plt.ylim(0, 12)  # Standardized y-axis range for foliage %
    plt.xticks([0, 5, 10, 15, 20])  # Simplified tick labels
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Green pixel count over time
    plt.subplot(2, 3, 3)
    for group in sorted(time_data['Group'].unique()):
        group_data = time_data[time_data['Group'] == group]
        plt.errorbar(group_data['Days_Post_Treatment'], group_data['Green_Mean']/1000, 
                    yerr=group_data['Green_Std']/1000, 
                    label=f'{group}', marker='o', linewidth=2, 
                    capsize=5, color=group_colors[group])
    
    plt.title('Green Pixel Count Over Time', fontweight='bold', fontsize=10)
    plt.xlabel('Days Post-Treatment')
    plt.ylabel('Green Pixels (k)')
    plt.xticks([0, 5, 10, 15, 20])  # Simplified tick labels
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Plant height over time
    plt.subplot(2, 3, 4)
    for group in sorted(time_data['Group'].unique()):
        group_data = time_data[time_data['Group'] == group]
        plt.errorbar(group_data['Days_Post_Treatment'], group_data['Height_Mean'], 
                    yerr=group_data['Height_Std'], 
                    label=f'{group}', marker='o', linewidth=2, 
                    capsize=5, color=group_colors[group])
    
    plt.title('Plant Height Over Time', fontweight='bold', fontsize=10)
    plt.xlabel('Days Post-Treatment')
    plt.ylabel('Height (px)')
    plt.ylim(0, 400)  # Standardized y-axis range for height
    plt.xticks([0, 5, 10, 15, 20])  # Simplified tick labels
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Foliage change from baseline
    plt.subplot(2, 3, 5)
    for group in sorted(time_data['Group'].unique()):
        group_data = time_data[time_data['Group'] == group].sort_values('Days_Post_Treatment')
        if len(group_data) > 1:
            baseline = group_data['Foliage_Mean'].iloc[0]
            change = group_data['Foliage_Mean'] - baseline
            plt.plot(group_data['Days_Post_Treatment'], change, 
                    label=f'{group}', marker='o', linewidth=2, 
                    color=group_colors[group])
    
    plt.title('Foliage Change from Baseline', fontweight='bold', fontsize=10)
    plt.xlabel('Days Post-Treatment')
    plt.ylabel('Foliage % Change')
    plt.ylim(-3, 3)  # Standardized y-axis range for change
    plt.xticks([0, 5, 10, 15, 20])  # Simplified tick labels
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
        patch.set_facecolor(group_colors[group_names[i]])
        patch.set_alpha(0.7)
    
    # Perform ANOVA for height
    f_stat_h, p_value_h = f_oneway(*height_data)
    
    # Add statistical annotations
    plt.title(f'Plant Height Distribution\nANOVA: F={f_stat_h:.2f}, p={p_value_h:.3f}', fontweight='bold', fontsize=10)
    
    # Add pairwise comparisons for height
    if p_value_h < 0.05:
        max_y_h = max([max(data) for data in height_data])
        # Compare groups with largest difference
        group_a_height = df[df['Group'] == 'A']['Plant Height (pixels)']
        group_d_height = df[df['Group'] == 'D']['Plant Height (pixels)']
        t_stat_h, p_val_h_ad = ttest_ind(group_a_height, group_d_height)
        
        if p_val_h_ad < 0.05:
            significance_h = '***' if p_val_h_ad < 0.001 else '**' if p_val_h_ad < 0.01 else '*'
            plt.plot([1, 4], [max_y_h + 20, max_y_h + 20], 'k-', linewidth=1)
            plt.text(2.5, max_y_h + 30, significance_h, ha='center', fontsize=12, fontweight='bold')
    
    plt.xlabel('Group')
    plt.ylabel('Height (px)')
    plt.grid(True, alpha=0.3)
    
    # Adjust spacing between subplots for better readability and wider plots
    plt.subplots_adjust(hspace=0.4, wspace=0.3, left=0.05, right=0.98, top=0.95, bottom=0.08)
    
    # Save the comprehensive combined plot
    combined_plot_path = p.parent / f"{p.stem}_COMBINED_ANALYSIS.png"
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
    print(f"� Combined analysis plots saved to: {combined_plot_path}")
    
    plt.show()
    
else:
    print("❌ No results to save")
