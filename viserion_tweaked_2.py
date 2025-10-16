#!/usr/bin/env python3
"""
viserion_tweaked_2_copy.py - Full version that processes ALL images per folder
"""

import numpy, pandas, matplotlib, seaborn, scipy, cv2
print("All good! (FULL DATASET VERSION)")

#Block One - FULL VERSION (Skip file renaming for analysis)
import os
import re
import shutil
from pathlib import Path

print("Skipping file renaming, going straight to analysis...")

#Block Two - Enhanced Image Analysis (FULL VERSION)
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import label

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
csv_file_path = os.path.join(top_level_folder_path, f"{folder_name}_FULL_ANALYSIS.csv")

PATTERN = re.compile(r"^(?P<date>\d{8})_(?P<prefix>[A-Z])(?P<well>\d{1,2})_(?P<rot>\d+)$", re.IGNORECASE)
results = []
malformed = 0
unreadable = 0
processed_count = 0

# FULL VERSION: Process ALL images in each folder
print("🔍 FULL MODE: Processing ALL images in each date folder...")

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
    
    jpg_files = [f for f in files if f.lower().endswith('.jpg')]
    images_processed_in_folder = 0
    
    for filename in jpg_files:
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

        # Resize for processing
        height, width = image.shape[:2]
        if width > 800:  # Slightly larger for better quality
            scale = 800 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))

        total_pixel_count = int(image.shape[0] * image.shape[1])
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # HSV ranges
        lower_yellow = np.array([25, 50, 50]); upper_yellow = np.array([30, 255, 255])
        lower_green = np.array([30, 20, 20]); upper_green = np.array([60, 255, 255])
        lower_red1 = np.array([0, 50, 50]); upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50]); upper_red2 = np.array([180, 255, 255])

        yellow_mask = cv2.inRange(image_hsv, lower_yellow, upper_yellow)
        green_mask = cv2.inRange(image_hsv, lower_green, upper_green)
        red_mask = cv2.bitwise_or(cv2.inRange(image_hsv, lower_red1, upper_red1),
                                 cv2.inRange(image_hsv, lower_red2, upper_red2))

        plant_height, plant_width, min_x, max_x, min_y, max_y = calculate_height_width(green_mask)

        yellow_count = int(np.count_nonzero(yellow_mask))
        green_count = int(np.count_nonzero(green_mask))
        red_count = int(np.count_nonzero(red_mask))
        
        # Calculate normalized yellow ratio (yellow / (yellow + green))
        plant_pixels = yellow_count + green_count
        normalized_yellow = yellow_count / plant_pixels if plant_pixels > 0 else 0

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
            "Yellow Pixel Count": yellow_count,
            "Green Pixel Count": green_count,
            "Red Pixel Count": red_count,
            "Normalized Yellow Ratio": normalized_yellow,
        })
        
        processed_count += 1
        images_processed_in_folder += 1
        
        # Progress tracking for large datasets
        if processed_count % 100 == 0:
            print(f"   🔄 Progress: {processed_count} images processed...")
        elif processed_count % 25 == 0:
            print(f"   ✅ Progress milestone: {processed_count} images")
        
    print(f"   📊 Completed {folder_name}: {images_processed_in_folder} images processed")

print(f"\n🎯 FULL ANALYSIS COMPLETE!")
print(f"Total images processed: {processed_count}")
print(f"Malformed: {malformed}, Unreadable: {unreadable}")

if results:
    df = pd.DataFrame(results)
    
    # 🗓️ FILTER FOR 2025 DATA ONLY
    print("\n🗓️ Filtering for 2025 data only...")
    df['Date_Extracted'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    df_2025 = df[df['Date_Extracted'].dt.year == 2025].copy()
    
    print(f"📊 Total images before filtering: {len(df)}")
    print(f"📊 Images from 2025: {len(df_2025)}")
    print(f"📅 2025 date range: {df_2025['Date_Extracted'].min().strftime('%Y-%m-%d')} to {df_2025['Date_Extracted'].max().strftime('%Y-%m-%d')}")
    
    # Use the filtered 2025 data for analysis
    df = df_2025
    
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
    
    # Use earliest date as day 0 (baseline) - Date_Extracted already created during filtering
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
    
    # Create comprehensive combined visualization (both static and time series) - Optimized for macOS screen
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(20, 8))
    
    # STATIC ANALYSIS PLOTS (Row 1)
    
    # Plot 1: Box plot of foliage percentage distribution
    plt.subplot(2, 4, 1)
    groups = df['Group'].unique()
    foliage_data = [df[df['Group'] == group]['Foliage_Percentage'] for group in sorted(groups)]
    box_plot = plt.boxplot(foliage_data, tick_labels=sorted(groups), patch_artist=True)
    
    # Color the boxes green
    for patch in box_plot['boxes']:
        patch.set_facecolor('lightgreen')
        patch.set_alpha(0.7)
    
    plt.title('Foliage % Distribution', fontweight='bold', fontsize=10)
    plt.xlabel('Group (Flat)')
    plt.ylabel('Foliage % (Green Pixels)')
    plt.grid(True, alpha=0.3)
    
    # TIME SERIES PLOTS
    
    # Plot 2: Foliage percentage over time by group
    plt.subplot(2, 4, 2)
    colors = ['#E691A8', '#5A9FD4', '#7AE582', '#F4A460', '#C48FC4', '#D4B86A']  # Darker pastel colors
    for i, group in enumerate(sorted(time_data['Group'].unique())):
        group_data = time_data[time_data['Group'] == group]
        plt.errorbar(group_data['Days_Post_Treatment'], group_data['Foliage_Mean'], 
                    yerr=group_data['Foliage_Std'], 
                    label=f'Group {group}', marker='o', linewidth=2, 
                    capsize=5, color=colors[i % len(colors)])
    
    plt.title('Foliage % Over Time by Group', fontweight='bold', fontsize=10)
    plt.xlabel('Days Post-Treatment')
    plt.ylabel('Foliage % (Green Pixels)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Green pixel count over time
    plt.subplot(2, 4, 3)
    for i, group in enumerate(sorted(time_data['Group'].unique())):
        group_data = time_data[time_data['Group'] == group]
        plt.errorbar(group_data['Days_Post_Treatment'], group_data['Green_Mean']/1000, 
                    yerr=group_data['Green_Std']/1000, 
                    label=f'Group {group}', marker='o', linewidth=2, 
                    capsize=5, color=colors[i % len(colors)])
    
    plt.title('Green Pixel Count Over Time', fontweight='bold', fontsize=10)
    plt.xlabel('Days Post-Treatment')
    plt.ylabel('Green Pixels (thousands)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Yellow pixel count over time
    plt.subplot(2, 4, 4)
    for i, group in enumerate(sorted(time_data['Group'].unique())):
        group_data = time_data[time_data['Group'] == group]
        plt.errorbar(group_data['Days_Post_Treatment'], group_data['Yellow_Mean']/1000, 
                    yerr=group_data['Yellow_Std']/1000, 
                    label=f'Group {group}', marker='o', linewidth=2, 
                    capsize=5, color=colors[i % len(colors)])
    
    plt.title('Yellow Pixel Count Over Time', fontweight='bold', fontsize=10)
    plt.xlabel('Days Post-Treatment')
    plt.ylabel('Yellow Pixels (thousands)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Plant height over time
    plt.subplot(2, 4, 5)
    for i, group in enumerate(sorted(time_data['Group'].unique())):
        group_data = time_data[time_data['Group'] == group]
        plt.errorbar(group_data['Days_Post_Treatment'], group_data['Height_Mean'], 
                    yerr=group_data['Height_Std'], 
                    label=f'Group {group}', marker='o', linewidth=2, 
                    capsize=5, color=colors[i % len(colors)])
    
    plt.title('Plant Height Over Time', fontweight='bold', fontsize=10)
    plt.xlabel('Days Post-Treatment')
    plt.ylabel('Height (pixels)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Foliage change from baseline
    plt.subplot(2, 4, 6)
    for i, group in enumerate(sorted(time_data['Group'].unique())):
        group_data = time_data[time_data['Group'] == group].sort_values('Days_Post_Treatment')
        if len(group_data) > 1:
            baseline = group_data['Foliage_Mean'].iloc[0]
            change = group_data['Foliage_Mean'] - baseline
            plt.plot(group_data['Days_Post_Treatment'], change, 
                    label=f'Group {group}', marker='o', linewidth=2, 
                    color=colors[i % len(colors)])
    
    plt.title('Foliage Change from Baseline', fontweight='bold', fontsize=10)
    plt.xlabel('Days Post-Treatment')
    plt.ylabel('Foliage % Change')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 7: Plant height distribution by group
    plt.subplot(2, 4, 7)
    groups = df['Group'].unique()
    height_data = [df[df['Group'] == group]['Plant Height (pixels)'] for group in sorted(groups)]
    box_plot = plt.boxplot(height_data, tick_labels=sorted(groups), patch_artist=True)
    
    # Color the boxes
    for patch in box_plot['boxes']:
        patch.set_facecolor('lightcoral')
        patch.set_alpha(0.7)
    
    plt.title('Plant Height Distribution', fontweight='bold', fontsize=10)
    plt.xlabel('Group (Flat)')
    plt.ylabel('Height (pixels)')
    plt.grid(True, alpha=0.3)
    
    # Plot 8: Normalized Yellow Ratio over time
    plt.subplot(2, 4, 8)
    for i, group in enumerate(sorted(time_data['Group'].unique())):
        group_data = time_data[time_data['Group'] == group]
        plt.errorbar(group_data['Days_Post_Treatment'], group_data['Normalized_Yellow_Mean'], 
                    yerr=group_data['Normalized_Yellow_Std'], 
                    label=f'Group {group}', marker='o', linewidth=2, 
                    capsize=5, color=colors[i % len(colors)])
    
    plt.title('Normalized Yellow Ratio Over Time', fontweight='bold', fontsize=10)
    plt.xlabel('Days Post-Treatment')
    plt.ylabel('Yellow/(Yellow+Green) Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Adjust spacing between subplots for better readability and wider plots
    plt.subplots_adjust(hspace=0.4, wspace=0.3, left=0.05, right=0.98, top=0.95, bottom=0.08)
    
    # Save the comprehensive combined plot
    combined_plot_path = p.parent / f"{p.stem}_COMBINED_ANALYSIS.png"
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Combined analysis plots saved to: {combined_plot_path}")
    
    plt.show()
    
else:
    print("❌ No results to save")
