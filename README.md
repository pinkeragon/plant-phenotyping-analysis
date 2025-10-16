# Viserion Heat Test Analysis

A Python script for automated plant phenotyping analysis that processes image data to extract plant measurements and color information.

## Features

- **File Management**: Automatically renames image files to standardized format
- **Image Analysis**: Extracts plant height, width, and color distribution from images
- **Data Export**: Saves results to CSV format for further analysis
- **Batch Processing**: Processes multiple timepoints and flats efficiently

## Requirements

- Python 3.8+
- OpenCV for image processing
- pandas for data manipulation
- numpy for numerical operations
- scipy for image analysis

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Update the `top_level_folder_path` variable in the script to point to your data directory
2. Modify the `timepoints` and `flats` dictionaries to match your experimental setup
3. Run the script:
   ```bash
   python viserion_heat_analysis.py
   ```

## File Structure Expected

```
Your Data Folder/
├── T0/
│   ├── T0 flat A/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── T0 flat B/
│   └── ...
└── results.csv (generated)
```

## Output

The script generates a CSV file with the following columns:
- Date
- Flat
- Prefix
- Well
- Rotation
- Filename
- Plant Height (pixels)
- Plant Width (pixels)
- Total Pixel Count
- Yellow Pixel Count
- Green Pixel Count
- Red Pixel Count

## Configuration

Edit the following variables in the main() function:
- `top_level_folder_path`: Path to your image data
- `timepoints`: List of timepoint folders to process
- `flats`: Dictionary mapping flat folder names to prefixes
