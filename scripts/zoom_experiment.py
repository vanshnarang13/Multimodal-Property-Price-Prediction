"""
zoom_experiment.py

Script to experiment with different zoom levels (18-21) for satellite images.
Downloads the same location at multiple zoom levels for comparison.
"""

import os
import requests
import pandas as pd
from pathlib import Path
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Google Maps API Key
API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

# Input/output settings
INPUT_CSV = "/Users/vanshnarang/Desktop/cdc open project/train(1).csv"
OUTPUT_DIR = "/Users/vanshnarang/Desktop/cdc open project/zoom_experiments"

LAT_COLUMN = "lat"
LON_COLUMN = "long"

# Zoom levels to test
ZOOM_LEVELS = [18, 19, 20, 21]

# Image settings
IMAGE_SIZE = "640x640"
MAP_TYPE = "satellite"

# Number of sample locations to test (from your dataset)
NUM_SAMPLES = 3

GOOGLE_MAPS_STATIC_API_URL = "https://maps.googleapis.com/maps/api/staticmap"


def load_sample_coordinates(filepath: str, n_samples: int) -> list:
    """Load sample coordinates from the CSV."""
    df = pd.read_csv(filepath)

    if LAT_COLUMN not in df.columns or LON_COLUMN not in df.columns:
        raise ValueError(f"Columns '{LAT_COLUMN}' or '{LON_COLUMN}' not found")

    # Drop rows with invalid coordinates
    df = df.dropna(subset=[LAT_COLUMN, LON_COLUMN])

    # Take first n samples
    samples = df.head(n_samples)[[LAT_COLUMN, LON_COLUMN]].to_dict('records')

    logger.info(f"Loaded {len(samples)} sample coordinates")
    return samples


def download_image_at_zoom(lat: float, lon: float, zoom: int, filepath: Path) -> bool:
    """Download a satellite image at a specific zoom level."""
    params = {
        "center": f"{lat},{lon}",
        "zoom": zoom,
        "size": IMAGE_SIZE,
        "maptype": MAP_TYPE,
        "key": API_KEY
    }

    try:
        response = requests.get(
            GOOGLE_MAPS_STATIC_API_URL,
            params=params,
            timeout=30
        )
        response.raise_for_status()

        # Check if response is an image
        content_type = response.headers.get('content-type', '')
        if 'image' not in content_type:
            logger.error(f"API error at zoom {zoom}: {response.text[:200]}")
            return False

        # Save the image
        with open(filepath, 'wb') as f:
            f.write(response.content)

        logger.info(f"✓ Downloaded zoom {zoom}: {filepath.name}")
        return True

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download at zoom {zoom}: {e}")
        return False


def run_zoom_experiment():
    """Download sample images at different zoom levels for comparison."""
    # Validate API key
    if not API_KEY:
        logger.error("Google Maps API key is not set. Please set 'GOOGLE_MAPS_API_KEY' in your .env file.")
        return

    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load sample coordinates
    samples = load_sample_coordinates(INPUT_CSV, NUM_SAMPLES)

    logger.info(f"\nStarting zoom level experiment (zooms {min(ZOOM_LEVELS)}-{max(ZOOM_LEVELS)})")
    logger.info(f"Testing {len(samples)} locations × {len(ZOOM_LEVELS)} zoom levels = {len(samples) * len(ZOOM_LEVELS)} images\n")

    # For each sample location
    for idx, coord in enumerate(samples):
        lat = coord[LAT_COLUMN]
        lon = coord[LON_COLUMN]

        logger.info(f"\n📍 Location {idx + 1}/{len(samples)}: ({lat:.6f}, {lon:.6f})")

        # Create a subdirectory for this location
        location_dir = output_dir / f"location_{idx + 1}"
        location_dir.mkdir(exist_ok=True)

        # Download at each zoom level
        for zoom in ZOOM_LEVELS:
            filename = f"zoom_{zoom}_lat_{lat:.6f}_lon_{lon:.6f}.png"
            filepath = location_dir / filename

            # Skip if already exists
            if filepath.exists():
                logger.info(f"⊙ Skipped zoom {zoom}: already exists")
                continue

            download_image_at_zoom(lat, lon, zoom, filepath)

    logger.info(f"\nExperiment complete! Images saved to: {OUTPUT_DIR}")
    logger.info(f"\nCompare the images in each location_X folder to see the difference between zoom levels.")

    # Create a summary
    summary_path = output_dir / "README.txt"
    with open(summary_path, 'w') as f:
        f.write("Zoom Level Experiment Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Zoom levels tested: {ZOOM_LEVELS}\n")
        f.write(f"Number of locations: {len(samples)}\n")
        f.write(f"Image size: {IMAGE_SIZE}\n\n")
        f.write("Zoom Level Guide:\n")
        f.write("  18 - Regional view, shows neighborhood area\n")
        f.write("  19 - Detailed view, individual buildings visible\n")
        f.write("  20 - Very detailed, can see building features\n")
        f.write("  21 - Maximum detail, highest resolution\n\n")
        f.write("Each location_X folder contains the same coordinates\n")
        f.write("at different zoom levels for comparison.\n")

    logger.info(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    run_zoom_experiment()
