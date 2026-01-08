import os
import requests
import pandas as pd
from pathlib import Path
import logging
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

INPUT_CSV = "/Users/vanshnarang/Desktop/cdc open project/data/test2.xlsx"

OUTPUT_DIR = "/Users/vanshnarang/Desktop/cdc open project/data/test_images_19"

LAT_COLUMN = "lat"
LON_COLUMN = "long"

# Image settings
ZOOM = 19
IMAGE_SIZE = "640x640"
MAP_TYPE = "satellite"

# Multithreading settings
NUM_WORKERS = 20  # Number of parallel download threads

MAX_IMAGES = None

# ============================================================

GOOGLE_MAPS_STATIC_API_URL = "https://maps.googleapis.com/maps/api/staticmap"

# Thread-safe counter for progress tracking
class ProgressTracker:
    def __init__(self, total):
        self.successful = 0
        self.failed = 0
        self.skipped = 0
        self.total = total
        self.lock = threading.Lock()
    
    def increment(self, status):
        with self.lock:
            if status == "success":
                self.successful += 1
            elif status == "failed":
                self.failed += 1
            elif status == "skipped":
                self.skipped += 1
            
            progress = self.successful + self.failed + self.skipped
            if progress % 100 == 0 or progress == self.total:
                logger.info(f"Progress: {progress}/{self.total} (Success: {self.successful}, Failed: {self.failed}, Skipped: {self.skipped})")


def load_csv(filepath: str) -> pd.DataFrame:
    """Load the housing CSV dataset."""
    df = pd.read_excel(filepath)
    
    if LAT_COLUMN not in df.columns:
        raise ValueError(f"Column '{LAT_COLUMN}' not found. Available: {list(df.columns)}")
    if LON_COLUMN not in df.columns:
        raise ValueError(f"Column '{LON_COLUMN}' not found. Available: {list(df.columns)}")
    
    logger.info(f"Loaded {len(df)} rows from {filepath}")
    return df


def generate_filename(lat: float, lon: float, index: int) -> str:
    """Generate a unique filename for the satellite image."""
    return f"img_{index:05d}_{lat:.6f}_{lon:.6f}.png"


def download_single_image(args):
    """Download a single satellite image. Used by thread pool."""
    idx, lat, lon, output_dir, tracker = args
    
    # Skip invalid coordinates
    if pd.isna(lat) or pd.isna(lon):
        tracker.increment("failed")
        return idx, None
    
    filename = generate_filename(lat, lon, idx)
    filepath = output_dir / filename
    
    # Skip if file already exists
    if filepath.exists():
        tracker.increment("skipped")
        return idx, str(filepath)
    
    params = {
        "center": f"{lat},{lon}",
        "zoom": ZOOM,
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
        
        content_type = response.headers.get('content-type', '')
        if 'image' not in content_type:
            tracker.increment("failed")
            return idx, None
        
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        tracker.increment("success")
        return idx, str(filepath)
        
    except requests.exceptions.RequestException as e:
        tracker.increment("failed")
        return idx, None


def download_satellite_images(df: pd.DataFrame) -> pd.DataFrame:
    """Download satellite images using multithreading."""
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = df.copy()
    df['image_path'] = None
    
    # Prepare download tasks
    if MAX_IMAGES:
        df = df.head(MAX_IMAGES)
    
    total = len(df)
    tracker = ProgressTracker(total)
    
    # Create list of arguments for each download
    download_args = [
        (idx, row[LAT_COLUMN], row[LON_COLUMN], output_dir, tracker)
        for idx, row in df.iterrows()
    ]
    
    logger.info(f"Starting download of {total} satellite images with {NUM_WORKERS} threads...")
    
    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(download_single_image, args): args[0] for args in download_args}
        
        for future in as_completed(futures):
            idx, filepath = future.result()
            if filepath:
                df.at[idx, 'image_path'] = filepath
    
    logger.info(f"Download complete! Success: {tracker.successful}, Failed: {tracker.failed}, Skipped: {tracker.skipped}")
    
    return df


def main():
    """Main execution function."""
    if not API_KEY:
        logger.error("Google Maps API key is not set. Please set 'GOOGLE_MAPS_API_KEY' in your .env file.")
        return
    
    df = load_csv(INPUT_CSV)
    
    result_df = download_satellite_images(df)
    
    output_csv = Path(OUTPUT_DIR) / "data_with_image_paths.csv"
    result_df.to_csv(output_csv, index=False)
    logger.info(f"Results saved to: {output_csv}")


if __name__ == "__main__":
    main()