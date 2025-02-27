# main.py
import json
from video_decoder.decoding_and_sampling import DecodeAndSample
from video_decoder.crop_maker import CropMaker
from logger import logger

# Load configuration
with open("config.json", "r") as f:
    config = json.load(f)

logger.info("Starting video processing pipeline...")

# Build the pipeline dynamically using the config file
pipeline = (
    DecodeAndSample(config["video_decoder"], next_processor= \
    CropMaker(config["crop_maker"], next_processor= \
    None),
    )
)
# Run the pipeline
processed_frames = pipeline.process()
logger.info(f"Final output: {len(processed_frames)} frames processed")  # Temporary
