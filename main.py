# main.py
import json
from video_pipeline.decoding_and_sampling import DecodeAndSample
from video_pipeline.video_preprocessing import VideoPreprocessing
from video_pipeline.vfm_preprocessing import VFMPreprocessing
from logger import logger

# Load configuration
with open("config.json", "r") as f:
    config = json.load(f)

logger.info("Starting video processing pipeline...")

# Build the pipeline dynamically using the config file
pipeline = (
    DecodeAndSample(config["video_decoder"], next_processor= \
    VideoPreprocessing(config["video_preprocessing"], next_processor= \
    VFMPreprocessing(config['vfm_preprocessing'], next_processor= \
    None
    )))
)
# Run the pipeline
processed_frames = pipeline.process()
