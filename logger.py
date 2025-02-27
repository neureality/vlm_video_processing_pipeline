# logger.py
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more detailed logs
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        # logging.FileHandler("pipeline.log"),  # Save logs to a file
        logging.StreamHandler(),  # Print logs to console
    ],
)

# Get a logger instance
logger = logging.getLogger("VideoPipeline")
