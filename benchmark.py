import time
import json
from statistics import mean
from video_pipeline.decoding_and_sampling import DecodeAndSample
from video_pipeline.video_preprocessing import VideoPreprocessing
from video_pipeline.vfm_preprocessing import VFMPreprocessing
from logger import logger

def benchmark_pipeline(num_runs=5):
    """
    Benchmark the pipeline components with multiple runs and average the results.

    Args:
        num_runs (int): Number of runs to average the benchmark results.
    """
    # Load configuration
    with open("config.json", "r") as f:
        config = json.load(f)

    logger.info(f"Starting pipeline benchmarking with {num_runs} runs...")

    decode_times = []
    preprocess_times = []
    vfm_times = []

    for run in range(num_runs):
        logger.info(f"Run {run + 1}/{num_runs}...")

        # Benchmark DecodeAndSample
        start_time = time.time()
        decode_and_sample = DecodeAndSample(config["video_decoder"])
        frames = decode_and_sample.process()
        decode_times.append(time.time() - start_time)

        # Benchmark VideoPreprocessing
        start_time = time.time()
        video_preprocessing = VideoPreprocessing(config["video_preprocessing"])
        processed_frames = video_preprocessing.process(frames)
        preprocess_times.append(time.time() - start_time)

        # Benchmark VFMPreprocessing
        start_time = time.time()
        vfm_preprocessing = VFMPreprocessing(config["vfm_preprocessing"])
        vfm_model_input_dict = vfm_preprocessing.process(processed_frames)
        vfm_times.append(time.time() - start_time)

    # Calculate averages
    avg_decode_time = mean(decode_times)
    avg_preprocess_time = mean(preprocess_times)
    avg_vfm_time = mean(vfm_times)
    total_avg_time = avg_decode_time + avg_preprocess_time + avg_vfm_time

    logger.info("Pipeline benchmarking completed.")
    logger.info(f"Average DecodeAndSample time: {avg_decode_time:.2f} seconds.")
    logger.info(f"Average VideoPreprocessing time: {avg_preprocess_time:.2f} seconds.")
    logger.info(f"Average VFMPreprocessing time: {avg_vfm_time:.2f} seconds.")
    logger.info(f"Total average time: {total_avg_time:.2f} seconds.")

if __name__ == "__main__":
    benchmark_pipeline(num_runs=20)
