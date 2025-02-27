# Video Processing Pipeline

A modular video processing pipeline designed for the VLM (MiniCPM-V) project. This pipeline efficiently decodes videos and prepares frame data for downstream vision language models.

## Overview

This pipeline provides a flexible framework for processing video data through a series of operations:

1. **Video Decoding and Sampling**: Extracts frames from videos at specified intervals
2. **Crop Generation**: Resizes frames and generates strategic crops for detailed analysis

The architecture follows a modular design pattern where each processing step can be chained together to form a complete pipeline.

## Requirements

- Python 3.8+
- PyTorch
- torchcodec
- NumPy
- Pillow

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Configuration

The pipeline is configured through a JSON file (`config.json`) with the following structure:

```json
{
    "video_decoder": {
        "input_path": "/path/to/video.mp4",
        "input_resolution": [720, 1280],
        "dimension_order": "NHWC",
        "codec": "h264",
        "num_output_frames": 10,
        "device": "cpu",
        "save_output": true
    },
    "crop_maker": {
        "num_crops": 2,
        "global": {
            "resize": [336, 602]
        },
        "refine": {
            "resize": [476, 840]
        },
        "crop": {
            "best_grid": [1, 2]
        },
        "save_output": true
    }
}
```

### Configuration Parameters

#### Video Decoder
- `input_path`: Path to the input video file
- `input_resolution`: Expected resolution of the input video [height, width]
- `dimension_order`: Tensor dimension order (NHWC or NCHW)
- `codec`: Video codec (e.g., h264)
- `num_output_frames`: Number of frames to extract
- `device`: Processing device (cpu or cuda)
- `save_output`: Whether to save intermediate outputs

#### Crop Maker
- `num_crops`: Number of crops to generate
- `global.resize`: Dimensions for global resize [height, width]
- `refine.resize`: Dimensions for refined crops [height, width]
- `crop.best_grid`: Grid division for cropping [rows, columns]
- `save_output`: Whether to save intermediate outputs

## Usage

Run the pipeline with:

```bash
python main.py
```

The pipeline will:
1. Decode the video and sample frames at regular intervals
2. Generate a global resized view of each frame
3. Create additional crops of areas of interest
4. Save outputs to the `outputs/` directory (if configured)

## Pipeline Architecture

The pipeline follows a modular design with a processing chain:

- `BasePipelineOp`: Abstract base class that defines the interface for all pipeline operations
- `DecodeAndSample`: Decodes video and extracts frames
- `CropMaker`: Resizes frames and creates crops

Each processor can be configured independently and chained together using the `next_processor` parameter.

## Output

Processed frames are saved as PyTorch tensors in pickle format in the `outputs/` directory:

- `decoder_output.pkl`: Raw decoded frames
- `crop_maker_output.pkl`: Resized and cropped frames

## Development

To extend the pipeline with new operations:

1. Create a new class that inherits from `BasePipelineOp`
2. Implement the `process()` and `save_output()` methods
3. Update the configuration file to include settings for your new processor
4. Chain it into the pipeline in `main.py`
