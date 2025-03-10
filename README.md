# Video Processing Pipeline

A modular video processing pipeline designed for Vision Language Models (particularly MiniCPM-V) that efficiently decodes videos and prepares frame data through a multi-stage transformation process optimized for ViT (Siglip) inference.

## Overview

This pipeline provides a flexible framework for processing video data through a series of specialized operations:

1. **Video Decoding and Sampling**: Extracts frames from videos at uniform intervals
2. **Video Preprocessing**: Performs spatial transformations including resizing and strategic cropping
3. **VFM Preprocessing**: Prepares the processed frames for the ViT (Siglip) by handling patching, padding, and attention masks

The architecture follows a modular design pattern where each processing step is implemented as a separate component that can be chained together to form a complete pipeline. Each stage transforms the data into a format required by the next stage, ultimately producing ViT (Siglip)-Ready tensors.

## Requirements

- Python 3.8+
- PyTorch
- torchvision
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
    "video_preprocessing": {
        "num_crops": 2,
        "resize_global": [336, 602],
        "resize_refine": [476, 840],
        "crop_best_grid": [1, 2],
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "patch_size": 14,
        "device": "cpu",
        "save_output": true
    },
    "vfm_preprocessing": {
        "dtype": "bfloat16",
        "batch_size": 30,
        "patch_size": 14,
        "num_patches_in_global_slice": 1032,
        "num_patches_in_crop_slice": 1020,
        "device": "cpu",
        "save_output": true
    }
}
```

### Configuration Parameters

#### Video Decoder (`video_decoder`)
- `input_path`: Path to the input video file
- `input_resolution`: Expected resolution of the input video [height, width]
- `dimension_order`: Tensor dimension order (NHWC or NCHW)
- `codec`: Video codec (e.g., h264)
- `num_output_frames`: Number of frames to extract
- `device`: Processing device (cpu or cuda)
- `save_output`: Whether to save intermediate outputs

#### Video Preprocessing (`video_preprocessing`)
- `num_crops`: Number of crops to generate
- `resize_global`: Dimensions for global resize [height, width]
- `resize_refine`: Dimensions for refined crops [height, width]
- `crop_best_grid`: Grid division for cropping [rows, columns]
- `mean`: Mean values for normalization [R, G, B]
- `std`: Standard deviation values for normalization [R, G, B]
- `patch_size`: Size of patches for VLM processing (typically 14 or 16)
- `device`: Processing device (cpu or cuda)
- `save_output`: Whether to save intermediate outputs

#### VFM Preprocessing (`vfm_preprocessing`)
- `dtype`: Data type for output tensors (e.g., bfloat16)
- `batch_size`: Batch size for processing
- `patch_size`: Size of image patches (should match the video_preprocessing patch_size)
- `num_patches_in_global_slice`: Number of patches in the global slice
- `num_patches_in_crop_slice`: Number of patches in each crop slice
- `device`: Processing device (cpu or cuda)
- `save_output`: Whether to save intermediate outputs

## Usage

Run the pipeline with:

```bash
python main.py
```

The pipeline will:
1. Decode the video and sample frames at regular intervals
2. Process the frames through video preprocessing:
   - Generate a global resized view of each frame
   - Create additional two crops (vertical slicing)
   - Normalize pixel values
   - Reshape frames into patches
3. Process the frames through VFM preprocessing:
   - Convert to specified data type
   - Prepare attention masks
   - Format tensors for ViT (Siglip) inference
4. Save outputs to the `outputs/` directory (if configured)

## Pipeline Architecture

The pipeline follows a modular design with a processing chain:

- `BasePipelineOp`: Abstract base class that defines the interface for all pipeline operations
- `DecodeAndSample`: Decodes video and extracts frames at uniform intervals
- `VideoPreprocessing`: Performs spatial transformations on frames
- `VFMPreprocessing`: Prepares processed frames for ViT (Siglip) inference

Each processor can be configured independently and chained together using the `next_processor` parameter. The pipeline uses a sequential execution model where the output of one processor becomes the input to the next.

## Output

Processed data is saved as PyTorch tensors in pickle format in the `outputs/` directory:

- `decoder_output.pkl`: Raw decoded frames
- `pixel_values.pkl`: Processed image patches
- `tgt_sizes.pkl`: Target sizes for each processed image patch
- `image_sizes.pkl`: Original image sizes
- `all_pixel_values.pkl`: Padded pixel values
- `patch_attn_mask.pkl`: Attention masks for patches

## Logging

The pipeline uses Python's built-in logging module to provide information about the processing steps. You can adjust the logging level in `logger.py` to get more or less detailed information.