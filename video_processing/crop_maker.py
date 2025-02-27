# video_pipeline/resize_video.py
import os
from typing import List
import torch
import torch.nn.functional as F
from .base_pipeline_op import BasePipelineOp
from logger import logger


class CropMaker(BasePipelineOp):
    def __init__(self, config, next_processor=None):
        super().__init__(config, next_processor)
        self.is_save_output = config["save_output"]

    def process(self, _frames: List[torch.Tensor]) -> List[torch.Tensor]:
        logger.info("Resizing and Cropping the images")

        slice_images = []
        for frame in _frames:
            # Get 3 patches for each frame
            slice_images.append(
                self.resize_torch_image(frame, self.config["global"]["resize"])
            )
            patches: list = self._get_patches(frame)
            if len(patches) > 0:
                for i in range(len(patches)):
                    slice_images.append(patches[i])

        self.save_output(slice_images) if self.is_save_output else None  # Save output
        return (
            self.next_processor.process(slice_images)
            if self.next_processor
            else slice_images
        )

    def _get_patches(self, frame: torch.Tensor) -> torch.Tensor:
        """Get patches from the image"""
        # 1. Resize the image to refined size
        refined_frame = self.resize_torch_image(frame, self.config["refine"]["resize"])
        # 2. Slice the image into patches
        return self.split_into_patches(refined_frame, self.config["crop"]["best_grid"])

    @staticmethod
    def split_into_patches(image, grid):
        patches = []
        height, width, C = image.shape  # image: (H, W, C)
        patch_width = width // grid[1]
        patch_height = height // grid[0]

        for i in range(0, height, patch_height):
            for j in range(0, width, patch_width):
                # Slice the tensor: now height is axis 0, width is axis 1, and channels is axis 2
                patch = image[i : i + patch_height, j : j + patch_width, :]
                patches.append(patch)

        return patches

    @staticmethod
    def resize_torch_image(frame: torch.Tensor, size: tuple | list) -> torch.Tensor:
        """Resize the image"""

        need_permute_back = False
        if frame.shape[-1] == 3:
            frame = frame.permute(2, 0, 1)  # (H, W, C) to (C, H, W)
            need_permute_back = True

        # Ensure input is a 4D tensor (batch dimension)
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)

        # Perform resizing using bicubic interpolation
        resized = F.interpolate(
            frame, size=tuple(size), mode="bicubic", align_corners=False
        )
        resized = resized.squeeze(0)  # Remove batch dimension

        if need_permute_back:
            resized = resized.permute(1, 2, 0)  # (C, H, W) back to (H, W, C)

        return resized

    @staticmethod
    def save_output(object):
        """Saves the output to a pickel"""
        # Create output directory if it doesn't exist
        os.makedirs("outputs", exist_ok=True)
        torch.save(object, "outputs/crop_maker_output.pkl")
        logger.info(f"crop maker output saved to crop_maker_output.pkl")
