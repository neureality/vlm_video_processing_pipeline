# video_pipeline/resize_video.py
from logger import logger
from .base_pipeline_op import BasePipelineOp
import os
from typing import List

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize



class VideoPreprocessing(BasePipelineOp):
    def __init__(self, config, next_processor=None):
        super().__init__(config, next_processor)
        self.is_save_output = config["save_output"]

    def process(self, _frames: List[torch.Tensor]) -> List[torch.Tensor]:
        """Process the video frames
        
        Args:
            _frames (List[torch.Tensor]): List of video frames in (H, W, C) format
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the processed video frames
        """
        
        logger.info("Resizing and Cropping the images")

        image_sizes = [(image.shape[0], image.shape[1]) for image in _frames]
        slice_images = []
        new_images = []
        tgt_sizes = []
        for frame in _frames:
            # 1. First patch - resized image
            slice_images.append(
                self.resize_torch_image(frame, self.config["resize_global"])
            )
            # 2. Get two patches from the image
            patches: list = self._get_patches(frame)
            if len(patches) > 0:
                for i in range(len(patches)):
                    slice_images.append(patches[i])

        # 3. convert the image from [0, 255] → [0, 1].
        slice_images = [img.float() / 255.0 for img in slice_images]
        # 3. (H, W, C) → (C, H, W)
        slice_images = [img.permute(2, 0, 1) for img in slice_images]
        # 4. Normalize the image
        slice_images = [
            normalize(
                img, mean=self.config["mean"], std=self.config["std"]
            )  # Gets (C, H, W)
            for img in slice_images
        ]

        # 5. Reshape the image by patch
        for slice_image in slice_images:
            new_images.append(self.reshape_by_patch(slice_image))
            tgt_sizes.append(
                (
                    slice_image.shape[1] // self.config["patch_size"],
                    slice_image.shape[2] // self.config["patch_size"],
                )
            )
        tgt_sizes = torch.tensor(tgt_sizes)
        self.save_output(new_images, "pixel_values") if self.is_save_output else None  # Save output
        self.save_output(tgt_sizes, "tgt_sizes") if self.is_save_output else None  # Save output
        self.save_output(image_sizes, "image_sizes") if self.is_save_output else None  # Save output
        
        return (
            self.next_processor.process({"pixel_values": new_images, "tgt_sizes": tgt_sizes, "image_sizes": image_sizes})
            if self.next_processor
            else {"pixel_values": new_images, "tgt_sizes": tgt_sizes, "image_sizes": image_sizes}
        )

    def _get_patches(self, frame: torch.Tensor) -> torch.Tensor:
        """Get patches from the image"""
        # 1. Resize the image to refined size
        refined_frame = self.resize_torch_image(frame, self.config["resize_refine"])
        # 2. Slice the image into patches
        return self.split_into_patches(refined_frame, self.config["crop_best_grid"])

    def split_into_patches(self, image, grid):
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

    def resize_torch_image(
        self, frame: torch.Tensor, size: tuple | list
    ) -> torch.Tensor:
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

    def reshape_by_patch(self, image):
        """
        :param image: shape [3, H, W]
        :param patch_size:
        :return: [3, patch_size, HW/patch_size]
        """
        patch_size = self.config["patch_size"]
        patches = F.unfold(
            image, (patch_size, patch_size), stride=(patch_size, patch_size)
        )

        patches = patches.reshape(image.size(0), patch_size, patch_size, -1)
        patches = patches.permute(0, 1, 3, 2).reshape(image.size(0), patch_size, -1)
        return patches

    @staticmethod
    def save_output(object, name):
        """Saves the output to a pickel"""
        # Create output directory if it doesn't exist
        os.makedirs("outputs", exist_ok=True)
        torch.save(object, f"outputs/{name}.pkl")
