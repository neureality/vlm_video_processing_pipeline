import os
import torch
from torchcodec.decoders import VideoDecoder
import torchcodec
from logger import logger
import numpy as np
import torch.nn.functional as F


class MiniCPMPreprocessingEnsemblePipeline:
    def __init__(self, video_path, device="cuda"):
        self.device = device
        self.video_path = video_path
        self.num_output_frames = 10
        self.resize_global = ( 336, 602)
        self.resize_refine = ( 476, 840)
        self.crop_best_grid = (1, 2)
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        self.patch_size = 14
        self.batch_size = 30
        self.num_patches_in_global_slice = 1032
        self.num_patches_in_crop_slice = 1020
        

    def forward(self,):
        # 1. Video decoding
        decoded_frames = self.decode_and_sample_frames() # [N, H, W, 3] 

        # 2. Preprocessing
        
        # 2.1 Gettin image sizes
        # image_sizes = [(frame.shape[0], frame.shape[1]) for frame in decoded_frames]
        slice_images = []
        new_images = []
        tgt_sizes = []


        for frame in decoded_frames:
            # 1. First patch - global resized image
            slice_images.append(self.resize_image(frame, self.resize_global)) # [N, 3, 336, 602]
            # 2. Get two patches from the image
            patches = self.get_patches(frame) # [N, 4, 476, 840]
            slice_images.append(patches[0])
            slice_images.append(patches[1])
      
        # 3. Normalize
        # FIXME: Can our normalization kernel return float32 for uint8 input?
        #        OR is it more optimal to cast it to float32 first?
        slice_images = [ops.vision.normalize(
                image,
                mean=self.mean,
                std=self.std,
                dst_dtype="uint8",
            ) for image in slice_images]

        # 4. Reshape the image by patch
        for slice_image in slice_images:
            new_images.append(self.reshape_by_patch(slice_image))
            tgt_sizes.append(
                (
                    slice_image.shape[1] // self.patch_size,
                    slice_image.shape[2] // self.patch_size,
                )
            )
        
        # FIXME: What to do with that? Should we used precompiled ones?
        tgt_sizes = torch.tensor(tgt_sizes, device=self.device)

        # 5. convert the image from [0, 255] → [0, 1].
        new_images = [img.float() / 255.0 for img in new_images] # cast then div ------------------------------------------


        # VFM Preprocessing
        pixel_values = new_images
        max_patches = self.num_patches_in_global_slice
        B = self.batch_size
        L = max_patches * self.patch_size 
        
        # 1 This expect (N, 3, patch_size, HW/patch_size) -> [3 * patch_size, HW/patch_size] -> [HW/patch_size, 3 * patch_size]
        #   WARNING: THERE ARE TWO DIFFERENT SIZES, BECAUSE OF TWO DIFFERENT HW PAIRS
        # FIXME: DSP transpose expects batch_size as first dimension
        all_pixel_values_lst = [
            i.flatten(end_dim=2).permute(1, 0) for i in pixel_values # reshape then transpose ------------------------------------------
        ]

        # 2 
        # FIXME: This is a new kernel (but it does not have to be)
        # -> [30, HW/patch_size + padding (for smaller patches), 3 * patch_size]
        all_pixel_values = torch.nn.utils.rnn.pad_sequence(
            all_pixel_values_lst, batch_first=True, padding_value=0.0)
        
        # 3
        all_pixel_values = all_pixel_values.permute(0, 2, 1) # transpose [30, 3 * patch_size, HW/patch_size]
        all_pixel_values = all_pixel_values.reshape(B, 3, -1, L) # reshape [30, 3, 14, 1032 * 14]
        
        # 4
        # FIXME: We should probably load this from somewhere
        patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool, device=device)
        for i in range(B):
            patch_attn_mask[i, 0, :tgt_sizes[i][0] * tgt_sizes[i][1]] = True
            
        return all_pixel_values, patch_attn_mask

    def reshape_by_patch(self, image):
        """
        :param image: shape [N, 3, H, W]
        :param patch_size:
        :return: [N, 3, patch_size, HW/patch_size]
        """
        patch_size = self.patch_size
        patches = F.unfold(
            image, (patch_size, patch_size), stride=(patch_size, patch_size)
        )

        # FIXME: reshape is a new kernel, must work on NCHW and return [N, 3, patch_size, HW/patch_size]
        patches = patches.reshape(image.size(0), patch_size, patch_size, -1)
        patches = patches.permute(0, 1, 3, 2)  # transpose ----------------------------------------------------------
        patches = patches.reshape(image.size(0), patch_size, -1)
        return patches

    def get_patches(self, frame):
        """Get patches from the image"""
        # 1. Resize the image to refined size
        refined_frame = self.resize_image(frame, self.resize_refine)
        # 2. Slice the image into patches
        return self.split_into_patches(refined_frame, self.crop_best_grid)

    def split_into_patches(self, image, grid):
        """Vectorized version of splitting an image into patches"""
        height, width, C = image.shape
        
        patch_height = height // grid[0]
        patch_width = width // grid[1]
        
        # Pre-allocate all patches at once
        patches = []
        
        # Use torch.chunk for more efficient slicing
        h_chunks = torch.chunk(image, grid[0], dim=0) # STRIDED_SLICE ---------------------------------------------------------
        for h_chunk in h_chunks:
            w_chunks = torch.chunk(h_chunk, grid[1], dim=1) # STRIDED_SLICE ---------------------------------------------------------
            patches.extend(w_chunks)
            
        return patches

    def resize_image(self, frame: np.ndarray, size) -> np.ndarray:
        """Resize the image
        
        Assumptions:
            1. Assuming N,C,H,W format for each frame

        Returns:
            Resized frames in NCHW format
        """
        if frame.dtype == torch.uint8:
            frame = frame.float()
            
        # Perform resizing using bicubic interpolation
        resized = F.interpolate(
            frame, size=tuple(size), mode="bicubic", align_corners=False
        )
        resized = resized.clamp(0, 255).to(torch.uint8)

        return resized
        
    
    def decode_and_sample_frames(self):
        """Lean version of the video decoder."""
        frames = []
        frame = None
        decoder = torchcodec.decoders._core.create_from_file(self.video_path)
        num_threads = None
        if self.device == "cuda":
            num_threads = 1
        width = None
        height = None
        torchcodec.decoders._core._add_video_stream(
            decoder,
            stream_index=-1,
            device=self.device,
            num_threads=num_threads,
            width=width,
            height=height,
        )
        frame_count = 0
        fps = round(torchcodec.decoders._core.get_container_metadata(
            decoder=decoder).streams[0].average_fps)
        while True:
            try:
                frame, *_ = torchcodec.decoders._core.get_next_frame(decoder)
                frames.append(frame) if frame_count % fps == 0 else None
                frame_count += 1
            except Exception as e:
                break

        def uniform_sample(l, n):
            gap = l / n
            idxs = [int(i * gap + gap / 2) for i in range(n)]
            return idxs

        if len(frames) > self.num_output_frames:
            frame_idx = uniform_sample(len(frames), self.num_output_frames)
        else:
            frame_idx = list(range(len(frames)))

        return torch.stack([frames[i] for i in frame_idx]).permute(0, 2, 3, 1) # [N, 3, H, W] To meet the NR decoder
    
    
if __name__ == "__main__":
    video_path = "/home/odedh/nr_value_prop/videos/SampleVideo_1280x720_300frames.mp4"
    pipeline = MiniCPMPreprocessingEnsemblePipeline(video_path)
    all_pixel_values, patch_attn_mask = pipeline.forward()
    print(all_pixel_values.shape)
    print(patch_attn_mask.shape)