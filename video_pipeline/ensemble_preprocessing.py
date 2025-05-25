# This should run only on CPU
import torch
import torchcodec
from logger import logger

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

try:
    import nvtx
except ImportError:
    print("nvtx not found, skipping nvtx annotations")


class EnsemblePreprocessing():
    def __init__(self,
                 input_video_path: str = None,
                 device: str = 'cpu',
                 do_video_decoding: bool = False,
                 do_preprocessing: bool = False,
                 num_video_decoder_threads: int = 1,
                 ):
        # Video decoder
        self.do_video_decoding = do_video_decoding
        self.do_preprocessing = do_preprocessing
        self.input_video_path = input_video_path
        self.device = device
        self.num_video_decoder_threads = num_video_decoder_threads
        self.num_output_frames = 10

        # preprocessing
        self.resize_global = [336, 602]
        self.resize_refine = [476, 840]
        self.crop_best_grid = [1, 2]
        self.patch_size = 14
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]

        self.batch_size = 30
        self.num_patches_in_global_slice = 1032
        self.num_patches_in_crop_slice = 1020

    # setter for the input video path
    def set_input_video_path(self, input_video_path: str):
        self.input_video_path = input_video_path
        logger.info(f"Input video path set to: {self.input_video_path}")

    @nvtx.annotate("process", color="blue")
    @torch.no_grad()
    def process(self, in_decoded_frames=None):
        decoded_frames = self.video_decoder() if self.do_video_decoding else in_decoded_frames
        assert decoded_frames is not None, "Decoded frames should not be None"
        # Preprocessing
        all_pixel_values = self.preprocessing(
            decoded_frames) if self.do_preprocessing else decoded_frames
        print(f"all_pixel_values shape: {all_pixel_values.shape}")
        return all_pixel_values

    @nvtx.annotate("processing", color="blue")
    def preprocessing(self, _frames):
        _frames = [frame.to(self.device) for frame in _frames]
        slice_images = []
        new_images = []
        for frame in _frames:
            # 1. First patch - resized image
            slice_images.append(
                self.resize_torch_image(frame, self.resize_global)
            )
            # 2. Get two patches from the image
            patches: list = self._get_patches(frame)
            if len(patches) > 0:
                for i in range(len(patches)):
                    slice_images.append(patches[i])

        # 3. convert the image from [0, 255] → [0, 1].
        slice_images = [img.float() / 255.0 for img in slice_images]
        # 4. (H, W, C) → (C, H, W)
        slice_images = [img.permute(2, 0, 1) for img in slice_images]
        # 5. Normalize the image
        slice_images = [
            normalize(
                img,
                mean=self.mean,
                std=self.std,
            )  # Normalize accepts tensors of shape (C, H, W)
            for img in slice_images
        ]

        # 6. Reshape the image by patch
        for slice_image in slice_images:
            new_images.append(self.reshape_by_patch(slice_image))

        pixel_values = new_images
        max_patches = self.num_patches_in_global_slice if self.num_patches_in_global_slice > self.num_patches_in_crop_slice else self.num_patches_in_crop_slice
        B = self.batch_size
        L = max_patches * self.patch_size

        # 7
        all_pixel_values_lst = [
            i.flatten(end_dim=1).permute(1, 0) for i in pixel_values
        ]

        # 8
        all_pixel_values = torch.nn.utils.rnn.pad_sequence(
            all_pixel_values_lst, batch_first=True, padding_value=0.0)

        # 9
        all_pixel_values = all_pixel_values.permute(
            0, 2, 1).reshape(B, 3, -1, L)

        return all_pixel_values

    @nvtx.annotate("video_decoder", color="blue")
    def video_decoder(self):
        frames = []
        frame = None
        decoder = torchcodec.decoders._core.create_from_file(
            self.input_video_path)
        num_threads = None
        if self.device == "cuda":
            num_threads = 1
        width = None
        height = None
        torchcodec.decoders._core._add_video_stream(
            decoder,
            stream_index=-1,
            device=self.device,
            num_threads=self.num_video_decoder_threads,
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

        if len(frames) >= self.num_output_frames:
            frame_idx = uniform_sample(len(frames), self.num_output_frames)

        return torch.stack([frames[i] for i in frame_idx]).permute(0, 2, 3, 1)

    @nvtx.annotate("_get_patches", color="blue")
    def _get_patches(self, frame: torch.Tensor) -> torch.Tensor:
        """Get patches from the image"""
        # 1. Resize the image to refined size
        refined_frame = self.resize_torch_image(frame, self.resize_refine)
        # 2. Slice the image into patches
        return self.split_into_patches(refined_frame, self.crop_best_grid)

    @nvtx.annotate("split_into_patches", color="blue")
    def split_into_patches(self, image, grid):
        """Vectorized version of splitting an image into patches"""
        height, width, C = image.shape
        patch_height = height // grid[0]
        patch_width = width // grid[1]

        # Pre-allocate all patches at once
        patches = []

        # Use torch.chunk for more efficient slicing
        h_chunks = torch.chunk(image, grid[0], dim=0)
        for h_chunk in h_chunks:
            w_chunks = torch.chunk(h_chunk, grid[1], dim=1)
            patches.extend(w_chunks)

        return patches

    @nvtx.annotate("resize_torch_image", color="blue")
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

        # # Convert to float32 for interpolation if necessary
        if frame.dtype == torch.uint8:
            frame = frame.float()

        # Perform resizing using bicubic interpolation
        resized = F.interpolate(
            frame, size=tuple(size), mode="bicubic", align_corners=False
        )
        resized = resized.clamp(0, 255).to(torch.uint8)
        resized = resized.squeeze(0)  # Remove batch dimension

        if need_permute_back:
            resized = resized.permute(1, 2, 0)  # (C, H, W) back to (H, W, C)

        return resized

    @nvtx.annotate("reshape_by_patch", color="blue")
    def reshape_by_patch(self, image):
        """
        :param image: shape [3, H, W]
        :param patch_size:
        :return: [3, patch_size, HW/patch_size]
        """
        patch_size = self.patch_size
        patches = F.unfold(
            image, (patch_size, patch_size), stride=(patch_size, patch_size)
        )

        patches = patches.reshape(image.size(0), patch_size, patch_size, -1)
        patches = patches.permute(0, 1, 3, 2).reshape(
            image.size(0), patch_size, -1)
        return patches
