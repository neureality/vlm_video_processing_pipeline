# Vectorized rewrite of EnsemblePreprocessing
# Author: ChatGPT – generated on request of Oded (patch 2)
# -----------------------------------------------------------------------------
# Key fix: the previous version concatenated slice images of different spatial
# sizes (336×602 vs 476×420) before passing them through `torch.nn.functional.
# unfold`, which requires a uniform H×W across the batch.  The code below keeps
# **two separate groups** (global‑resized frames and crop patches), runs the
# patch‑extraction on each group independently, then concatenates the flattened
# outputs.  This preserves correctness while remaining fully vectorised.
# -----------------------------------------------------------------------------

import os
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
import torchcodec
from typing import List, Tuple

try:
    import nvtx
except ImportError:  # pragma: no cover
    class _NVTX:  # dummy replacement
        def annotate(self, *_, **__):
            def _wrap(func):
                return func
            return _wrap
    nvtx = _NVTX()


class EnsemblePreprocessing:
    """Vectorised implementation (CPU‑only) of the original pipeline."""

    # --------------------------- initialisation ------------------------------

    def __init__(
        self,
        input_video_path: str | None = None,
        *,
        device: str = "cpu",
        do_video_decoding: bool = False,
        do_preprocessing: bool = False,
        num_video_decoder_threads: int = 1,
    ) -> None:
        self.input_video_path = input_video_path
        self.device = device
        self.do_video_decoding = do_video_decoding
        self.do_preprocessing = do_preprocessing
        self.num_video_decoder_threads = num_video_decoder_threads

        # hyper‑params --------------------------------------------------------
        self.num_output_frames = 10
        self.resize_global = (336, 602)          # (H,W)
        self.resize_refine = (476, 840)          # (H,W)
        self.grid_cols = 2                       # crop_best_grid = [1,2]
        self.patch_size = 14
        self.mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        self.std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)

        # constants that define output tensor shape --------------------------
        self.batch_size = 30                     # 10 global + 20 crops
        self.patches_global = 1032               # 43×24
        self.patches_crop = 1020                 # 30×34

    # --------------------------- public API ----------------------------------

    def set_input_video_path(self, path: str):
        self.input_video_path = path
        if path != self._cached_path:
            self._cached_path = None
            self._cached_frames = None

    @nvtx.annotate("process", color="blue")
    @torch.no_grad()
    def process(self, in_decoded_frames: torch.Tensor | None = None) -> torch.Tensor:
        if self.do_video_decoding:
            frames = self._decode_video()
        else:
            frames = in_decoded_frames
        if frames is None:
            raise RuntimeError("Decoded frames required but missing")
        if not self.do_preprocessing:
            return frames
        return self._preprocess_vectorised(frames)

    # --------------------------- decoding ------------------------------------

    @nvtx.annotate("video_decoder", color="blue")
    def _decode_video(self) -> torch.Tensor:
        if self.input_video_path is None:
            raise ValueError("input_video_path not set")

        dec = torchcodec.decoders._core.create_from_file(self.input_video_path)
        torchcodec.decoders._core._add_video_stream(
            dec,
            stream_index=-1,
            device=self.device,
            num_threads=self.num_video_decoder_threads,
            width=None,
            height=None,
        )
        fps = round(
            torchcodec.decoders._core.get_container_metadata(dec).streams[0].average_fps
        )
        frames: list[torch.Tensor] = []
        frame_no = 0
        while True:
            try:
                fr, *_ = torchcodec.decoders._core.get_next_frame(dec)
                if frame_no % fps == 0:
                    frames.append(fr)
                frame_no += 1
            except Exception:
                break

        if len(frames) < self.num_output_frames:
            raise RuntimeError("Video shorter than expected")

        idxs = torch.linspace(0, len(frames) - 1, steps=self.num_output_frames, dtype=torch.long)
        out = torch.stack([frames[i] for i in idxs]).permute(0, 2, 3, 1)  # (N,H,W,C)
        return out

    # --------------------------- preprocessing -------------------------------

    @nvtx.annotate("_preprocess_vectorised", color="blue")
    def _preprocess_vectorised(self, hwcu8: torch.Tensor) -> torch.Tensor:
        """Vectorised, but keeps separate tensors for the two spatial sizes.

        The patch dimension (last) must be equal across the batch **before** we
        concatenate along the sample dimension.  We therefore pad *each* group
        (global‑resized frames and crop patches) to the common length `L` prior
        to concatenation.
        """

        device = self.device
        p = self.patch_size
        mean = self.mean.to(device)[:, None, None]
        std = self.std.to(device)[:, None, None]

        def _prep(img: torch.Tensor, L: int) -> torch.Tensor:  # (B,C,H,W) float32 0‑1
            img = (img - mean) / std
            unfolded = F.unfold(img, kernel_size=p, stride=p)  # (B, C*p*p, Npatch)
            unfolded = unfolded.view(img.size(0), 3, p, p, -1)
            unfolded = (
                unfolded.permute(0, 1, 2, 4, 3)  # (B,3,p,Npatch,p)
                .reshape(img.size(0), 3, p, -1)   # (B,3,p,Npatch)
                .flatten(start_dim=1, end_dim=2)  # (B, 3*p, Npatch)
            )
            if unfolded.shape[2] < L:
                unfolded = F.pad(unfolded, (0, L - unfolded.shape[2]))
            return unfolded  # (B,3*p,L)

        # target length -------------------------------------------------------
        L = max(self.patches_global, self.patches_crop) * p  # 14448

        # ---------- GLOBAL ---------------------------------------------------
        frames = hwcu8.to(device).permute(0, 3, 1, 2).float() / 255.0  # (N,C,H,W)
        global_rs = F.interpolate(frames, size=self.resize_global, mode="bicubic", align_corners=False)
        flat_global = _prep(global_rs, L)         # (N,3*p,L)

        # ---------- CROPS -----------------------------------------------------
        refine = F.interpolate(frames, size=self.resize_refine, mode="bicubic", align_corners=False)
        crops = torch.chunk(refine, self.grid_cols, dim=3)      # two (N,C,Hc,Wc)
        crops_batched = torch.cat(crops, dim=0)                 # (2N,C,Hc,Wc)
        flat_crops = _prep(crops_batched, L)                    # (2N,3*p,L)

        # ---------- CONCAT ----------------------------------------------------
        all_slices = torch.cat([flat_global, flat_crops], dim=0)  # (30,3*p,L)
        if all_slices.size(0) != self.batch_size:
            raise RuntimeError("Unexpected batch size after concatenation")

        out = all_slices.view(self.batch_size, 3, -1, L)  # (30,3,?,L)
        print(f"all_pixel_values shape: {out.shape}")
        torch.save(out, os.path.join("/home/odedh/nr_value_prop", "_test_all_pixel_values.pt"))
        return out
