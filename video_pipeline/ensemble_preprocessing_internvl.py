import os
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
import torchcodec

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
        self.resize_global = (448, 448)          # (H,W)
        self.resize_refine = (448, 896)          # (H,W)
        self.grid_cols = 2                       # crop_best_grid = [1,2]
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        self.batch_size = 30                     # 10 global + 20 crops

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
            torchcodec.decoders._core.get_container_metadata(
                dec).streams[0].average_fps
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

        idxs = torch.linspace(0, len(frames) - 1,
                              steps=self.num_output_frames, dtype=torch.long)
        out = torch.stack([frames[i] for i in idxs]).permute(
            0, 2, 3, 1)  # (N,H,W,C)
        return out

    # --------------------------- preprocessing -------------------------------
    @nvtx.annotate("_preprocess_vectorised", color="blue")
    def _preprocess_vectorised(self, hwcu8: torch.Tensor) -> torch.Tensor:
        device = self.device
        mean = self.mean.to(device)[:, None, None]
        std = self.std.to(device)[:, None, None]

        # ---------- GLOBAL ---------------------------------------------------
        # Normalise the frames [10, 1080, 1920, 3]
        frames = normalize(hwcu8.permute(0, 3, 1, 2).float(
        ) / 255.0, mean=mean, std=std)  # [10, 3, 1080, 1920]
        global_rs = F.interpolate(
            frames, size=self.resize_global, mode="bicubic", align_corners=False)  # [10, 3, 448, 448]

        # ---------- CROPS -----------------------------------------------------
        refine = F.interpolate(
            frames, size=self.resize_refine, mode="bicubic", align_corners=False)  # [10, 3, 448, 896]
        crops = torch.chunk(refine, self.grid_cols, dim=3) # 2 X [10, 3, 448, 448]
        crops_batched = torch.cat(crops, dim=0) # (20,3,448,448)

        # ---------- CONCAT ----------------------------------------------------
        crops_grouped = crops_batched.view(
            10, 2, 3, 448, 448)  # [10, 2, 3, 448, 448]
        global_expanded = global_rs.unsqueeze(1)  # [10, 1, 3, 448, 448]
        pixel_values = torch.cat(
            [global_expanded, crops_grouped], dim=1).view(30, 3, 448, 448)

        if pixel_values.size(0) != self.batch_size:
            raise RuntimeError("Unexpected batch size after concatenation")

        return pixel_values


if __name__ == "__main__":
    # Example usage
    video_path = "videos/1080p/snowboard_1080p_12M_30fps_300frames.mp4"
    ensemble_preprocessing = EnsemblePreprocessing(input_video_path=video_path,
                                                   device="cpu",
                                                   do_video_decoding=True,
                                                   do_preprocessing=True
                                                   )
    pixel_values = ensemble_preprocessing.process()
    print(f"Processed frames shape: {pixel_values.shape}")

    torch.save(pixel_values, os.path.join(
        "/home/odedh/nr_value_prop", "_test_all_pixel_values.pt"))
