# video_decoder/decoding_and_sampling.py
import os
import torch
from torchcodec.decoders import VideoDecoder
import torchcodec
from logger import logger
from .base_pipeline_op import BasePipelineOp

# import nvtx


class DecodeAndSample(BasePipelineOp):
    def __init__(self, config, next_processor=None):
        super().__init__(config, next_processor)
        self.num_output_frames = config["num_output_frames"]
        self.input_path = config["input_path"]
        self.device = config["device"]
        self.dim_order = config["dimension_order"]
        self.input_resolution = config["input_resolution"]
        self.codec = config["codec"]

    def process(self, is_lean_process=True):
        """Decodes the video and samples frames
        Args:
            is_lean_process (bool): If True, use the lean process. Defaults to True.
            
        Returns:
            tensor_frames (torch.Tensor): Tensor of shape (num_frames, height, width, channels)
        """

        if is_lean_process:
            tensor_frames = self.lean_process()
        else:
            self.decoder = VideoDecoder(
                source=self.input_path,
                device=self.device,
                dimension_order=self.dim_order,
            )

            def uniform_sample(l, n):
                gap = len(l) / n
                idxs = [int(i * gap + gap / 2) for i in range(n)]
                return [l[i] for i in idxs]

            sample_fps = round(self.decoder.metadata.average_fps / 1)  # FPS
            frame_idx = [
                i for i in range(0, self.decoder.metadata.num_frames, sample_fps)
            ]

            if len(frame_idx) > self.num_output_frames:
                frame_idx = uniform_sample(frame_idx, self.num_output_frames)

            tensor_frames = self.decoder.get_frames_at(indices=frame_idx).data
        self.save_output(tensor_frames) if self.is_save_output else None  # Save output
        return (
            self.next_processor.process(tensor_frames)
            if self.next_processor
            else tensor_frames
        )

    def lean_process(self):
        """Lean version of the video decoder."""
        frames = []
        frame = None
        decoder = torchcodec.decoders._core.create_from_file(self.input_path)
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

        while True:
            try:
                frame, *_ = torchcodec.decoders._core.get_next_frame(decoder)
                frames.append(frame)
                frame_count += 1
            except Exception as e:
                break

        def uniform_sample(l, n):
            gap = l / n
            idxs = [int(i * gap + gap / 2) for i in range(n)]
            return idxs

        if frame_count > self.num_output_frames:
            frame_idx = uniform_sample(frame_count, self.num_output_frames)

        return torch.stack([frames[i] for i in frame_idx]).permute(0, 2, 3, 1)

    @staticmethod
    def save_output(object):
        """Saves the output to a pickel."""
        # Create output directory if it doesn't exist
        os.makedirs("outputs", exist_ok=True)
        torch.save(object, "outputs/decoder_output.pkl")
        logger.info(f"decoder output saved to decoder_output.pkl")
