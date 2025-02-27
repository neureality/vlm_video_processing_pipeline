# video_decoder/decoding_and_sampling.py
import os
import torch
from torchcodec.decoders import VideoDecoder
from logger import logger
from .base_pipeline_op import BasePipelineOp


class DecodeAndSample(BasePipelineOp):
    def __init__(self, config, next_processor=None):
        super().__init__(config, next_processor)
        self.num_output_frames = config["num_output_frames"]
        self.is_save_output = config["save_output"]
        self.decoder = VideoDecoder(
            source=config["input_path"],
            device=config["device"],
            dimension_order=config["dimension_order"],
        )
        assert (
            self.decoder.metadata.height == config["input_resolution"][0]
        ), "Height mismatch"
        assert (
            self.decoder.metadata.width == config["input_resolution"][1]
        ), "Width mismatch"
        assert self.decoder.metadata.codec == config["codec"], "Codec mismatch"
        logger.info(f"Decoding video from {config['input_path']}")
        logger.info(f"Video metadata: {self.decoder.metadata}")

    def process(self):
        """Decodes the video and samples frames"""

        def uniform_sample(l, n):
            gap = len(l) / n
            idxs = [int(i * gap + gap / 2) for i in range(n)]
            return [l[i] for i in idxs]

        sample_fps = round(self.decoder.metadata.average_fps / 1)  # FPS
        frame_idx = [i for i in range(0, self.decoder.metadata.num_frames, sample_fps)]

        if len(frame_idx) > self.num_output_frames:
            frame_idx = uniform_sample(frame_idx, self.num_output_frames)

        tensor_frames = self.decoder.get_frames_at(indices=frame_idx).data
        self.save_output(tensor_frames) if self.is_save_output else None  # Save output
        return (
            self.next_processor.process(tensor_frames)
            if self.next_processor
            else tensor_frames
        )

    @staticmethod
    def save_output(object):
        """Saves the output to a pickel."""
        # Create output directory if it doesn't exist
        os.makedirs('outputs', exist_ok=True)
        torch.save(object, "outputs/decoder_output.pkl")
        logger.info(f"decoder output saved to decoder_output.pkl")
