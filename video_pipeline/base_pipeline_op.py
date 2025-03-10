# video_decoder/base_video_decoder.py
from abc import ABC, abstractmethod


class BasePipelineOp(ABC):
    def __init__(self, config, next_processor=None):
        self.config = config
        self.next_processor = next_processor  # Links to the next step
        self.is_save_output = config["save_output"]
        self.name = self.__class__.__name__

    @abstractmethod
    def process(self, frames):
        """Processes frames and passes them to the next processor."""
        pass
    
    @abstractmethod
    def save_output(self, object):
        """Saves the output to a pickel."""
        pass