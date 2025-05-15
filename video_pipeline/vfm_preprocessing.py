# video_decoder/vfm_preprocessing.py
from logger import logger
from .base_pipeline_op import BasePipelineOp
import os
import torch
# import nvtx
class VFMPreprocessing(BasePipelineOp):
    def __init__(self, config, next_processor=None):
        super().__init__(config, next_processor)
        self.device = config["device"]
        self.dtype = config["dtype"]
        self.batch_size = config["batch_size"] # Default: 30 = 10frames*3patches
        self.patch_size = config["patch_size"] # Default: 14
        self.num_patches_in_global_slice = config["num_patches_in_global_slice"] # Default: 1032 = (336/14)*(602/14)
        self.num_patches_in_crop_slice = config["num_patches_in_crop_slice"] # Default: 1020 = (476/14)*(840/2/14)  

    # @nvtx.annotate("process [VFMPreprocessing]", color="blue")
    def process(self, data: dict[str, torch.Tensor]):
        # Based on https://huggingface.co/openbmb/MiniCPM-V-2_6/blob/main/modeling_minicpmv.py#L78
        pixel_values = data["pixel_values"]
        tgt_sizes = data["tgt_sizes"]

        device = self.device
        dtype = self.dtype
        max_patches = self.num_patches_in_global_slice if self.num_patches_in_global_slice > self.num_patches_in_crop_slice else self.num_patches_in_crop_slice
        B = self.batch_size
        L = max_patches * self.patch_size 
        
        # 1
        all_pixel_values_lst = [
            i.flatten(end_dim=1).permute(1, 0) for i in pixel_values
        ]

        # 2 
        all_pixel_values = torch.stack(all_pixel_values_lst)  # [30, HW/patch_size + padding (for smaller patches), 3 * patch_size]
        
        # 3
        all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)
        
        # 4
        patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool, device=device)
        for i in range(B):
            patch_attn_mask[i, 0, :tgt_sizes[i][0] * tgt_sizes[i][1]] = True
            
            
        # Save the output
        self.save_output(all_pixel_values, "all_pixel_values") if self.is_save_output else None  # Save output
        self.save_output(patch_attn_mask, "patch_attn_mask") if self.is_save_output else None  # Save output
        self.save_output(tgt_sizes, "tgt_sizes") if self.is_save_output else None  # Save output
        returned_dict = {
            "all_pixel_values": all_pixel_values.type(getattr(torch, dtype)),
            "patch_attn_mask": patch_attn_mask,
            "tgt_sizes": tgt_sizes,
        }
        return self.next_processor.process(returned_dict) if self.next_processor else returned_dict
    
    @staticmethod
    def save_output(object, name):
        """Saves the output to a pickel"""
        # Create output directory if it doesn't exist
        os.makedirs("outputs", exist_ok=True)
        torch.save(object, f"outputs/{name}.pkl")