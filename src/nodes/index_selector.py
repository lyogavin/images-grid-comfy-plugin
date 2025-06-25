import typing as t

import torch

from ..base import BaseNode

try:
    from comfy.utils import common_upscale
except ImportError:
    # Fallback if comfy utils are not available
    def common_upscale(image, width, height, upscale_method, crop):
        return torch.nn.functional.interpolate(image, size=(height, width), mode='bilinear', align_corners=False)


class IndexSelectorNode(BaseNode):
    RETURN_TYPES: tuple[str, ...] = ("IMAGE", "MASK")
    RETURN_NAMES: tuple[str, ...] = ("control_images", "masks")

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, t.Any]:
        return {
            "required": {
                "images": ("IMAGE",),
                "index_selector": ("STRING", {"default": "0,1,2", "multiline": False}),
                "empty_frame_level": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "inpaint_mask": ("MASK",),
            }
        }

    def execute(
        self,
        images: torch.Tensor,
        index_selector: str,
        empty_frame_level: float,
        inpaint_mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        # Parse the index selector string
        try:
            selected_indices = []
            if index_selector.strip():
                selected_indices = [int(x.strip()) for x in index_selector.split(',') if x.strip()]
        except ValueError:
            # If parsing fails, default to selecting all images
            selected_indices = list(range(images.shape[0]))
        
        # Get image dimensions
        num_input_images, H, W, C = images.shape
        device = images.device
        
        # Filter valid indices
        selected_indices = [i for i in selected_indices if 0 <= i < num_input_images]
        
        # Create control images
        control_images = torch.ones((num_input_images, H, W, C), device=device) * empty_frame_level
        
        # Place selected images in their positions
        for idx in selected_indices:
            control_images[idx] = images[idx]
        
        # Create masks (0 for selected images, 1 for empty frames)
        masks = torch.ones((num_input_images, H, W), device=device)
        for idx in selected_indices:
            masks[idx] = 0
        
        # Handle inpaint mask if provided
        if inpaint_mask is not None:
            # Ensure inpaint mask matches the dimensions
            if inpaint_mask.dim() == 2:
                inpaint_mask = inpaint_mask.unsqueeze(0)
            
            # Resize inpaint mask to match image dimensions if needed
            if inpaint_mask.shape[-2:] != (H, W):
                inpaint_mask = common_upscale(
                    inpaint_mask.unsqueeze(1), W, H, "nearest-exact", "disabled"
                ).squeeze(1).to(device)
            
            # Adjust inpaint mask to match number of frames
            if inpaint_mask.shape[0] > num_input_images:
                inpaint_mask = inpaint_mask[:num_input_images]
            elif inpaint_mask.shape[0] < num_input_images:
                # Repeat the mask to match the number of images
                repeat_count = num_input_images // inpaint_mask.shape[0] + 1
                inpaint_mask = inpaint_mask.repeat(repeat_count, 1, 1)[:num_input_images]
            
            # Combine with the generated masks
            masks = inpaint_mask * masks
        
        return (control_images.cpu().float(), masks.cpu().float()) 