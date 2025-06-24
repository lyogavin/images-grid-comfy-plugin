import typing as t
import json
import torch
import numpy as np

from ..base import BaseNode


class MaskToBoundingBoxNode(BaseNode):
    RETURN_TYPES: tuple[str, ...] = ("STRING",)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, t.Any]:
        return {
            "required": {
                "mask": ("MASK",),
            },
        }

    def execute(self, mask: torch.Tensor) -> tuple[str,]:
        """
        Convert mask(s) to bounding box coordinates.
        
        Args:
            mask: Tensor of shape (batch_size, height, width) or (height, width)
            
        Returns:
            JSON string containing list of bounding boxes in format:
            [{"x": int, "y": int, "width": int, "height": int}, ...]
        """
        # Ensure mask is at least 3D (batch_size, height, width)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        elif mask.dim() == 4:
            # If mask has channel dimension, remove it
            mask = mask.squeeze(-1)
        
        bounding_boxes = []
        
        for i in range(mask.shape[0]):
            current_mask = mask[i]
            
            # Convert to numpy for easier processing
            mask_np = current_mask.cpu().numpy()
            
            # Find non-zero pixels
            nonzero_coords = np.nonzero(mask_np)
            
            if len(nonzero_coords[0]) == 0:
                # Empty mask - add empty bounding box
                bounding_boxes.append({
                    "x": 0,
                    "y": 0,
                    "width": 0,
                    "height": 0
                })
            else:
                # Get min/max coordinates
                y_coords, x_coords = nonzero_coords
                
                min_x = int(np.min(x_coords))
                max_x = int(np.max(x_coords))
                min_y = int(np.min(y_coords))
                max_y = int(np.max(y_coords))
                
                # Calculate bounding box
                width = max_x - min_x + 1
                height = max_y - min_y + 1
                
                bounding_boxes.append({
                    "x": min_x,
                    "y": min_y,
                    "width": width,
                    "height": height
                })
        
        # Convert to JSON string
        json_result = json.dumps(bounding_boxes, indent=2)
        
        return (json_result,) 