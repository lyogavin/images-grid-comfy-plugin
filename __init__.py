from .src import (
    LatentCombineNode,
    ImagesGridByColumnsNode,
    ImagesGridByRowsNode,
    ImageCombineNode,
    GridAnnotationNode,
    MaskToBoundingBoxNode,
)


NODE_CLASS_MAPPINGS = {
    "LatentCombine": LatentCombineNode,
    "ImagesGridByColumns": ImagesGridByColumnsNode,
    "ImagesGridByRows": ImagesGridByRowsNode,
    "ImageCombine": ImageCombineNode,
    "GridAnnotation": GridAnnotationNode,
    "MaskToBoundingBox": MaskToBoundingBoxNode,
}
