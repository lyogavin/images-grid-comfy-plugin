from .src import (
    LatentCombineNode,
    ImagesGridByColumnsNode,
    ImagesGridByRowsNode,
    ImageCombineNode,
    GridAnnotationNode,
    MaskToBoundingBoxNode,
    IndexSelectorNode,
)


NODE_CLASS_MAPPINGS = {
    "LatentCombine": LatentCombineNode,
    "ImagesGridByColumns": ImagesGridByColumnsNode,
    "ImagesGridByRows": ImagesGridByRowsNode,
    "ImageCombine": ImageCombineNode,
    "GridAnnotation": GridAnnotationNode,
    "MaskToBoundingBox": MaskToBoundingBoxNode,
    "IndexSelector": IndexSelectorNode,
}
