# ------------------------------------------------------------------------
# Triplet Dataset for Deformable DETR
# Based on official COCO dataset implementation
# ------------------------------------------------------------------------

from pathlib import Path
from .coco import CocoDetection, make_coco_transforms
from util.misc import get_local_rank, get_local_size


def build(image_set, args):
    """Build triplet dataset in COCO format"""
    root = Path(args.coco_path)
    assert root.exists(), f'provided triplet dataset path {root} does not exist'

    # Triplet dataset structure
    PATHS = {
        "train": (root / "train", root / "train_annotations.json"),
        "val": (root / "val", root / "val_annotations.json"),
        "test": (root / "test", root / "test_annotations.json"),
    }

    img_folder, ann_file = PATHS[image_set]
    assert img_folder.exists(), f'Image folder {img_folder} does not exist'
    assert ann_file.exists(), f'Annotation file {ann_file} does not exist'

    dataset = CocoDetection(
        img_folder, ann_file,
        transforms=make_coco_transforms(image_set),
        return_masks=args.masks,
        cache_mode=args.cache_mode,
        local_rank=get_local_rank(),
        local_size=get_local_size()
    )

    return dataset
