#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to convert labelme annotations to COCO format for surgical tool detection with triplets
(Only outputs triplet ID and bounding box coordinates)
"""

import os
import json
import shutil
import argparse
from pathlib import Path
import numpy as np
from datetime import datetime

TOOLS = {
    "scissors": 0,
    "forceps": 1,
    "aspirator": 2, 
    "needle driver": 3,
    "grasper": 4,
    "clip applier": 5,
    "Endobag": 6
}

ACTIONS = {
    "retract": 0,
    "coagulate": 1,
    "cut": 2,
    "dissect": 3,
    "grasp": 4,
    "bag": 5,
    "suture": 6,
    "suck": 7,
    "clip": 8,
    "null": 9
}

TARGETS = {
    "bladder": 0,
    "catheter": 1,
    "seminal vesicle": 2,
    "prostate": 3,
    "fascias": 4,
    "gauze": 5,
    "Endobag": 6,
    "thread": 7,
    "fluid": 8,
    "null": 9
}

TRIPLETS = {
    1: ["scissors", "retract", "bladder"],
    2: ["scissors", "retract", "catheter"],
    3: ["scissors", "retract", "seminal vesicle"],
    4: ["scissors", "retract", "prostate"],
    5: ["scissors", "retract", "fascias"],
    6: ["scissors", "retract", "gauze"],
    7: ["scissors", "retract", "Endobag"],
    8: ["scissors", "coagulate", "bladder"],
    9: ["scissors", "coagulate", "seminal vesicle"],
    10: ["scissors", "coagulate", "prostate"],
    11: ["scissors", "coagulate", "fascias"],
    12: ["scissors", "cut", "bladder"],
    13: ["scissors", "cut", "catheter"],
    14: ["scissors", "cut", "seminal vesicle"],
    15: ["scissors", "cut", "prostate"],
    16: ["scissors", "cut", "fascias"],
    17: ["scissors", "cut", "thread"],
    18: ["scissors", "dissect", "bladder"],
    19: ["scissors", "dissect", "seminal vesicle"],
    20: ["scissors", "dissect", "prostate"],
    21: ["scissors", "dissect", "fascias"],
    22: ["scissors", "null", "null"],
    23: ["forceps", "retract", "bladder"],
    24: ["forceps", "retract", "catheter"],
    25: ["forceps", "retract", "seminal vesicle"],
    26: ["forceps", "retract", "prostate"],
    27: ["forceps", "retract", "fascias"],
    28: ["forceps", "retract", "Endobag"],
    29: ["forceps", "coagulate", "bladder"],
    30: ["forceps", "coagulate", "seminal vesicle"],
    31: ["forceps", "coagulate", "prostate"],
    32: ["forceps", "coagulate", "fascias"],
    33: ["forceps", "dissect", "seminal vesicle"],
    34: ["forceps", "dissect", "prostate"],
    35: ["forceps", "dissect", "fascias"],
    36: ["forceps", "grasp", "catheter"],
    37: ["forceps", "grasp", "seminal vesicle"],
    38: ["forceps", "grasp", "prostate"],
    39: ["forceps", "grasp", "fascias"],
    40: ["forceps", "grasp", "gauze"],
    41: ["forceps", "grasp", "Endobag"],
    42: ["forceps", "grasp", "thread"],
    43: ["forceps", "suture", "bladder"],
    44: ["forceps", "suture", "prostate"],
    45: ["forceps", "suture", "fascias"],
    46: ["forceps", "null", "null"],
    47: ["aspirator", "retract", "bladder"],
    48: ["aspirator", "retract", "seminal vesicle"],
    49: ["aspirator", "retract", "prostate"],
    50: ["aspirator", "retract", "fascias"],
    51: ["aspirator", "retract", "Endobag"],
    52: ["aspirator", "suck", "fluid"],
    53: ["aspirator", "null", "null"],
    54: ["needle driver", "retract", "bladder"],
    55: ["needle driver", "retract", "prostate"],
    56: ["needle driver", "retract", "fascias"],
    57: ["needle driver", "grasp", "bladder"],
    58: ["needle driver", "grasp", "catheter"],
    59: ["needle driver", "grasp", "prostate"],
    60: ["needle driver", "grasp", "fascias"],
    61: ["needle driver", "grasp", "gauze"],
    62: ["needle driver", "grasp", "Endobag"],
    63: ["needle driver", "grasp", "thread"],
    64: ["needle driver", "suture", "bladder"],
    65: ["needle driver", "suture", "prostate"],
    66: ["needle driver", "suture", "fascias"],
    67: ["needle driver", "null", "null"],
    68: ["grasper", "retract", "bladder"],
    69: ["grasper", "retract", "catheter"],
    70: ["grasper", "retract", "seminal vesicle"],
    71: ["grasper", "retract", "prostate"],
    72: ["grasper", "retract", "fascias"],
    73: ["grasper", "grasp", "catheter"],
    74: ["grasper", "grasp", "seminal vesicle"],
    75: ["grasper", "grasp", "prostate"],
    76: ["grasper", "grasp", "fascias"],
    77: ["grasper", "grasp", "gauze"],
    78: ["grasper", "grasp", "Endobag"],
    79: ["grasper", "grasp", "thread"],
    80: ["grasper", "null", "null"],
    81: ["clip applier", "clip", "bladder"],
    82: ["clip applier", "clip", "seminal vesicle"],
    83: ["clip applier", "clip", "prostate"],
    84: ["clip applier", "clip", "fascias"],
    85: ["clip applier", "clip", "Endobag"],
    86: ["clip applier", "null", "null"],
    87: ["Endobag", "bag", "prostate"],
    88: ["Endobag", "bag", "fascias"],
    89: ["Endobag", "null", "null"]
}

TRIPLET_MAP = {}
for triplet_id, triplet in TRIPLETS.items():
    TRIPLET_MAP[tuple(triplet)] = triplet_id

DATASET_SPLIT1 = {
     'test': ['esadv1', 'psiv1', 'psiv4', 'pwhv8'],
     'val':  ['psiv7', 'pwhv5'],
     'train':['esadv2', 'esadv3', 'esadv4', 'psiv2', 'psiv3', 'psiv14', 'psiv15', 'psiv21', 
               'pwhv1', 'pwhv2', 'pwhv3', 'pwhv4', 'pwhv6', 'pwhv7', 'pwhv9']
 }
 
DATASET_SPLIT2 = {
     'test': ['esadv2', 'psiv7', 'pwhv4', 'pwhv9'],
     'val':  ['psiv1', 'pwhv6'],
     'train':['esadv1', 'esadv3', 'esadv4', 'psiv2', 'psiv3', 'psiv4', 'psiv14', 'psiv15', 'psiv21',
               'pwhv1', 'pwhv2', 'pwhv3', 'pwhv5', 'pwhv7', 'pwhv8']
 }
 
DATASET_SPLIT3 = {
     'test': ['esadv3', 'psiv14', 'pwhv1', 'psiv2'],
     'val': ['esadv1', 'pwhv2'],
     'train': ['esadv2', 'esadv4', 'psiv1', 'psiv3', 'psiv4', 'psiv7', 'psiv15', 'psiv21',
               'pwhv3', 'pwhv4', 'pwhv5', 'pwhv6', 'pwhv7', 'pwhv8', 'pwhv9']
 }

DATASET_SPLIT4 = {
     'test': ['esadv4', 'psiv15', 'pwhv2', 'pwhv7'],
     'val': ['esadv3', 'pwhv3'],
     'train': ['esadv1', 'esadv2', 'psiv1', 'psiv2', 'psiv3', 'psiv4', 'psiv7', 'psiv14', 'psiv21',
              'pwhv1', 'pwhv4', 'pwhv5', 'pwhv6', 'pwhv8', 'pwhv9']
}

DATASET_SPLIT5 = {
     'test': ['psiv3', 'psiv21', 'pwhv3', 'pwhv5', 'pwhv6'],
     'val': ['psiv14', 'pwhv1'],
     'train': ['esadv1', 'esadv2', 'esadv3', 'esadv4', 'psiv1', 'psiv2', 'psiv4', 'psiv7', 'psiv15',
               'pwhv2', 'pwhv4', 'pwhv7', 'pwhv8', 'pwhv9']
}

def parse_args():
    parser = argparse.ArgumentParser(description="Convert labelme annotations to COCO format with triplets (only outputs triplet ID)")
    parser.add_argument('--input-dir', type=str, default='/ssd/prostate/prostate_track_v2/prostate_v2_640',
                      help='Directory containing video folders with labelme annotations')
    parser.add_argument('--output-dir', type=str, default='/ssd/prostate/prostate_track_v2/dataset_coco_triplet',
                      help='Output directory for COCO format dataset')
    return parser.parse_args()

def convert_bbox_to_coco(bbox, img_width, img_height):
    """
    Convert bounding box to COCO format (x, y, width, height)
    
    Args:
        bbox: List of two points [[x1,y1], [x2,y2]]
        img_width: Image width
        img_height: Image height
        
    Returns:
        List of [x, y, width, height] in absolute coordinates
    """
    x1, y1 = min(bbox[0][0], bbox[1][0]), min(bbox[0][1], bbox[1][1])
    x2, y2 = max(bbox[0][0], bbox[1][0]), max(bbox[0][1], bbox[1][1])
    
    x = x1
    y = y1
    width = x2 - x1
    height = y2 - y1
    
    return [x, y, width, height]

def create_coco_info():
    return {
        "description": "Surgical Tool Detection Dataset with Triplets (Tool-Action-Target)",
        "url": "",
        "version": "2.0",
        "year": datetime.now().year,
        "contributor": "Prostate Surgery Analysis",
        "date_created": datetime.now().isoformat()
    }

def create_coco_categories():
    """
    Create COCO categories from triplet definitions
    """
    categories = []
    for triplet_id, triplet in TRIPLETS.items():
        tool, action, target = triplet
        categories.append({
            "id": triplet_id,
            "name": f"{tool}_{action}_{target}",
            "supercategory": "surgical_tool"
        })
    return categories

def process_json_file_for_coco(json_path, video_name, image_id_counter, annotation_id_counter, images_list, annotations_list, output_image_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)

    img_width = data['imageWidth']
    img_height = data['imageHeight']

    base_name = os.path.splitext(os.path.basename(json_path))[0]
    image_filename = f"{video_name}_{base_name}.jpg"
    output_img_path = os.path.join(output_image_dir, image_filename)

    input_img_path = os.path.join(os.path.dirname(json_path), f"{base_name}.jpg")
    if os.path.exists(input_img_path):
        shutil.copy(input_img_path, output_img_path)
    else:
        print(f"Warning: Image file not found: {input_img_path}")
        return image_id_counter, annotation_id_counter

    image_info = {
        "id": image_id_counter,
        "width": img_width,
        "height": img_height,
        "file_name": image_filename
    }
    images_list.append(image_info)

    for shape in data['shapes']:
        tool = shape['label']
        if tool not in TOOLS:
            continue

        attributes = shape.get('attributes', {})
        action = attributes.get('Action', 'null')
        target = attributes.get('Target', 'null')

        triplet = (tool, action, target)
        if triplet not in TRIPLET_MAP:
            raise ValueError(f"Cannot find triplet mapping for {triplet} in image {json_path}. "
                           f"Please check if the triplet combination is valid.")
        triplet_id = TRIPLET_MAP[triplet]

        bbox = shape['points']
        coco_bbox = convert_bbox_to_coco(bbox, img_width, img_height)

        area = coco_bbox[2] * coco_bbox[3]

        annotation = {
            "id": annotation_id_counter,
            "image_id": image_id_counter,
            "category_id": triplet_id,
            "bbox": coco_bbox,
            "area": area,
            "iscrowd": 0
        }
        annotations_list.append(annotation)
        annotation_id_counter += 1

    return image_id_counter + 1, annotation_id_counter

def create_coco_dataset(input_dir, output_dir, dataset_split, split_name):
    print(f"Creating COCO dataset for {split_name}...")

    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    for split_type in ['train', 'val', 'test']:
        print(f"  Processing {split_type} split...")

        coco_data = {
            "info": create_coco_info(),
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": create_coco_categories()
        }

        image_id_counter = 1
        annotation_id_counter = 1

        videos_in_split = dataset_split.get(split_type, [])

        for video_dir in os.listdir(input_dir):
            video_path = os.path.join(input_dir, video_dir)

            if not os.path.isdir(video_path):
                continue

            if video_dir not in videos_in_split:
                continue

            print(f"    Processing video: {video_dir}")

            output_image_dir = os.path.join(output_dir, split_type)

            for filename in os.listdir(video_path):
                if filename.endswith('.json'):
                    json_path = os.path.join(video_path, filename)

                    image_id_counter, annotation_id_counter = process_json_file_for_coco(
                        json_path, video_dir, image_id_counter, annotation_id_counter,
                        coco_data["images"], coco_data["annotations"], output_image_dir
                    )

        annotation_file = os.path.join(output_dir, f"{split_type}_annotations.json")
        with open(annotation_file, 'w') as f:
            json.dump(coco_data, f, indent=2)

        print(f"    {split_type}: {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")

def main():
    args = parse_args()

    input_dir = args.input_dir
    base_output_dir = args.output_dir

    dataset_splits = {
        'split1': DATASET_SPLIT1,
        'split2': DATASET_SPLIT2,
        'split3': DATASET_SPLIT3,
        'split4': DATASET_SPLIT4,
        'split5': DATASET_SPLIT5
    }

    for split_name, dataset_split in dataset_splits.items():
        print(f"\nProcessing {split_name}...")

        output_dir = os.path.join(base_output_dir, split_name)

        create_coco_dataset(input_dir, output_dir, dataset_split, split_name)

        print(f"{split_name} conversion completed. Dataset saved to {output_dir}")

    print(f"\nAll COCO dataset conversions completed! Generated {len(dataset_splits)} dataset splits in total.")

if __name__ == "__main__":
    main()
