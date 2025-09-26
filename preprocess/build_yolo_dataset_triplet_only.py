#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to convert labelme annotations to YOLO format for surgical tool detection with triplets
(Only outputs triplet ID and bounding box coordinates)
"""

import os
import json
import shutil
import argparse
from pathlib import Path
import numpy as np

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
    0: ["scissors", "retract", "bladder"],
    1: ["scissors", "retract", "catheter"],
    2: ["scissors", "retract", "seminal vesicle"],
    3: ["scissors", "retract", "prostate"],
    4: ["scissors", "retract", "fascias"],
    5: ["scissors", "retract", "gauze"],
    6: ["scissors", "retract", "Endobag"],
    7: ["scissors", "coagulate", "bladder"],
    8: ["scissors", "coagulate", "seminal vesicle"],
    9: ["scissors", "coagulate", "prostate"],
    10: ["scissors", "coagulate", "fascias"],
    11: ["scissors", "cut", "bladder"],
    12: ["scissors", "cut", "catheter"],
    13: ["scissors", "cut", "seminal vesicle"],
    14: ["scissors", "cut", "prostate"],
    15: ["scissors", "cut", "fascias"],
    16: ["scissors", "cut", "thread"],
    17: ["scissors", "dissect", "bladder"],
    18: ["scissors", "dissect", "seminal vesicle"],
    19: ["scissors", "dissect", "prostate"],
    20: ["scissors", "dissect", "fascias"],
    21: ["scissors", "null", "null"],
    22: ["forceps", "retract", "bladder"],
    23: ["forceps", "retract", "catheter"],
    24: ["forceps", "retract", "seminal vesicle"],
    25: ["forceps", "retract", "prostate"],
    26: ["forceps", "retract", "fascias"],
    27: ["forceps", "retract", "Endobag"],
    28: ["forceps", "coagulate", "bladder"],
    29: ["forceps", "coagulate", "seminal vesicle"],
    30: ["forceps", "coagulate", "prostate"],
    31: ["forceps", "coagulate", "fascias"],
    32: ["forceps", "dissect", "seminal vesicle"],
    33: ["forceps", "dissect", "prostate"],
    34: ["forceps", "dissect", "fascias"],
    35: ["forceps", "grasp", "catheter"],
    36: ["forceps", "grasp", "seminal vesicle"],
    37: ["forceps", "grasp", "prostate"],
    38: ["forceps", "grasp", "fascias"],
    39: ["forceps", "grasp", "gauze"],
    40: ["forceps", "grasp", "Endobag"],
    41: ["forceps", "grasp", "thread"],
    42: ["forceps", "suture", "bladder"],
    43: ["forceps", "suture", "prostate"],
    44: ["forceps", "suture", "fascias"],
    45: ["forceps", "null", "null"],
    46: ["aspirator", "retract", "bladder"],
    47: ["aspirator", "retract", "seminal vesicle"],
    48: ["aspirator", "retract", "prostate"],
    49: ["aspirator", "retract", "fascias"],
    50: ["aspirator", "retract", "Endobag"],
    51: ["aspirator", "suck", "fluid"],
    52: ["aspirator", "null", "null"],
    53: ["needle driver", "retract", "bladder"],
    54: ["needle driver", "retract", "prostate"],
    55: ["needle driver", "retract", "fascias"],
    56: ["needle driver", "grasp", "bladder"],
    57: ["needle driver", "grasp", "catheter"],
    58: ["needle driver", "grasp", "prostate"],
    59: ["needle driver", "grasp", "fascias"],
    60: ["needle driver", "grasp", "gauze"],
    61: ["needle driver", "grasp", "Endobag"],
    62: ["needle driver", "grasp", "thread"],
    63: ["needle driver", "suture", "bladder"],
    64: ["needle driver", "suture", "prostate"],
    65: ["needle driver", "suture", "fascias"],
    66: ["needle driver", "null", "null"],
    67: ["grasper", "retract", "bladder"],
    68: ["grasper", "retract", "catheter"],
    69: ["grasper", "retract", "seminal vesicle"],
    70: ["grasper", "retract", "prostate"],
    71: ["grasper", "retract", "fascias"],
    72: ["grasper", "grasp", "catheter"],
    73: ["grasper", "grasp", "seminal vesicle"],
    74: ["grasper", "grasp", "prostate"],
    75: ["grasper", "grasp", "fascias"],
    76: ["grasper", "grasp", "gauze"],
    77: ["grasper", "grasp", "Endobag"],
    78: ["grasper", "grasp", "thread"],
    79: ["grasper", "null", "null"],
    80: ["clip applier", "clip", "bladder"],
    81: ["clip applier", "clip", "seminal vesicle"],
    82: ["clip applier", "clip", "prostate"],
    83: ["clip applier", "clip", "fascias"],
    84: ["clip applier", "clip", "Endobag"],
    85: ["clip applier", "null", "null"],
    86: ["Endobag", "bag", "prostate"],
    87: ["Endobag", "bag", "fascias"],
    88: ["Endobag", "null", "null"]
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
    parser = argparse.ArgumentParser(description="Convert labelme annotations to YOLO format with triplets (only outputs triplet ID)")
    parser.add_argument('--input-dir', type=str, default='../prostate_v2_640',
                      help='Directory containing video folders with labelme annotations')
    parser.add_argument('--output-dir', type=str, default='../dataset_yolo_triplet',
                      help='Output directory for YOLO format dataset')
    return parser.parse_args()

def convert_bbox_to_yolo(bbox, img_width, img_height):
    """
    Convert bounding box to YOLO format (x_center, y_center, width, height)
    
    Args:
        bbox: List of two points [[x1,y1], [x2,y2]]
        img_width: Image width
        img_height: Image height
        
    Returns:
        List of normalized [x_center, y_center, width, height]
    """
    x1, y1 = min(bbox[0][0], bbox[1][0]), min(bbox[0][1], bbox[1][1])
    x2, y2 = max(bbox[0][0], bbox[1][0]), max(bbox[0][1], bbox[1][1])
    
    x_center = (x1 + x2) / 2.0 / img_width
    y_center = (y1 + y2) / 2.0 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    return [x_center, y_center, width, height]

def process_json_file(json_path, video_name, output_image_dir, output_label_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    img_width = data['imageWidth']
    img_height = data['imageHeight']

    base_name = os.path.splitext(os.path.basename(json_path))[0]
    output_img_path = os.path.join(output_image_dir, f"{video_name}_{base_name}.jpg")
    output_txt_path = os.path.join(output_label_dir, f"{video_name}_{base_name}.txt")

    input_img_path = os.path.join(os.path.dirname(json_path), f"{base_name}.jpg")
    if os.path.exists(input_img_path):
        shutil.copy(input_img_path, output_img_path)

    with open(output_txt_path, 'w') as f:
        for shape in data['shapes']:
            tool = shape['label']
            if tool not in TOOLS:
                continue

            attributes = shape.get('attributes', {})
            action = attributes.get('Action', 'null')
            target = attributes.get('Target', 'null')

            triplet = (tool, action, target)
            triplet_id = TRIPLET_MAP.get(triplet, -1)

            if triplet_id == -1:
                print(f"Warning: Cannot find triplet mapping {triplet}, image path: {json_path}")
                null_triplet = (tool, 'null', 'null')
                triplet_id = TRIPLET_MAP.get(null_triplet, 0)

            bbox = shape['points']
            yolo_bbox = convert_bbox_to_yolo(bbox, img_width, img_height)

            f.write(f"{triplet_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")

def create_dataset_yaml(output_dir):
    abs_output_dir = os.path.abspath(output_dir)

    yaml_content = f"""# YOLO dataset configuration with triplets (only triplet ID)
path: {abs_output_dir}  # dataset root directory
train: train/images  # train images
val: val/images  # validation images
test: test/images  # test images

# Classes
nc: {len(TRIPLETS)}  # number of triplet classes (0-88)
names:
"""
    
    for triplet_id, triplet in TRIPLETS.items():
        tool, action, target = triplet
        yaml_content += f"  {triplet_id}: {tool}_{action}_{target}\n"

    with open(os.path.join(output_dir, 'dataset_triplet.yaml'), 'w') as f:
        f.write(yaml_content)

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

        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)

        for video_dir in os.listdir(input_dir):
            video_path = os.path.join(input_dir, video_dir)

            if not os.path.isdir(video_path):
                continue

            assigned_splits = []
            for split, videos in dataset_split.items():
                if video_dir in videos:
                    assigned_splits.append(split)

            if not assigned_splits:
                print(f"Warning: {video_dir} not assigned to any dataset split, skipping processing.")
                continue

            for filename in os.listdir(video_path):
                if filename.endswith('.json'):
                    json_path = os.path.join(video_path, filename)

                    for split in assigned_splits:
                        output_image_dir = os.path.join(output_dir, split, 'images')
                        output_label_dir = os.path.join(output_dir, split, 'labels')
                        process_json_file(json_path, video_dir, output_image_dir, output_label_dir)

        create_dataset_yaml(output_dir)

        print(f"{split_name} conversion completed. Dataset saved to {output_dir}")
        print(f"{split_name} dataset statistics:")
        for split in ['train', 'val', 'test']:
            img_count = len(os.listdir(os.path.join(output_dir, split, 'images')))
            print(f"  {split}: {img_count} images")

    print(f"\nAll dataset conversions completed! Generated {len(dataset_splits)} dataset splits in total.")

if __name__ == "__main__":
    main()
