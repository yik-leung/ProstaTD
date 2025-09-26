#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Independent IVTD Metrics Calculator
Calculate IVT metrics from saved prediction txt files without running YOLO model.

Usage:
    python calculate_ivtd_simple.py --data "/path/to/gt/labels" --cal-ivtd "/path/to/predictions/labels" --mapping-file "/path/to/triplet_maps.txt"
"""

import os
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
import glob
import ivtdmetrics
from ivt_metrics_patch import TripletMapper


class ExtendedTripletMapper(TripletMapper):
    def __init__(self, mapping_file):
        super().__init__(mapping_file)
        self.triplet_to_verb = {}
        self.triplet_to_target = {}
        self._load_extended_mapping(mapping_file)

    def _load_extended_mapping(self, mapping_file):
        try:
            from pathlib import Path
            mapping_path = Path(mapping_file)
            if not mapping_path.exists():
                raise FileNotFoundError(f"Mapping file not found: {mapping_file}")

            with open(mapping_path, 'r') as f:
                lines = f.readlines()

            for line in lines[1:]: 
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split(',')
                    if len(parts) >= 6:
                        triplet_id = int(parts[0])  # IVT ID
                        verb_id = int(parts[2])     # V ID
                        target_id = int(parts[3])   # T ID
                        self.triplet_to_verb[triplet_id] = verb_id
                        self.triplet_to_target[triplet_id] = target_id

        except Exception as e:
            print(f"Warning: Could not load extended mapping: {e}")
            print("Using default mapping (verb_id = target_id = 0)")

    def get_verb_id(self, triplet_id):
        return self.triplet_to_verb.get(triplet_id, 0)

    def get_target_id(self, triplet_id):
        return self.triplet_to_target.get(triplet_id, 0)


def parse_args():
    parser = argparse.ArgumentParser(description='Calculate IVTD metrics from prediction files')
    parser.add_argument('--data', type=str, default="/root/autodl-tmp/prostate/dataset_yolo_triplet/split3/test/labels",
                       help='Path to ground truth labels directory')
    parser.add_argument('--cal-ivtd', type=str, default="inference_results_split3/labels",
                       help='Path to prediction labels directory')
    parser.add_argument('--mapping-file', type=str, default="./triplet_maps_v2.txt",
                       help='Path to triplet mapping file')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='IoU threshold for evaluation')
    parser.add_argument('--enable-map5095', action='store_true', default=True,
                       help='Enable mAP@50-95 calculation')
    
    return parser.parse_args()


def cal_ivtd(args):
    prediction_dir = args.cal_ivtd
    if not os.path.exists(prediction_dir):
        print(f"Error: Prediction directory not found: {prediction_dir}")
        return

    print(f"Reading predictions from: {prediction_dir}")

    gt_dir = args.data
    if not os.path.exists(gt_dir):
        print(f"Error: Ground truth directory not found: {gt_dir}")
        return

    print(f"Ground truth directory: {gt_dir}")

    pred_files = glob.glob(os.path.join(prediction_dir, "*.txt"))
    if not pred_files:
        print(f"Error: No prediction files found in {prediction_dir}")
        return

    print(f"Found {len(pred_files)} prediction files")

    video_groups = defaultdict(list)
    for pred_file in pred_files:
        filename = os.path.basename(pred_file)
        name_without_ext = os.path.splitext(filename)[0]

        last_underscore = name_without_ext.rfind('_')
        if last_underscore != -1:
            video_name = name_without_ext[:last_underscore]
        else:
            video_name = name_without_ext

        video_groups[video_name].append(filename)

    print(f"Found {len(video_groups)} video groups:")
    for video_name, files in video_groups.items():
        print(f"   - {video_name}: {len(files)} files")

    triplet_mapper = ExtendedTripletMapper(args.mapping_file)
    num_triplets = max(triplet_mapper.triplet_to_instrument.keys()) + 1
    num_tools = max(triplet_mapper.triplet_to_instrument.values()) + 1
    num_verbs = max(triplet_mapper.triplet_to_verb.values()) + 1
    num_targets = max(triplet_mapper.triplet_to_target.values()) + 1

    print(f"IVT detector: {num_triplets} triplets, {num_tools} tools, {num_verbs} verbs, {num_targets} targets")

    detector = ivtdmetrics.Detection(
        num_class=num_triplets,
        num_tool=num_tools,
        num_verb=num_verbs,
        num_target=num_targets,
        threshold=args.threshold,
        enable_map5095=args.enable_map5095
    )

    total_gt_files = 0
    total_pred_files = 0
    total_targets = 0
    total_predictions = 0

    for video_idx, (video_name, filenames) in enumerate(video_groups.items()):
        video_targets = []
        video_predictions = []
        gt_found = 0
        pred_found = 0

        for filename in sorted(filenames):
            gt_path = os.path.join(gt_dir, filename)
            targets = []
            if os.path.exists(gt_path):
                gt_found += 1
                with open(gt_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            cx, cy, w, h = map(float, parts[1:5])
                            x1, y1 = cx - w/2, cy - h/2
                            tool_id = triplet_mapper.get_instrument_id(cls_id)
                            verb_id = triplet_mapper.get_verb_id(cls_id)
                            target_id = triplet_mapper.get_target_id(cls_id)
                            targets.append([cls_id, tool_id, 1.0, verb_id, target_id, x1, y1, w, h])

            pred_path = os.path.join(prediction_dir, filename)
            predictions = []
            if os.path.exists(pred_path):
                pred_found += 1
                with open(pred_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) >= 6:
                                cls_id = int(parts[0])
                                cx, cy, w, h, conf = map(float, parts[1:6])
                                x1, y1 = cx - w/2, cy - h/2
                                tool_id = triplet_mapper.get_instrument_id(cls_id)
                                verb_id = triplet_mapper.get_verb_id(cls_id)
                                target_id = triplet_mapper.get_target_id(cls_id)
                                predictions.append([cls_id, tool_id, conf, verb_id, target_id, x1, y1, w, h])

            video_targets.append(targets)
            video_predictions.append(predictions)

        total_gt_files += gt_found
        total_pred_files += pred_found
        total_targets += sum(len(t) for t in video_targets)
        total_predictions += sum(len(p) for p in video_predictions)

        if video_targets or video_predictions:
            detector.update(video_targets, video_predictions, format="list")
            detector.video_end()

    print(f"\nProcessing summary:")
    print(f"  - GT files: {total_gt_files}")
    print(f"  - Prediction files: {total_pred_files}")
    print(f"  - Total targets: {total_targets}")
    print(f"  - Total predictions: {total_predictions}")

    print("\nCalculating metrics...")
    
    video_ivt = detector.compute_video_AP('ivt', style="coco")
    video_i = detector.compute_video_AP('i', style="coco")
    video_v = detector.compute_video_AP('v', style="coco")
    video_t = detector.compute_video_AP('t', style="coco")

    global_ivt = detector.compute_global_AP('ivt', style="coco")
    global_i = detector.compute_global_AP('i', style="coco")
    global_v = detector.compute_global_AP('v', style="coco")
    global_t = detector.compute_global_AP('t', style="coco")

    print("=" * 80)
    print(f"\n{'Metric Type':<12} {'Component':<8} {'mAP@50':<8} {'mAP@50-95':<10} {'Precision':<10} {'Recall':<8} {'F1':<8} {'AR':<10}")
    print("-" * 80)
    print(f"{'Global':<12} {'IVT':<8} {global_ivt.get('mAP', 0):<8.4f} {global_ivt.get('mAP_5095', 0):<10.4f} {global_ivt.get('mPre', 0):<10.4f} {global_ivt.get('mRec', 0):<8.4f} {global_ivt.get('mF1', 0):<8.4f} {global_ivt.get('mAR_5095', 0):<10.4f}")
    print(f"{'Global':<12} {'I':<8} {global_i.get('mAP', 0):<8.4f} {global_i.get('mAP_5095', 0):<10.4f} {global_i.get('mPre', 0):<10.4f} {global_i.get('mRec', 0):<8.4f} {global_i.get('mF1', 0):<8.4f} {global_i.get('mAR_5095', 0):<10.4f}")
    print(f"{'Global':<12} {'V':<8} {global_v.get('mAP', 0):<8.4f} {global_v.get('mAP_5095', 0):<10.4f} {global_v.get('mPre', 0):<10.4f} {global_v.get('mRec', 0):<8.4f} {global_v.get('mF1', 0):<8.4f} {global_v.get('mAR_5095', 0):<10.4f}")
    print(f"{'Global':<12} {'T':<8} {global_t.get('mAP', 0):<8.4f} {global_t.get('mAP_5095', 0):<10.4f} {global_t.get('mPre', 0):<10.4f} {global_t.get('mRec', 0):<8.4f} {global_t.get('mF1', 0):<8.4f} {global_t.get('mAR_5095', 0):<10.4f}")
    print("-" * 80)
    print(f"{'Video-wise':<12} {'IVT':<8} {video_ivt.get('mAP', 0):<8.4f} {video_ivt.get('mAP_5095', 0):<10.4f} {video_ivt.get('mPre', 0):<10.4f} {video_ivt.get('mRec', 0):<8.4f} {video_ivt.get('mF1', 0):<8.4f} {video_ivt.get('mAR_5095', 0):<10.4f}")
    print(f"{'Video-wise':<12} {'I':<8} {video_i.get('mAP', 0):<8.4f} {video_i.get('mAP_5095', 0):<10.4f} {video_i.get('mPre', 0):<10.4f} {video_i.get('mRec', 0):<8.4f} {video_i.get('mF1', 0):<8.4f} {video_i.get('mAR_5095', 0):<10.4f}")
    print(f"{'Video-wise':<12} {'V':<8} {video_v.get('mAP', 0):<8.4f} {video_v.get('mAP_5095', 0):<10.4f} {video_v.get('mPre', 0):<10.4f} {video_v.get('mRec', 0):<8.4f} {video_v.get('mF1', 0):<8.4f} {video_v.get('mAR_5095', 0):<10.4f}")
    print(f"{'Video-wise':<12} {'T':<8} {video_t.get('mAP', 0):<8.4f} {video_t.get('mAP_5095', 0):<10.4f} {video_t.get('mPre', 0):<10.4f} {video_t.get('mRec', 0):<8.4f} {video_t.get('mF1', 0):<8.4f} {video_t.get('mAR_5095', 0):<10.4f}")
    print("-" * 80)
    print("Calculation completed successfully!")

def main():
    args = parse_args()
    print("=" * 80)
    print("IVTD Metrics Calculator")
    print("=" * 80)
    print(f"Ground truth: {args.data}")
    print(f"Predictions: {args.cal_ivtd}")
    print(f"Mapping file: {args.mapping_file}")
    print(f"IoU threshold: {args.threshold}")
    print(f"mAP@50-95 enabled: {args.enable_map5095}")
    print("-" * 80)
    cal_ivtd(args)

if __name__ == "__main__":
    main()
