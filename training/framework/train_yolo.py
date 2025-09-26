#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
training example code
"""

import os
import argparse
import yaml
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics
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
                        triplet_id = int(parts[0])
                        verb_id = int(parts[2])
                        target_id = int(parts[3])
                        self.triplet_to_verb[triplet_id] = verb_id
                        self.triplet_to_target[triplet_id] = target_id

        except Exception as e:
            print(f"Error loading extended mapping file {mapping_file}: {e}")
            raise

    def get_verb_id(self, triplet_id):
        return self.triplet_to_verb.get(triplet_id, 0)

    def get_target_id(self, triplet_id):
        return self.triplet_to_target.get(triplet_id, 0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/ssd/prostate/dataset_triplet.yaml', help='config file path')
    parser.add_argument('--model', type=str, default='yolov11n.pt', help='model file or config file path')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--name', type=str, default='yolo11l', help='save results folder name')
    parser.add_argument('--exist-ok', action='store_true', help='overwrite existing experiment folder')
    parser.add_argument('--patience', type=int, default=50, help='early stopping epochs')
    parser.add_argument('--test-only', action='store_true', help='only test')
    parser.add_argument('--cal-ivtd', type=str, default=None, help='Calculate IVT metrics from saved predictions directory (e.g., runs/yolov11l_test/labels)')
    parser.add_argument('--weights', type=str, default=None, help='weights file path for testing')
    parser.add_argument('--conf', type=float, default=0.25, help='confidence threshold for testing')
    parser.add_argument('--iou', type=float, default=0.6, help='IoU threshold for testing')
    parser.add_argument('--save-txt', action='store_true', help='save predictions to txt file')
    parser.add_argument('--save-conf', action='store_true', help='save confidence scores')
    parser.add_argument('--project', type=str, default='runs', help='save results project name')
    parser.add_argument('--optimizer', type=str, default='auto', help='optimizer selection (SGD, Adam, AdamW, etc.)')
    parser.add_argument('--lr0', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01, help='final learning rate = initial learning rate * lrf')
    parser.add_argument('--cos-lr', action='store_true', help='use cosine learning rate scheduler')
    parser.add_argument('--warmup-epochs', type=float, default=3.0, help='warmup epochs')
    parser.add_argument('--warmup-momentum', type=float, default=0.8, help='warmup momentum')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--agnostic-nms', type=eval, default=False, help='use class-agnostic NMS (default False)')
    parser.add_argument('--tool-nms', action='store_true', help='Apply tool-based NMS patch')
    parser.add_argument('--mapping-file', type=str, default='/ssd/prostate/prostate_track_v2/triplet_maps_v2.txt', help='Path to triplet to tool mapping file')
    parser.add_argument('--apply-ivt-metrics', type=eval, default=True, help='Apply IVT metrics patch for AP50-95 calculation (default True)')
    return parser.parse_args()


def apply_ivt_metrics_patch(mapping_file):
    import ivt_metrics_patch
    success = ivt_metrics_patch.apply_patch(mapping_file)
    if success:
        print("IVT metrics patch applied")
    return success


def apply_tool_nms_if_requested(mapping_file):
    import tool_based_nms_patch
    success = tool_based_nms_patch.apply_patch(mapping_file)
    return success

def print_metrics(metrics, class_names=None):
    if class_names is not None and len(class_names) > 0:
        print("\n--- Average metrics ---")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall: {metrics.box.mr:.4f}")
        print(f"F1: {np.mean(metrics.box.f1):.4f}")


def train(args):
    model = YOLO(args.model)
    
    train_args = {
        'data': args.data,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'device': args.device,
        'workers': args.workers,
        'name': args.name,
        'exist_ok': args.exist_ok,
        'patience': args.patience,
        'verbose': True,
        'project': args.project,
        'optimizer': args.optimizer,
        'lr0': args.lr0,
        'lrf': args.lrf,
        'cos_lr': args.cos_lr,
        'warmup_epochs': args.warmup_epochs,
        'warmup_momentum': args.warmup_momentum,
        'dropout': args.dropout
    }
    
    results = model.train(**train_args)
    return model, results


def test(model, args):
    if args.weights and os.path.exists(args.weights):
        model = YOLO(args.weights)

    test_args = {
        'data': args.data,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'device': args.device,
        'conf': args.conf,
        'iou': args.iou,
        'verbose': True,
        'save_txt': args.save_txt,
        'save_conf': args.save_conf,
        'project': args.project,
        'name': args.name + '_test',
        'split': 'test',
        'plots': True,
        'agnostic_nms': args.agnostic_nms,
    }

    results = model.val(**test_args)

    class_names = []
    with open(args.data, 'r') as f:
        data_config = yaml.safe_load(f)
        if 'names' in data_config:
            class_names = data_config['names']


    print_metrics(results, class_names)
    return results

def cal_ivtd(args):
    if args.cal_ivtd:
        prediction_dir = args.cal_ivtd
    else:
        pattern = os.path.join(args.project, args.name + '_test*', 'labels')
        possible_dirs = glob.glob(pattern)
        if possible_dirs:
            prediction_dir = max(possible_dirs, key=os.path.getmtime)
            print(f"Auto-found prediction directory: {prediction_dir}")
        else:
            print("No prediction directories found! Please specify --cal-ivtd path")
            return

    if not os.path.exists(prediction_dir):
        print(f"Prediction directory not found: {prediction_dir}")
        return

    print(f"Reading predictions from: {prediction_dir}")

    pred_files = glob.glob(os.path.join(prediction_dir, "*.txt"))
    if not pred_files:
        print(f"No prediction files found")
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
        threshold=0.5,
        enable_map5095=True
    )

    with open(args.data, 'r') as f:
        data_config = yaml.safe_load(f)
    dataset_root = data_config['path']
    test_split = data_config.get('test', 'test')

    # Handle the case where test split includes 'images' directory
    if test_split.endswith('/images'):
        gt_dir = os.path.join(dataset_root, test_split.replace('/images', '/labels'))
    else:
        gt_dir = os.path.join(dataset_root, test_split, "labels")

    print(f"Dataset root: {dataset_root}")
    print(f"Test split from config: {test_split}")
    print(f"Ground truth directory: {gt_dir}")

    if not os.path.exists(gt_dir):
        print(f"Ground truth directory not found: {gt_dir}")
        return

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

        if video_targets or video_predictions:
            detector.update(video_targets, video_predictions, format="list")
            detector.video_end()

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
    print(f"{'Video-wise':<12} {'IVT':<8} {video_ivt.get('mAP', 0):<8.4f} {video_ivt.get('mAP_5095', 0):<10.4f} {video_ivt.get('mPre', 0):<10.4f} {video_ivt.get('mRec', 0):<8.4f} {video_ivt.get('mF1', 0):<8.4f} {video_ivt.get('mAR_5095', 0):<10.4f}")
    print(f"{'Global':<12} {'IVT':<8} {global_ivt.get('mAP', 0):<8.4f} {global_ivt.get('mAP_5095', 0):<10.4f} {global_ivt.get('mPre', 0):<10.4f} {global_ivt.get('mRec', 0):<8.4f} {global_ivt.get('mF1', 0):<8.4f} {global_ivt.get('mAR_5095', 0):<10.4f}")
    print(f"{'Video-wise':<12} {'I':<8} {video_i.get('mAP', 0):<8.4f} {video_i.get('mAP_5095', 0):<10.4f} {video_i.get('mPre', 0):<10.4f} {video_i.get('mRec', 0):<8.4f} {video_i.get('mF1', 0):<8.4f} {video_i.get('mAR_5095', 0):<10.4f}")
    print(f"{'Global':<12} {'I':<8} {global_i.get('mAP', 0):<8.4f} {global_i.get('mAP_5095', 0):<10.4f} {global_i.get('mPre', 0):<10.4f} {global_i.get('mRec', 0):<8.4f} {global_i.get('mF1', 0):<8.4f} {global_i.get('mAR_5095', 0):<10.4f}")
    print(f"{'Video-wise':<12} {'V':<8} {video_v.get('mAP', 0):<8.4f} {video_v.get('mAP_5095', 0):<10.4f} {video_v.get('mPre', 0):<10.4f} {video_v.get('mRec', 0):<8.4f} {video_v.get('mF1', 0):<8.4f} {video_v.get('mAR_5095', 0):<10.4f}")
    print(f"{'Global':<12} {'V':<8} {global_v.get('mAP', 0):<8.4f} {global_v.get('mAP_5095', 0):<10.4f} {global_v.get('mPre', 0):<10.4f} {global_v.get('mRec', 0):<8.4f} {global_v.get('mF1', 0):<8.4f} {global_v.get('mAR_5095', 0):<10.4f}")
    print(f"{'Video-wise':<12} {'T':<8} {video_t.get('mAP', 0):<8.4f} {video_t.get('mAP_5095', 0):<10.4f} {video_t.get('mPre', 0):<10.4f} {video_t.get('mRec', 0):<8.4f} {video_t.get('mF1', 0):<8.4f} {video_t.get('mAR_5095', 0):<10.4f}")
    print(f"{'Global':<12} {'T':<8} {global_t.get('mAP', 0):<8.4f} {global_t.get('mAP_5095', 0):<10.4f} {global_t.get('mPre', 0):<10.4f} {global_t.get('mRec', 0):<8.4f} {global_t.get('mF1', 0):<8.4f} {global_t.get('mAR_5095', 0):<10.4f}")
    print("-" * 80)


def main():
    args = parse_args()
    if args.cal_ivtd is not None:
        cal_ivtd(args)
        return
    if args.tool_nms:
        tool_nms_applied = apply_tool_nms_if_requested(args.mapping_file)
    if args.apply_ivt_metrics:
        ivt_metrics_applied = apply_ivt_metrics_patch(args.mapping_file)
    if args.test_only:
        if args.weights is None:
            args.weights = os.path.join(args.project, args.name, 'weights/best.pt')
            if not os.path.exists(args.weights):
                return
        
        model = YOLO(args.weights)
        test(model, args)
        return
    
    model, train_results = train(args)
    best_weights = Path(model.trainer.save_dir) / 'weights' / 'best.pt'
    args.weights = str(best_weights)
    test_results = test(model, args)


if __name__ == '__main__':
    main()
