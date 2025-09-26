#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os
import sys

# Add Deformable-DETR to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Deformable-DETR'))

import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate
from models import build_model
import util.misc as utils
from datasets.coco_eval import CocoEvaluator


@torch.no_grad()
def evaluate_and_collect_predictions(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, all_predictions):
    """Modified evaluate function that collects all predictions"""
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        for target, result in zip(targets, results):
            image_id = target['image_id'].item()
            boxes = result['boxes'].cpu()
            scores = result['scores'].cpu()
            labels = result['labels'].cpu()

            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box.tolist()
                all_predictions.append({
                    'image_id': image_id,
                    'category_id': label.item(),
                    'bbox': [x1, y1, x2 - x1, y2 - y1],  # COCO format: [x, y, width, height]
                    'score': score.item(),
                    'area': (x2 - x1) * (y2 - y1),
                    'iscrowd': 0
                })

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()

    return stats, coco_evaluator


def save_yolo_labels(coco_results, dataset, output_dir, yolo_labels_dir):
    yolo_dir = Path(output_dir) / yolo_labels_dir
    yolo_dir.mkdir(parents=True, exist_ok=True)

    image_id_to_info = {}
    for img_info in dataset.coco.imgs.values():
        image_id_to_info[img_info['id']] = img_info

    predictions_by_image = {}
    for pred in coco_results:
        image_id = pred['image_id']
        if image_id not in predictions_by_image:
            predictions_by_image[image_id] = []
        predictions_by_image[image_id].append(pred)

    saved_count = 0
    for image_id, preds in predictions_by_image.items():
        if image_id not in image_id_to_info:
            continue

        img_info = image_id_to_info[image_id]
        img_width = img_info['width']
        img_height = img_info['height']
        img_filename = img_info['file_name']

        label_filename = Path(img_filename).stem + '.txt'
        label_path = yolo_dir / label_filename

        with open(label_path, 'w') as f:
            for pred in preds:
                # Convert COCO bbox [x, y, w, h] to YOLO format [x_center, y_center, w, h] (normalized)
                x, y, w, h = pred['bbox']
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                w_norm = w / img_width
                h_norm = h / img_height

                # YOLO class_id (convert from COCO category_id to 0-based)
                class_id = pred['category_id'] - 1  # COCO categories start from 1, YOLO from 0
                confidence = pred['score']
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f} {confidence:.6f}\n")

        saved_count += 1

    return saved_count


def get_args_parser():
    parser = argparse.ArgumentParser('Triplet Deformable DETR Inference', add_help=False)
    
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--sgd', action='store_true')
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    parser.add_argument('--dataset_file', default='triplet')
    parser.add_argument('--coco_path', type=str, required=True, help='Path to triplet dataset')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', default='./inference_output', help='path where to save inference results')
    parser.add_argument('--predictions_file', default='predictions.json', help='filename for COCO predictions')
    parser.add_argument('--save_yolo_labels', action='store_true', help='Save predictions in YOLO format')
    parser.add_argument('--yolo_labels_dir', default='labels', help='Directory name for YOLO labels (relative to output_dir)')
    parser.add_argument('--conf_threshold', type=float, default=0.1, help='Confidence threshold for saving predictions (default: 0.1, mAP calculation uses all predictions)')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")

    print(f"Loading trained model from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location='cpu')

    if 'model' in checkpoint:
        model_state_dict = checkpoint['model']
    else:
        model_state_dict = checkpoint

    expected_num_classes = model_without_ddp.class_embed[0].out_features

    keys_to_remove = []
    need_remove_class_head = False

    for key in list(model_state_dict.keys()):
        if 'class_embed' in key and 'weight' in key:
            checkpoint_num_classes = model_state_dict[key].shape[0]
            if checkpoint_num_classes != expected_num_classes:
                need_remove_class_head = True
                print(f"Class number mismatch: checkpoint has {checkpoint_num_classes}, model expects {expected_num_classes}")
                break

    if need_remove_class_head:
        print("Removing classification head due to class number mismatch")
        for key in list(model_state_dict.keys()):
            if 'class_embed' in key:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del model_state_dict[key]
    else:
        print("Classification head dimensions match, keeping all weights")

    missing_keys, unexpected_keys = model_without_ddp.load_state_dict(model_state_dict, strict=False)
    print(f"Loaded trained model successfully!")
    if len(missing_keys) > 0:
        print(f"Missing keys: {len(missing_keys)}")
    if len(unexpected_keys) > 0:
        print(f"Unexpected keys: {len(unexpected_keys)}")

    dataset_test = build_dataset(image_set='test', args=args)

    if args.distributed:
        if args.cache_mode:
            sampler_test = samplers.NodeDistributedSampler(dataset_test)
        else:
            sampler_test = samplers.DistributedSampler(dataset_test, shuffle=False)
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)

    base_ds = get_coco_api_from_dataset(dataset_test)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting inference on test set...")
    print(f"Test dataset size: {len(dataset_test)}")

    start_time = time.time()

    print("Running inference and collecting predictions...")
    all_predictions = []

    test_stats, coco_evaluator = evaluate_and_collect_predictions(
        model, criterion, postprocessors, data_loader_test, base_ds, device, args.output_dir, all_predictions
    )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Inference time: {total_time_str}')
    coco_results = [pred for pred in all_predictions if pred['score'] >= args.conf_threshold]
    print(f"Total predictions (all): {len(all_predictions)}")
    print(f"Filtered predictions (conf >= {args.conf_threshold}): {len(coco_results)}")

    if all_predictions:
        scores = [pred['score'] for pred in all_predictions]
        print(f"Score distribution:")
        print(f"   - Min: {min(scores):.4f}")
        print(f"   - Max: {max(scores):.4f}")
        print(f"   - Mean: {sum(scores)/len(scores):.4f}")
        print(f"   - Predictions >= 0.1: {len([s for s in scores if s >= 0.1])}")
        print(f"   - Predictions >= 0.3: {len([s for s in scores if s >= 0.3])}")
        print(f"   - Predictions >= 0.5: {len([s for s in scores if s >= 0.5])}")
        print(f"   - Predictions >= 0.7: {len([s for s in scores if s >= 0.7])}")
        print(f"   - FPS: {len(dataset_test) / total_time:.2f} images/second")

        image_ids = [pred['image_id'] for pred in all_predictions]
        unique_images = set(image_ids)
        print(f"   - Images with predictions: {len(unique_images)} / {len(dataset_test)}")

    if coco_results:
        predictions_path = output_dir / args.predictions_file

        with open(predictions_path, 'w') as f:
            json.dump(coco_results, f, indent=2)

        print(f"COCO predictions saved to: {predictions_path}")
        if args.save_yolo_labels and len(coco_results) > 0:
            print(f"Converting to YOLO format...")
            yolo_count = save_yolo_labels(coco_results, dataset_test, args.output_dir, args.yolo_labels_dir)
            yolo_path = output_dir / args.yolo_labels_dir
            print(f"YOLO labels saved to: {yolo_path}")
            print(f"YOLO label files created: {yolo_count}")
    else:
        print("No predictions generated")

    print("\n" + "="*50)
    print("INFERENCE SUMMARY")
    print("="*50)
    print(f"Model: {args.model_path}")
    print(f"Test samples: {len(dataset_test)}")
    print(f"Inference time: {total_time_str}")
    print(f"Output directory: {args.output_dir}")
    print(f"COCO predictions: {args.predictions_file} (filtered with conf >= {args.conf_threshold})")
    if args.save_yolo_labels:
        print(f"YOLO labels: {args.yolo_labels_dir}/ (filtered with conf >= {args.conf_threshold})")
    if coco_evaluator is not None and hasattr(coco_evaluator, 'coco_eval') and 'bbox' in coco_evaluator.coco_eval:
        coco_stats = coco_evaluator.coco_eval['bbox'].stats
        print(f"\n📈 PERFORMANCE METRICS (calculated using ALL predictions):")
        print(f"mAP@0.5:0.95: {coco_stats[0]:.4f}")
        print(f"mAP@0.5:     {coco_stats[1]:.4f}")
        print(f"mAP@0.75:    {coco_stats[2]:.4f}")
    print("="*50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Triplet Deformable DETR Inference', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
