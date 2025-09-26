#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from pathlib import Path
import csv

try:
    import ivtdmetrics
    IVTMETRICS_AVAILABLE = True
except ImportError:
    IVTMETRICS_AVAILABLE = False
    print("Warning: ivtmetrics not installed, cannot calculate IVT metrics")

try:
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from ultralytics.utils import LOGGER
except ImportError as e:
    print(f"Import failed: {e}")
    raise


class TripletMapper:
    def __init__(self, mapping_file):
        if not mapping_file:
            raise ValueError("mapping_file parameter is required")
        self.triplet_to_instrument = {}
        self.load_mapping(mapping_file)
    
    def load_mapping(self, mapping_file):
        try:
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
                        instrument_id = int(parts[1])  # I ID
                        self.triplet_to_instrument[triplet_id] = instrument_id
            
        except Exception as e:
            LOGGER.error(f"Error loading mapping file {mapping_file}: {e}")
            raise
    
    def get_instrument_id(self, triplet_id):
        return self.triplet_to_instrument.get(triplet_id, 0) 


class IVTMetricsValidator(DetectionValidator):
    
    def __init__(self, *args, mapping_file=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ivt_detector = None
        self.triplet_mapper = None
        self.ivt_enabled = IVTMETRICS_AVAILABLE
        self.mapping_file = mapping_file
        
        if self.ivt_enabled:
            self.setup_ivt_metrics()
        else:
            LOGGER.warning("IVT metrics not enabled")
    
    def setup_ivt_metrics(self):
        """Setup IVT metrics calculator"""
        try:
            LOGGER.info("Initializing IVT metrics...")
            
            if not self.mapping_file:
                self.mapping_file = "./triplet_maps_v2.txt"
                LOGGER.warning(f"No mapping file provided, using default: {self.mapping_file}")
            
            self.triplet_mapper = TripletMapper(self.mapping_file)
            num_triplets = max(self.triplet_mapper.triplet_to_instrument.keys()) + 1 if self.triplet_mapper.triplet_to_instrument else 100
            num_tools = max(self.triplet_mapper.triplet_to_instrument.values()) + 1 if self.triplet_mapper.triplet_to_instrument else 6
             
            self.ivt_detector = ivtdmetrics.Detection(
                num_class=num_triplets,
                num_tool=num_tools,
                threshold=0.5,
            )
            
        except Exception as e:
            LOGGER.error(f"Error initializing IVT metrics: {e}")
            import traceback
            LOGGER.error(f"Detailed error: {traceback.format_exc()}")
            self.ivt_enabled = False
    
    def convert_to_ivt_format(self, preds, batch, batch_idx):
        """
        convert YOLO pred and gt to IVT format
        """
        targets_ivt = []
        predictions_ivt = []
        
        try:
            pbatch = self._prepare_batch(batch_idx, batch)
            gt_classes = pbatch["cls"].cpu().numpy() if len(pbatch["cls"]) > 0 else np.array([])
            gt_boxes = pbatch["bbox"].cpu().numpy() if len(pbatch["bbox"]) > 0 else np.array([]).reshape(0, 4)
            
            img_h, img_w = pbatch["ori_shape"]
            for i, (cls_id, box) in enumerate(zip(gt_classes, gt_boxes)):
                triplet_id = int(cls_id)
                tool_id = self.triplet_mapper.get_instrument_id(triplet_id)

                if np.all(box <= 1.0):
                    x1, y1, x2, y2 = box
                    x = x1  # left top x
                    y = y1  # left top y
                    w = x2 - x1  # width
                    h = y2 - y1  # height
                else:
                    x1, y1, x2, y2 = box
                    x = x1 / img_w
                    y = y1 / img_h  
                    w = (x2 - x1) / img_w
                    h = (y2 - y1) / img_h
                targets_ivt.append([triplet_id, tool_id, 1.0, x, y, w, h])
            
            if len(preds) > 0:
                pred = preds[batch_idx] if batch_idx < len(preds) else torch.empty(0, 6)
                
                if len(pred) > 0:
                    pred = self._prepare_pred(pred, pbatch)
                    pred_np = pred.cpu().numpy()
                    
                    for detection in pred_np:
                        x1, y1, x2, y2, conf, cls_id = detection
                        triplet_id = int(cls_id)
                        tool_id = self.triplet_mapper.get_instrument_id(triplet_id)
                        
                        bbox = [x1, y1, x2, y2]
                        if np.all(np.array(bbox) <= 1.0):
                            x, y, w, h = x1, y1, x2 - x1, y2 - y1
                        else:
                            x = x1 / img_w
                            y = y1 / img_h
                            w = (x2 - x1) / img_w
                            h = (y2 - y1) / img_h
                        predictions_ivt.append([triplet_id, tool_id, conf, x, y, w, h])
        
        except Exception as e:
            LOGGER.warning(f"Error converting to IVT format: {e}")
        
        return targets_ivt, predictions_ivt
    
    def update_metrics(self, preds, batch):
        """Update both YOLO and IVT metrics"""
        super().update_metrics(preds, batch)
        
        if self.ivt_enabled and self.ivt_detector is not None:
            all_targets = []
            all_predictions = []
            
            for si in range(len(preds)):
                targets_ivt, predictions_ivt = self.convert_to_ivt_format(preds, batch, si)
                all_targets.append(targets_ivt)
                all_predictions.append(predictions_ivt)
            
            if all_targets or all_predictions:
                self.ivt_detector.update(all_targets, all_predictions, format="list")
    
    def finalize_metrics(self, *args, **kwargs):
        super().finalize_metrics(*args, **kwargs)
        if self.ivt_enabled and self.ivt_detector is not None:
            self.ivt_detector.video_end()
    
    def get_stats(self):
        """Get both YOLO and IVT statistics"""
        stats = super().get_stats()

        if self.ivt_enabled and self.ivt_detector is not None:
            ivt_results = self.ivt_detector.compute_video_AP('ivt', style="coco")
            i_results = self.ivt_detector.compute_video_AP('i', style="coco")
            
            stats.update({
                'ivt/mAP@50': ivt_results.get('mAP', 0.0),
                'ivt/Recall': ivt_results.get('mRec', 0.0),
                'ivt/Precision': ivt_results.get('mPre', 0.0),
                'ivt/mAP@50-95': ivt_results.get('mAP_5095', 0.0),
                'ivt/AR': ivt_results.get('mAR_5095', 0.0),
                'i/mAP@50': i_results.get('mAP', 0.0),
                'i/Recall': i_results.get('mRec', 0.0),
                'i/Precision': i_results.get('mPre', 0.0),
                'i/mAP@50-95': i_results.get('mAP_5095', 0.0),
                'i/AR': i_results.get('mAR_5095', 0.0),
            })
        
        return stats
    
    def print_results(self):
        """Print YOLO results and IVT metrics"""
        super().print_results()

        if self.ivt_enabled and self.ivt_detector is not None:
            LOGGER.info("\n" + "="*50)
            LOGGER.info("IVT Triplet Detection Results")
            LOGGER.info("="*50)
            
            components = [
                ('ivt', 'IVT (Triplet)', 'Instrument+Verb+Target'),
                ('i', 'I (Instrument)', 'Instrument only')
            ]
            
            LOGGER.info(f"{'Component':<15} {'mAP@50':<8} {'Rec':<8} {'Pre':<8} {'F1':<8} {'mAP@50-95':<10} {'AR':<8}")
            LOGGER.info("-" * 68)
            
            for comp_code, comp_name, description in components:
                results = self.ivt_detector.compute_video_AP(comp_code, style="coco")
                mAP = results.get('mAP', 0.0)
                mRec = results.get('mRec', 0.0)
                mPre = results.get('mPre', 0.0)
                mF1 = results.get('mF1', 0.0)
                mAP_5095 = results.get('mAP_5095', 0.0)
                mAR_5095 = results.get('mAR_5095', 0.0)
                
                LOGGER.info(f"{comp_name:<15} {mAP:<8.4f} {mRec:<8.4f} {mPre:<8.4f} {mF1:<8.4f} {mAP_5095:<10.4f} {mAR_5095:<8.4f}")
            
            LOGGER.info("="*68)


def apply_patch(mapping_file=None):
    if not IVTMETRICS_AVAILABLE:
        print("Warning: ivtmetrics not available, skipping patch application")
        return False
    
    class IVTMetricsValidatorWrapper(IVTMetricsValidator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, mapping_file=mapping_file, **kwargs)
    
    import ultralytics.models.yolo.detect.val as val_module
    import ultralytics.models.yolo.detect as detect_module
    import ultralytics.models.yolo as yolo_module
    
    val_module.DetectionValidator = IVTMetricsValidatorWrapper
    detect_module.DetectionValidator = IVTMetricsValidatorWrapper
    if hasattr(yolo_module, 'detect'):
        yolo_module.detect.DetectionValidator = IVTMetricsValidatorWrapper
    
    print("IVT metrics patch applied to all relevant modules")
    return True 