#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%%%%%%% imports %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
import sys
import warnings
from ivtdmetrics.recognition import Recognition

#%%%%%%%%%% DETECTION AND ASSOCIATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Detection(Recognition):
    """
    @args: 
        init(num_class, num_tool, num_target, threshold):
            num_class: number of triplet categories
            num_tool: number of tool categories
            num_target: number of target categories
            threshold: IoU overlap threshold score
        
    @call: 
        update(targets, predictions):
            targets: groundtruth json file with "frameid" as key
            predictions: prediction json file with "frameid" as key
            json format:
                frameid1: {
                            "recognition": List of class-wise probabilities in the 
                                format:  [score1, score2, score3, ..., score100],
                                example: [0.001, 0.025, 0.911, ...].
                            "detection" List of list of box detection for each triplet in the 
                                format:  [[class1,score,x,y,w,h], [class3,score,x,y,w,h], [class56,score,x,y,w,h], ...],
                                example: [[1,0.842,0.21,0.09,0.32,0.33], [3,0.992,0.21,0.09,0.32,0.33], [56,0.422,0.21,0.09,0.32,0.33], ....] 
                         }
                frameid2: {...}
                    :
        extract(input, componet): filter a component labels from the inputs labels
        compute_AP('i/ivt') return AP for current run
        compute_video_AP('i/ivt') return AP from video-wise averaging
        compute_global_AP('i/ivt') return AP for all seen examples
        reset_video()
          
    @return     
        output:
            detection and association performances:
                AP: average precision
                mAP: mean average precision
                Rec: Recall
                mRec: Mean Recall
                Pre: Precision
                mPrec: Mean Precision
                lm: localize and match percent
                plm: partially localize and match
                ids: identity switch
                idm: identity miss
                mil: missed localization
                fp: false positives
                fn: false negatives    
    @params
    --------  
    @methods
    -------
    @format
        box format: [{"triplet":tid, "instrument":[tool, 1.0, x,y,w,h], "target":[]}]
    """
    def __init__(self, num_class=100, num_tool=6, num_verb=10, num_target=15, threshold=0.5, enable_map5095=True):
        super(Recognition, self).__init__()
        self.num_class      = num_class
        self.num_tool       = num_tool
        self.num_verb       = num_verb
        self.num_target     = num_target
        self.classwise_ap   = []
        self.classwise_rec  = []
        self.classwise_prec = []
        self.accumulator    = {}
        self.video_count    = 0
        self.end_call       = False
        self.threshold      = threshold
        self.enable_map5095 = enable_map5095  # Renamed parameter to control mAP50-95 computation
        # IoU thresholds for mAP50-95 (0.5 to 0.95 in steps of 0.05)
        self.iou_thresholds = np.arange(0.5, 1.0, 0.05) if enable_map5095 else [threshold]
        self.reset()
                
    def reset(self):
        self.video_count = 0
        self.video_end()  
        
    def reset_global(self):
        self.video_count = 0
        self.video_end()    
        
    def video_end(self):
        self.video_count += 1
        self.end_call = True
        # Initialize storage for different IoU thresholds
        num_thresholds = len(self.iou_thresholds)
        self.accumulator[self.video_count] = {
            "hits":  [[] for _ in range(self.num_class)],
            "ndet":  [0  for _ in range(self.num_class)],
            "npos":  [0  for _ in range(self.num_class)],                             
            "hits_i":[[] for _ in range(self.num_tool)],
            "ndet_i":[0  for _ in range(self.num_tool)] ,
            "npos_i":[0  for _ in range(self.num_tool)] ,
            # Add hits for multiple IoU thresholds (for mAP50-95)
            "hits_5095": [[[] for _ in range(self.num_class)] for _ in range(num_thresholds)] if self.enable_map5095 else None,
            "hits_i_5095": [[[] for _ in range(self.num_tool)] for _ in range(num_thresholds)] if self.enable_map5095 else None,
            "hits_v_5095": [[[] for _ in range(self.num_verb)] for _ in range(num_thresholds)] if self.enable_map5095 else None,
            "hits_t_5095": [[[] for _ in range(self.num_target)] for _ in range(num_thresholds)] if self.enable_map5095 else None,
            "fp": 0,
            "fn": 0,
            "lm": 0,
            "plm": 0,
            "ids": 0,
            "idm": 0,
            "mil": 0,
            "conf": [[] for _ in range(self.num_class)],
            "conf_i": [[] for _ in range(self.num_tool)],
            "hits_v":[[] for _ in range(self.num_verb)],
            "ndet_v":[0  for _ in range(self.num_verb)],
            "npos_v":[0  for _ in range(self.num_verb)],
            "conf_v": [[] for _ in range(self.num_verb)],
            "hits_t":[[] for _ in range(self.num_target)],
            "ndet_t":[0  for _ in range(self.num_target)],
            "npos_t":[0  for _ in range(self.num_target)],
            "conf_t": [[] for _ in range(self.num_target)]
        }
    
    def xywh2xyxy(self, bb):
         # [Flag] 
        bb_copy = bb.copy() if hasattr(bb, 'copy') else list(bb)
        bb_copy[2] += bb_copy[0]
        bb_copy[3] += bb_copy[1]
        return bb_copy    
    
    def iou(self, bb1, bb2):
        bb1 = self.xywh2xyxy(bb1)
        bb2 = self.xywh2xyxy(bb2)
        x1 = bb1[2] - bb1[0]
        y1 = bb1[3] - bb1[1]
        if x1 < 0: x1 = 0
        if y1 < 0: y1 = 0
        x2 = bb2[2] - bb2[0]
        y2 = bb2[3] - bb2[1]
        if x2 < 0: x2 = 0
        if y2 < 0: y2 = 0
        xiou = min(bb1[2], bb2[2]) - max(bb1[0], bb2[0])
        yiou = min(bb1[3], bb2[3]) - max(bb1[1], bb2[1])
        if xiou < 0: xiou = 0
        if yiou < 0: yiou = 0
        if xiou * yiou <= 0:
            return 0
        else:
            return xiou * yiou / (x1 * y1 + x2 * y2 - xiou * yiou)        
        
    def is_match(self, det_gt, det_pd, threshold):  
        if det_gt[0] == det_pd[0]: # cond 1: correct identity        
            if self.iou(det_gt[-4:], det_pd[-4:]) >= threshold: # cond 2: sufficient iou
                return True
        return False 
    
    def is_match_multi_iou(self, det_gt, det_pd, iou_thresholds):  
        """Check if detection matches ground truth for multiple IoU thresholds"""
        if det_gt[0] == det_pd[0]: # cond 1: correct identity        
            iou_value = self.iou(det_gt[-4:], det_pd[-4:])
            return [iou_value >= threshold for threshold in iou_thresholds]
        return [False] * len(iou_thresholds)
    
    def is_partial_match(self, det_gt, det_pd):  
        if det_gt[0] == det_pd[0]: # cond 1: correct identity        
            if self.iou(det_gt[-4:], det_pd[-4:]) > 0.0: # cond 2: insufficient iou
                return True
        return False
        
    def is_id_switch(self, det_gt, det_pd, det_gts, threshold):        
        if self.iou(det_gt[-4:], det_pd[-4:]) > threshold: # cond 2: insufficient/sufficient iou 
            gt_ids = list(det_gts[:,0])
            if det_pd[0] in gt_ids: # cond 1: switched identity
                return np.where(gt_ids==det_pd[0])[0][0]
        return False    
    
    def is_id_miss(self, det_gt, det_pd, threshold):        
        if self.iou(det_gt[-4:], det_pd[-4:]) > threshold: # cond 2: insufficient/sufficient iou 
                return True
        return False
        
    def is_miss_loc(self, det_gt, det_pd, det_gts):        
        gt_ids = list(det_gts[:,0])
        if det_pd[0] in gt_ids: # cond 1: correct identity only
            return np.where(gt_ids==det_pd[0])[0][0]
        return False   
    
    def separate_detection(self, det_gts, det_pds):
        pos_ids = list(det_gts[:,0])
        matching_dets   = [list(x) for x in det_pds if x[0] in pos_ids]
        unmatching_dets = [list(x) for x in det_pds if x[0] not in pos_ids]
        return matching_dets, unmatching_dets
    
    def list2stack(self, x):
        # if x == []: x = [[-1,-1,-1,-1,-1,-1]] # empty
        if x == []: 
            return np.array([[]]), []  # Return (detection_array, conf_list)
        #x format for a single frame: list(list): each list = [tripletID, toolID, toolProbs, x, y, w, h] bbox is scaled (0..1)
        assert isinstance(x[0], list), "Each frame must be a list of lists, each list a prediction of triplet and object locations"        
        if len(x[0]):
            x_array = np.stack(x, axis=0)
            conf_list = x_array[:, 2].tolist()  # Extract confidence
            sorted_indices = x_array[:,2].argsort()[::-1]  # Keep single frame sorting
            x_sorted = x_array[sorted_indices]
            conf_sorted = [conf_list[i] for i in sorted_indices]  # Sort confidence similarly
            return x_sorted, conf_sorted
        return np.array([[]]), []
    
    def sortstack(self, x):
        #x format for a single frame: list(list): each list = [tripletID, toolID, toolProbs, x, y, w, h] bbox is scaled (0..1)
        assert isinstance(x, np.ndarray), "Each frame must be an n-dim array with each row a unique prediction of triplet and object locations"
        x = x[x[:,2].argsort()[::-1]]
        return x
        
    def dict2stack(self, x):
        #x format for a single frame: list(dict): each dict = {"triplet":ID, "instrument": [ID, Probs, x, y, w, h]} bbox is scaled (0..1)
        assert isinstance(x, list), "Each frame must be a list of dictionaries"        
        y = []
        for d in x:
            assert isinstance(d, dict), "Each frame must be a list of dictionaries, each dictionary a prediction of triplet and object locations"
            p = [d['triplet']]
            p.extend(d["instrument"])
            y.append(p)
        detection_array, conf_list = self.list2stack(y)
        return detection_array, conf_list
    
    def update(self, targets, predictions, format="list"): 
        [self.update_frame(y, f, format) for y,f in zip(targets, predictions)]
#        print("First")
#        formats = [format]* len(targets)
#        map(self.update_frame, targets, predictions, formats)  
#        for item in range(len(targets)):
#            self.update_frame(targets[item], predictions[item], format)
        self.end_call = False 
    
    def update_frame(self, targets, predictions, format="list"):
        if format=="list":            
            detection_gt, _ = self.list2stack(targets)  # GT does not need confidence
            detection_pd, pred_conf = self.list2stack(predictions)  # Get confidence simultaneously
        elif format=="dict":
            detection_gt, _ = self.dict2stack(targets)  
            detection_pd, pred_conf = self.dict2stack(predictions)  
        else:
            sys.exit("unkown input format for update function. Must be a list or dict")
        
        # Check if there is data
        gt_has_data = len(detection_gt) > 0 and len(detection_gt[0]) > 0
        pd_has_data = len(detection_pd) > 0 and len(detection_pd[0]) > 0

        if not gt_has_data and not pd_has_data:
            return

        # Auto-detect format based on input length
        # 7 elements: [cls_id, tool_id, conf, x1, y1, w, h] - compute IVT + I only
        # 9 elements: [cls_id, tool_id, conf, verb_id, target_id, x1, y1, w, h] - compute all components
        compute_vt = False
        if pd_has_data and len(detection_pd[0]) == 9:
            compute_vt = True
        elif gt_has_data and len(detection_gt[0]) == 9:
            compute_vt = True
            
        detection_gt_ivt = detection_gt.copy()
        detection_pd_ivt = detection_pd.copy()
        # for triplet        
        for gt in detection_gt_ivt: 
            if len(gt): self.accumulator[self.video_count]["npos"][int(gt[0])] += 1
        for i, det_pd in enumerate(detection_pd_ivt):
            if len(det_pd):
                self.accumulator[self.video_count]["ndet"][int(det_pd[0])] += 1
                matched = False                
                for k, det_gt in enumerate(detection_gt_ivt):
                    if len(det_gt): 
                        y = det_gt[0:] 
                        f = det_pd[0:]
                        if self.is_match(y, f, threshold=self.threshold):
                            detection_gt_ivt = np.delete(detection_gt_ivt, obj=k, axis=0)
                            matched = True
                            break
                if matched:
                    self.accumulator[self.video_count]["hits"][int(det_pd[0])].append(1.0)
                    self.accumulator[self.video_count]["conf"][int(det_pd[0])].append(pred_conf[i])  # Collect confidence
                    
                    # Add hits for multiple IoU thresholds (mAP50-95)
                    if self.enable_map5095:
                        # Use the same matched GT (y) and detection (f) for multi-IoU check
                        matches = self.is_match_multi_iou(y, f, self.iou_thresholds)
                        for thresh_idx, match in enumerate(matches):
                            if match:
                                self.accumulator[self.video_count]["hits_5095"][thresh_idx][int(det_pd[0])].append(1.0)
                            else:
                                self.accumulator[self.video_count]["hits_5095"][thresh_idx][int(det_pd[0])].append(0.0)
                else:
                    self.accumulator[self.video_count]["hits"][int(det_pd[0])].append(0.0)
                    self.accumulator[self.video_count]["conf"][int(det_pd[0])].append(pred_conf[i])  # Collect confidence
                    
                    # Add hits for multiple IoU thresholds (mAP50-95)
                    if self.enable_map5095:
                        for thresh_idx in range(len(self.iou_thresholds)):
                            self.accumulator[self.video_count]["hits_5095"][thresh_idx][int(det_pd[0])].append(0.0)
        # for instrument       
        detection_gt_i = detection_gt.copy()
        detection_pd_i = detection_pd.copy() 
        for gt in detection_gt_i:
            if len(gt): self.accumulator[self.video_count]["npos_i"][int(gt[1])] += 1        
        for i, det_pd in enumerate(detection_pd_i):
            if len(det_pd):
                self.accumulator[self.video_count]["ndet_i"][int(det_pd[1])] += 1
                matched = False                
                for k, det_gt in enumerate(detection_gt_i): 
                    if len(det_gt): 
                        y = det_gt[1:] 
                        f = det_pd[1:]
                        if self.is_match(y, f, threshold=self.threshold):
                            detection_gt_i = np.delete(detection_gt_i, obj=k, axis=0)
                            matched = True
                            break
                if matched:
                    self.accumulator[self.video_count]["hits_i"][int(det_pd[1])].append(1.0)
                    self.accumulator[self.video_count]["conf_i"][int(det_pd[1])].append(pred_conf[i])  # Collect confidence
                    
                    # Add hits for multiple IoU thresholds (mAP50-95)
                    if self.enable_map5095:
                        # Use the same matched GT (y) and detection (f) for multi-IoU check
                        matches = self.is_match_multi_iou(y, f, self.iou_thresholds)
                        for thresh_idx, match in enumerate(matches):
                            if match:
                                self.accumulator[self.video_count]["hits_i_5095"][thresh_idx][int(det_pd[1])].append(1.0)
                            else:
                                self.accumulator[self.video_count]["hits_i_5095"][thresh_idx][int(det_pd[1])].append(0.0)
                else:
                    self.accumulator[self.video_count]["hits_i"][int(det_pd[1])].append(0.0)
                    self.accumulator[self.video_count]["conf_i"][int(det_pd[1])].append(pred_conf[i])  # Collect confidence
                    
                    # Add hits for multiple IoU thresholds (mAP50-95)
                    if self.enable_map5095:
                        for thresh_idx in range(len(self.iou_thresholds)):
                            self.accumulator[self.video_count]["hits_i_5095"][thresh_idx][int(det_pd[1])].append(0.0)

        # for verb component (index 3) - only if 8-element format
        if compute_vt:
            detection_gt_v = detection_gt.copy()
            detection_pd_v = detection_pd.copy()
            for gt in detection_gt_v:
                if len(gt): self.accumulator[self.video_count]["npos_v"][int(gt[3])] += 1
            for i, det_pd in enumerate(detection_pd_v):
                if len(det_pd):
                    self.accumulator[self.video_count]["ndet_v"][int(det_pd[3])] += 1
                    matched = False
                    for k, det_gt in enumerate(detection_gt_v):
                        if len(det_gt):
                            y = det_gt[3:]  # [verb_id, target_id, x1, y1, w, h]
                            f = det_pd[3:]  # [verb_id, target_id, x1, y1, w, h]
                            if self.is_match(y, f, threshold=self.threshold):
                                detection_gt_v = np.delete(detection_gt_v, obj=k, axis=0)
                                matched = True
                                break
                    if matched:
                        self.accumulator[self.video_count]["hits_v"][int(det_pd[3])].append(1.0)
                        self.accumulator[self.video_count]["conf_v"][int(det_pd[3])].append(pred_conf[i])  # Collect confidence

                        # Add hits for multiple IoU thresholds (mAP50-95)
                        if self.enable_map5095:
                            # Use the same matched GT (y) and detection (f) for multi-IoU check
                            matches = self.is_match_multi_iou(y, f, self.iou_thresholds)
                            for thresh_idx, match in enumerate(matches):
                                if match:
                                    self.accumulator[self.video_count]["hits_v_5095"][thresh_idx][int(det_pd[3])].append(1.0)
                                else:
                                    self.accumulator[self.video_count]["hits_v_5095"][thresh_idx][int(det_pd[3])].append(0.0)
                    else:
                        self.accumulator[self.video_count]["hits_v"][int(det_pd[3])].append(0.0)
                        self.accumulator[self.video_count]["conf_v"][int(det_pd[3])].append(pred_conf[i])  # Collect confidence

                        # Add hits for multiple IoU thresholds (mAP50-95)
                        if self.enable_map5095:
                            for thresh_idx in range(len(self.iou_thresholds)):
                                self.accumulator[self.video_count]["hits_v_5095"][thresh_idx][int(det_pd[3])].append(0.0)

        # for target component (index 4) - only if 8-element format
        if compute_vt:
            detection_gt_t = detection_gt.copy()
            detection_pd_t = detection_pd.copy()
            for gt in detection_gt_t:
                if len(gt): self.accumulator[self.video_count]["npos_t"][int(gt[4])] += 1
            for i, det_pd in enumerate(detection_pd_t):
                if len(det_pd):
                    self.accumulator[self.video_count]["ndet_t"][int(det_pd[4])] += 1
                    matched = False
                    for k, det_gt in enumerate(detection_gt_t):
                        if len(det_gt):
                            y = det_gt[4:]  # [target_id, x1, y1, w, h]
                            f = det_pd[4:]  # [target_id, x1, y1, w, h]
                            if self.is_match(y, f, threshold=self.threshold):
                                detection_gt_t = np.delete(detection_gt_t, obj=k, axis=0)
                                matched = True
                                break
                    if matched:
                        self.accumulator[self.video_count]["hits_t"][int(det_pd[4])].append(1.0)
                        self.accumulator[self.video_count]["conf_t"][int(det_pd[4])].append(pred_conf[i])  # Collect confidence

                        # Add hits for multiple IoU thresholds (mAP50-95)
                        if self.enable_map5095:
                            # Use the same matched GT (y) and detection (f) for multi-IoU check
                            matches = self.is_match_multi_iou(y, f, self.iou_thresholds)
                            for thresh_idx, match in enumerate(matches):
                                if match:
                                    self.accumulator[self.video_count]["hits_t_5095"][thresh_idx][int(det_pd[4])].append(1.0)
                                else:
                                    self.accumulator[self.video_count]["hits_t_5095"][thresh_idx][int(det_pd[4])].append(0.0)
                    else:
                        self.accumulator[self.video_count]["hits_t"][int(det_pd[4])].append(0.0)
                        self.accumulator[self.video_count]["conf_t"][int(det_pd[4])].append(pred_conf[i])  # Collect confidence

                        # Add hits for multiple IoU thresholds (mAP50-95)
                        if self.enable_map5095:
                            for thresh_idx in range(len(self.iou_thresholds)):
                                self.accumulator[self.video_count]["hits_t_5095"][thresh_idx][int(det_pd[4])].append(0.0)

        # process association
        self.association(targets=detection_gt.copy(), predictions=detection_pd.copy())
        
    def association(self, targets, predictions): 
        detection_gt = targets.copy()
        detection_pd = predictions.copy()
        if len(detection_gt[0])==0:
            self.accumulator[self.video_count]["fp"] += len([x for x in detection_pd if len(x)])
        elif len(detection_pd[0])==0:    
            self.accumulator[self.video_count]["fn"] += len([x for x in detection_gt if len(x)])
        else:
            # separate
            matched_dets, unmatched_dets = self.separate_detection(detection_gt, detection_pd)
            # compare
            matched_dets, detection_gt  = self.localized_box_matched_id(matched_dets, detection_gt)
            matched_dets, detection_gt  = self.partially_localized_box_matched_id(matched_dets, detection_gt)
            matched_dets, detection_gt  = self.localized_box_switched_id(matched_dets, detection_gt)
            matched_dets, detection_gt  = self.localized_box_missed_id(matched_dets, unmatched_dets, detection_gt)
            matched_dets, detection_gt  = self.unlocalized_box_matched_id(matched_dets, detection_gt)        
            # False positives and negatives
            self.accumulator[self.video_count]["fp"] += len([x for x in matched_dets if len(x)])
            self.accumulator[self.video_count]["fn"] += len([x for x in detection_gt if len(x)])
        return
    
    def localized_box_matched_id(self, matched_dets, detection_gt):
        # LM: fully localized and matched
        leftover = []
        if len(matched_dets):
            for det_pd in matched_dets: 
                f = det_pd[0:]
                matched = False
                # if len(detection_gt[0]):
                for k, det_gt in enumerate(detection_gt):
                    y = det_gt[0:]
                    if self.is_match(y, f, threshold=self.threshold):
                        detection_gt = np.delete(detection_gt, obj=k, axis=0)
                        matched = True
                        break                
                if matched:
                    self.accumulator[self.video_count]["lm"] += 1
                else:
                    leftover.append(det_pd)
        matched_dets = leftover.copy()
        return matched_dets, detection_gt            
            
    def partially_localized_box_matched_id(self, matched_dets, detection_gt):
        # pLM: partially localized and matched
        leftover = []
        if len(matched_dets):
            for det_pd in matched_dets: 
                f = det_pd[0:]
                matched = False
                # if len(detection_gt[0]):
                for k, det_gt in enumerate(detection_gt):
                    y = det_gt[0:]
                    if self.is_partial_match(y, f):
                        detection_gt = np.delete(detection_gt, obj=k, axis=0)
                        matched = True
                        break                
                if matched:
                    self.accumulator[self.video_count]["plm"] += 1
                else:
                    leftover.append(det_pd)
        matched_dets = leftover.copy()
        return matched_dets, detection_gt        
        
    def localized_box_switched_id(self, matched_dets, detection_gt):
        # IDS: partially localized but identity switched
        leftover = []
        if len(matched_dets):
            for det_pd in matched_dets: 
                f = det_pd[0:]
                matched = False
                # if len(detection_gt[0]):
                for k, det_gt in enumerate(detection_gt):
                    y   = det_gt[0:]
                    ids = self.is_id_switch(y, f, detection_gt, threshold=self.threshold)
                    if ids:
                        detection_gt = np.delete(detection_gt, obj=ids, axis=0)
                        matched = True
                        break  
                if matched:
                    self.accumulator[self.video_count]["ids"] += 1
                else:
                    leftover.append(det_pd)  
        matched_dets = leftover.copy()
        return matched_dets, detection_gt                
                
    def localized_box_missed_id(self, matched_dets, unmatched_dets, detection_gt):
        # IDS: partially localized but identity missed
        unmatched_dets += matched_dets
        leftover = []
        if len(matched_dets):
            for det_pd in unmatched_dets: 
                f = det_pd[0:]
                matched = False
                # if len(detection_gt[0]):
                for k, det_gt in enumerate(detection_gt):
                    y   = det_gt[0:]
                    if self.is_id_miss(y, f, threshold=self.threshold):
                        matched = True
                        break  
                if matched:
                    self.accumulator[self.video_count]["idm"] += 1
                else:
                    leftover.append(det_pd)  
        matched_dets = leftover.copy()
        return matched_dets, detection_gt
        
    def unlocalized_box_matched_id(self, matched_dets, detection_gt):
        # IDS: partially localized but identity switched
        leftover = []
        if len(matched_dets):
            for det_pd in matched_dets: 
                f = det_pd[0:]
                matched = False
                # if len(detection_gt[0]):
                for k, det_gt in enumerate(detection_gt):
                    y   = det_gt[0:]
                    ids = self.is_miss_loc(y, f, detection_gt)
                    if ids:
                        detection_gt = np.delete(detection_gt, obj=ids, axis=0)
                        matched = True
                        break  
                if matched:
                    self.accumulator[self.video_count]["mil"] += 1
                else:
                    leftover.append(det_pd)  
        matched_dets = leftover.copy()
        return matched_dets, detection_gt        
        
    def eval_association(self, accumulator):
        fp    = accumulator['fp']
        fn    = accumulator['fn']
        lm    = accumulator['lm']
        plm   = accumulator['plm']
        ids   = accumulator['ids']
        idm   = accumulator['idm']
        mil  = accumulator['mil']
        total = fp + fn + lm + plm + ids + idm + mil
        if total==0: 
            return [np.nan]*7
        fp   = fp/total
        fn   = fn/total
        lm   = lm/total
        plm  = plm/total
        ids  = ids/total
        idm  = idm/total
        mil = mil/total
        return (lm, plm, ids, idm, mil, fp, fn)        
                        
    def compute(self, component="ivt", video_id=None, style="coco"):
        if video_id == None: 
            video_id = self.video_count-1 if self.end_call else self.video_count
        # Set component-specific strings and class count
        if component == "ivt":
            hit_str, pos_str, det_str = "hits", "npos", "ndet"
            num_class = self.num_class
        elif component == "i":
            hit_str, pos_str, det_str = "hits_i", "npos_i", "ndet_i"
            num_class = self.num_tool
        elif component == "v":
            hit_str, pos_str, det_str = "hits_v", "npos_v", "ndet_v"
            num_class = self.num_verb
        elif component == "t":
            hit_str, pos_str, det_str = "hits_t", "npos_t", "ndet_t"
            num_class = self.num_target
        else:
            raise ValueError(f"Unknown component: {component}. Must be one of: 'ivt', 'i', 'v', 't'")
        # decide on accumulator for framewise / video wise / current
        if video_id == -1: # global AP
            accumulator = {}
            accumulator[hit_str] = [sum([p[k]for p in [self.accumulator[f][hit_str] for f in self.accumulator] ],[]) for k in range(num_class)]
            accumulator[pos_str] = list(np.sum(np.stack([self.accumulator[f][pos_str] for f in self.accumulator]), axis=0))
            accumulator[det_str] = list(np.sum(np.stack([self.accumulator[f][det_str] for f in self.accumulator]), axis=0))
            # Merge confidence information for global AP calculation
            if component == "ivt":
                conf_str = "conf"
            elif component == "i":
                conf_str = "conf_i"
            elif component == "v":
                conf_str = "conf_v"
            elif component == "t":
                conf_str = "conf_t"
            else:
                conf_str = "conf"
            accumulator[conf_str] = [sum([p[k]for p in [self.accumulator[f][conf_str] for f in self.accumulator] ],[]) for k in range(num_class)]
            # Merge association metrics for global AP calculation
            for key in ["fp", "fn", "lm", "plm", "ids", "idm", "mil"]:
                accumulator[key] = sum([self.accumulator[f][key] for f in self.accumulator])
        else:
             accumulator = self.accumulator[video_id]        
        
        # COCO-style modification: Only consider classes with ground truth
        # no gt but has detections (all FP)
        if style == "coco":
            classes_with_gt = [i for i in range(num_class) if accumulator[pos_str][i] > 0]
        else:
            classes_with_gt = list(range(num_class))
        
        classwise_ap = [np.nan] * num_class
        classwise_rec = [np.nan] * num_class  
        classwise_prec = [np.nan] * num_class
        classwise_f1 = [np.nan] * num_class  # Add F1 array
        
        # Store precision and recall curves for F1 calculation (like ultralytics)
        p_curves = []
        r_curves = []
        f1_curves = []
        
        # computation
        # npos: TP + FN = all GT (except bg); ndet: TP + FP = all preds (except bg); hits: TP
        for class_id in classes_with_gt:
            hits, npos, ndet = accumulator[hit_str][class_id], accumulator[pos_str][class_id], accumulator[det_str][class_id]
            
            # Get corresponding confidence
            if component == "ivt":
                conf_str = "conf"
            elif component == "i":
                conf_str = "conf_i"
            elif component == "v":
                conf_str = "conf_v"
            elif component == "t":
                conf_str = "conf_t"
            else:
                conf_str = "conf"
            conf = accumulator[conf_str][class_id] if conf_str in accumulator else []
            
            if npos + ndet == 0: # no gt instance and no detection for the class
                classwise_ap[class_id] = np.nan
                classwise_rec[class_id] = np.nan
                classwise_prec[class_id] = np.nan
                p_curves.append(np.zeros(1000))
                r_curves.append(np.zeros(1000))
                f1_curves.append(np.zeros(1000))
            elif npos>0 and len(hits)==0: # no detections but there are gt instances for the class
                classwise_ap[class_id] = 0.0
                classwise_rec[class_id] = 0.0
                classwise_prec[class_id] = 0.0
                p_curves.append(np.zeros(1000))
                r_curves.append(np.zeros(1000))
                f1_curves.append(np.zeros(1000))
            else:
                # Global sorting: sort hits by confidence in descending order
                if len(conf) > 0 and len(conf) == len(hits):
                    # Sort by confidence
                    sorted_indices = np.argsort(-np.array(conf))
                    hits = np.array(hits)[sorted_indices]
                
                hits = np.cumsum(hits) # TPs
                rec  = hits / npos if npos else 0.0
                prec = hits / (np.array(range(len(hits)), dtype=float) + 1.0)
                
                if style == "coco":
                    # COCO-style 101-point interpolation AP calculation (like ultralytics)
                    # Append sentinel values to beginning and end
                    mrec = np.concatenate(([0.0], rec, [1.0]))
                    mpre = np.concatenate(([1.0], prec, [0.0]))
                    
                    # Compute the precision envelope
                    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
                    
                    # Integrate area under curve using 101-point interpolation
                    x = np.linspace(0, 1, 101)  # 101-point interp (COCO standard)
                    ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate using trapezoidal rule
                else:
                    # Original 11-point interpolation
                    ap = 0.0
                    for i in range(11):
                        mask = rec >= (i / 10.0)
                        if np.sum(mask) > 0:
                            ap += np.max(prec[mask]) / 11.0
                
                classwise_ap[class_id] = ap
                
                
                # Create precision and recall curves for 1000 confidence thresholds
                x = np.linspace(0, 1, 1000)
                if len(conf) > 0:
                    p_curve = np.interp(-x, -np.array(conf), prec, left=1)
                    r_curve = np.interp(-x, -np.array(conf), rec, left=0)
                else:
                    p_curve = np.zeros(1000)
                    r_curve = np.zeros(1000)
                
                p_curves.append(p_curve)
                r_curves.append(r_curve)
                f1_curves.append(2 * p_curve * r_curve / (p_curve + r_curve + 1e-16))
        
        # Find optimal F1 threshold globally 
        if len(f1_curves) > 0:
            f1_curves = np.array(f1_curves)
            p_curves = np.array(p_curves)  
            r_curves = np.array(r_curves)
            
            # Smooth the mean F1 curve and find max 
            mean_f1 = np.nanmean(f1_curves, axis=0)
            # Simple smoothing (moving average)
            kernel_size = int(len(mean_f1) * 0.1)  # 10% smoothing
            if kernel_size > 1:
                kernel = np.ones(kernel_size) / kernel_size
                mean_f1_smooth = np.convolve(mean_f1, kernel, mode='same')
            else:
                mean_f1_smooth = mean_f1
                
            best_f1_idx = np.argmax(mean_f1_smooth)
            
            # Set precision and recall at optimal F1 threshold for all classes
            for i, class_id in enumerate(classes_with_gt):
                if i < len(p_curves) and i < len(r_curves):
                    classwise_prec[class_id] = p_curves[i, best_f1_idx]
                    classwise_rec[class_id] = r_curves[i, best_f1_idx]
                    classwise_f1[class_id] = f1_curves[i, best_f1_idx] 
                else:
                    classwise_prec[class_id] = 0.0
                    classwise_rec[class_id] = 0.0
                    classwise_f1[class_id] = 0.0
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            assoc_results = self.eval_association(accumulator)
            # Both styles can use nanmean since we filtered at the beginning
            mAP = np.nanmean(classwise_ap)
            mRec = np.nanmean(classwise_rec)
            mPrec = np.nanmean(classwise_prec)
            
            return (classwise_ap, mAP), \
                    (classwise_rec, mRec), \
                    (classwise_prec, mPrec), \
                    (classwise_f1, np.nanmean(classwise_f1)), \
                    assoc_results
    
    def compute_5095(self, component="ivt", video_id=None, style="coco"):
        if not self.enable_map5095:
            return None
            
        if video_id == None: 
            video_id = self.video_count-1 if self.end_call else self.video_count
        # Set component-specific strings and class count
        if component == "ivt":
            hit_str, pos_str, det_str = "hits_5095", "npos", "ndet"
            num_class = self.num_class
        elif component == "i":
            hit_str, pos_str, det_str = "hits_i_5095", "npos_i", "ndet_i"
            num_class = self.num_tool
        elif component == "v":
            hit_str, pos_str, det_str = "hits_v_5095", "npos_v", "ndet_v"
            num_class = self.num_verb
        elif component == "t":
            hit_str, pos_str, det_str = "hits_t_5095", "npos_t", "ndet_t"
            num_class = self.num_target
        else:
            raise ValueError(f"Unknown component: {component}. Must be one of: 'ivt', 'i', 'v', 't'")
        
        # decide on accumulator for framewise / video wise / current
        if video_id == -1: # global AP
            accumulator = {}
            accumulator[hit_str] = [[sum([p[thresh_idx][k] for p in [self.accumulator[f][hit_str] for f in self.accumulator] ],[]) 
                                   for k in range(num_class)] for thresh_idx in range(len(self.iou_thresholds))]
            accumulator[pos_str] = list(np.sum(np.stack([self.accumulator[f][pos_str] for f in self.accumulator]), axis=0))       
            accumulator[det_str] = list(np.sum(np.stack([self.accumulator[f][det_str] for f in self.accumulator]), axis=0))
            # Merge confidence information for global AP calculation
            if component == "ivt":
                conf_str = "conf"
            elif component == "i":
                conf_str = "conf_i"
            elif component == "v":
                conf_str = "conf_v"
            elif component == "t":
                conf_str = "conf_t"
            else:
                conf_str = "conf"
            accumulator[conf_str] = [sum([p[k]for p in [self.accumulator[f][conf_str] for f in self.accumulator] ],[]) for k in range(num_class)]
        else:
             accumulator = self.accumulator[video_id]
        
        # COCO-style modification: Only consider classes with ground truth
        if style == "coco":
            classes_with_gt = [i for i in range(num_class) if accumulator[pos_str][i] > 0]
        else:
            classes_with_gt = list(range(num_class))
        
        classwise_ap_5095 = []
        classwise_ar_5095 = []  
        
        for thresh_idx in range(len(self.iou_thresholds)):
            classwise_ap = [np.nan] * num_class
            classwise_ar = [np.nan] * num_class  
            
            # computation for each IoU threshold
            for class_id in classes_with_gt:
                hits, npos, ndet = accumulator[hit_str][thresh_idx][class_id], accumulator[pos_str][class_id], accumulator[det_str][class_id]
                
                # Get corresponding confidence
                if component == "ivt":
                    conf_str = "conf"
                elif component == "i":
                    conf_str = "conf_i"
                elif component == "v":
                    conf_str = "conf_v"
                elif component == "t":
                    conf_str = "conf_t"
                else:
                    conf_str = "conf"
                conf = accumulator[conf_str][class_id] if conf_str in accumulator else []
                
                if npos + ndet == 0: # no gt instance and no detection for the class
                    classwise_ap[class_id] = np.nan
                    classwise_ar[class_id] = np.nan
                elif npos>0 and len(hits)==0: # no detections but there are gt instances for the class
                    classwise_ap[class_id] = 0.0
                    classwise_ar[class_id] = 0.0
                else:
                    # Global sorting: sort hits by confidence in descending order
                    if len(conf) > 0 and len(conf) == len(hits):
                        # Sort by confidence
                        sorted_indices = np.argsort(-np.array(conf))
                        hits = np.array(hits)[sorted_indices]
                    
                    hits = np.cumsum(hits) # TPs
                    rec  = hits / npos if npos else 0.0
                    prec = hits / (np.array(range(len(hits)), dtype=float) + 1.0)
                    
                    if style == "coco":
                        # COCO-style 101-point interpolation AP calculation
                        # Append sentinel values to beginning and end
                        mrec = np.concatenate(([0.0], rec, [1.0]))
                        mpre = np.concatenate(([1.0], prec, [0.0]))
                        
                        # Compute the precision envelope
                        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
                        
                        # Integrate area under curve using 101-point interpolation
                        x = np.linspace(0, 1, 101)  # 101-point interp (COCO standard)
                        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate using trapezoidal rule
                    else:
                        # Original 11-point interpolation
                        ap = 0.0
                        for i in range(11):
                            mask = rec >= (i / 10.0)
                            if np.sum(mask) > 0:
                                ap += np.max(prec[mask]) / 11.0
                    
                    classwise_ap[class_id] = ap
                    classwise_ar[class_id] = hits[-1] / npos if npos > 0 else 0.0
            
            classwise_ap_5095.append(classwise_ap)
            classwise_ar_5095.append(classwise_ar)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            classwise_ap_5095 = np.array(classwise_ap_5095)  # Shape: (num_thresholds, num_classes)
            classwise_ar_5095 = np.array(classwise_ar_5095)  # Shape: (num_thresholds, num_classes)
            classwise_map_5095 = np.nanmean(classwise_ap_5095, axis=0)  
            classwise_mar_5095 = np.nanmean(classwise_ar_5095, axis=0)  
            mAP_5095 = np.nanmean(classwise_map_5095)  
            mAR_5095 = np.nanmean(classwise_mar_5095)  
            
        return classwise_map_5095, mAP_5095, classwise_mar_5095, mAR_5095
    
    def compute_video_AP(self, component="ivt", style="coco"):
        classwise_ap    = []
        classwise_rec   = []
        classwise_prec  = []
        classwise_f1    = []  
        video_lm, video_plm, video_ids, video_idm, video_mil, video_fp, video_fn = [],[],[],[],[],[],[]
        for j in range(self.video_count):
            video_id = j+1
            (ap, _), (rec, _), (prec, _), (f1, _), asc = self.compute(component=component, video_id=video_id, style=style)            
            classwise_ap.append(ap)
            classwise_rec.append(rec)
            classwise_prec.append(prec)
            classwise_f1.append(f1)  
            video_lm.append(asc[0])  # association metrics starts
            video_plm.append(asc[1])
            video_ids.append(asc[2])
            video_idm.append(asc[3])
            video_mil.append(asc[4])
            video_fp.append(asc[5])
            video_fn.append(asc[6])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            classwise_ap    = np.nanmean(np.stack(classwise_ap, axis=0), axis=0)
            classwise_rec   = np.nanmean(np.stack(classwise_rec, axis=0), axis=0)
            classwise_prec  = np.nanmean(np.stack(classwise_prec, axis=0), axis=0)        
            classwise_f1    = np.nanmean(np.stack(classwise_f1, axis=0), axis=0)  
            mAP             = np.nanmean(classwise_ap)
            mRec            = np.nanmean(classwise_rec)
            mPrec           = np.nanmean(classwise_prec) 
            mF1             = np.nanmean(classwise_f1)  
            lm              = np.nanmean(video_lm)  # association metrics starts
            plm             = np.nanmean(video_plm)
            ids             = np.nanmean(video_ids)
            idm             = np.nanmean(video_idm)
            mil            = np.nanmean(video_mil)
            fp              = np.nanmean(video_fp)
            fn              = np.nanmean(video_fn)            
        result = {
                "AP":classwise_ap, "mAP":mAP, "Rec":classwise_rec, "mRec":mRec, "Pre":classwise_prec, "mPre":mPrec, "F1":classwise_f1, "mF1":mF1,
                "lm":lm, "plm":plm, "ids":ids, "idm":idm, "mil":mil, "fp":fp, "fn":fn,
               }
        
        if self.enable_map5095:
            ap_5095_result = self.compute_video_AP_5095(component=component, style=style)
            if ap_5095_result is not None:
                classwise_ap_5095, mAP_5095, classwise_ar_5095, mAR_5095 = ap_5095_result
                result["AP_5095"] = classwise_ap_5095
                result["mAP_5095"] = mAP_5095
                result["AR_5095"] = classwise_ar_5095
                result["mAR_5095"] = mAR_5095
        
        return result
    
    def compute_video_AP_5095(self, component="ivt", style="coco"):
        if not self.enable_map5095:
            return None
            
        classwise_ap_5095 = []
        classwise_ar_5095 = []
        for j in range(self.video_count):
            video_id = j+1
            ap_5095_result = self.compute_5095(component=component, video_id=video_id, style=style)
            if ap_5095_result is not None:
                classwise_map_5095, mAP_5095, classwise_mar_5095, mAR_5095 = ap_5095_result
                classwise_ap_5095.append(classwise_map_5095)
                classwise_ar_5095.append(classwise_mar_5095)
        
        if classwise_ap_5095:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                classwise_ap_5095 = np.nanmean(np.stack(classwise_ap_5095, axis=0), axis=0)
                classwise_ar_5095 = np.nanmean(np.stack(classwise_ar_5095, axis=0), axis=0)
                mAP_5095 = np.nanmean(classwise_ap_5095)
                mAR_5095 = np.nanmean(classwise_ar_5095)
            return classwise_ap_5095, mAP_5095, classwise_ar_5095, mAR_5095
        return None
    
    def compute_AP(self, component="ivt", style="coco"):
        a,r,p,f1, asc = self.compute(component=component, video_id=None, style=style)
        (lm, plm, ids, idm, mil, fp, fn) = asc
        result = {"AP":a[0], "mAP":a[1], "Rec":r[0], "mRec":r[1], "Pre":p[0], "mPre":p[1], "F1":f1[0], "mF1":f1[1],
                "lm":lm, "plm":plm, "ids":ids, "idm":idm, "mil":mil, "fp":fp, "fn":fn,}
        
        # Add mAP50-95 if enabled
        if self.enable_map5095:
            ap_5095_result = self.compute_5095(component=component, video_id=None, style=style)
            if ap_5095_result is not None:
                classwise_map_5095, mAP_5095, classwise_mar_5095, mAR_5095 = ap_5095_result
                result["AP_5095"] = classwise_map_5095
                result["mAP_5095"] = mAP_5095
                result["AR_5095"] = classwise_mar_5095
                result["mAR_5095"] = mAR_5095
        
        return result
        
    def compute_global_AP(self, component="ivt", style="coco"):
        a,r,p,f1, asc =  self.compute(component=component, video_id=-1, style=style)
        (lm, plm, ids, idm, mil, fp, fn) = asc
        result = {"AP":a[0], "mAP":a[1], "Rec":r[0], "mRec":r[1], "Pre":p[0], "mPre":p[1], "F1":f1[0], "mF1":f1[1],
                "lm":lm, "plm":plm, "ids":ids, "idm":idm, "mil":mil, "fp":fp, "fn":fn,}
        
        if self.enable_map5095:
            ap_5095_result = self.compute_5095(component=component, video_id=-1, style=style)
            if ap_5095_result is not None:
                classwise_map_5095, mAP_5095, classwise_mar_5095, mAR_5095 = ap_5095_result
                result["AP_5095"] = classwise_map_5095
                result["mAP_5095"] = mAP_5095
                result["AR_5095"] = classwise_mar_5095
                result["mAR_5095"] = mAR_5095
        
        return result
# %%
