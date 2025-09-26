#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
from pathlib import Path
import time
from ultralytics.utils import LOGGER

try:
    from ultralytics.utils import ops
    ORIGINAL_NMS = ops.non_max_suppression
except ImportError as e:
    print(f"Import failed: {e}")
    raise


class TripletToToolMapper:
    def __init__(self, mapping_file):
        if not mapping_file:
            raise ValueError("mapping_file parameter is required")
        self.triplet_to_tool = {}
        self.max_tool_id = 0
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
                        triplet_id = int(parts[0])  
                        tool_id = int(parts[1]) 
                        self.triplet_to_tool[triplet_id] = tool_id
                        self.max_tool_id = max(self.max_tool_id, tool_id)
            
        except Exception as e:
            LOGGER.error(f"Error loading mapping file {mapping_file}: {e}")
            raise
    
    def get_tool_id(self, triplet_id):
        return self.triplet_to_tool.get(int(triplet_id), int(triplet_id))
    
    def convert_classes_to_tools(self, classes_tensor):
        if len(classes_tensor) == 0:
            return classes_tensor
        
        # Convert to numpy for mapping, then back to tensor
        classes_np = classes_tensor.cpu().numpy()
        tool_classes = np.array([self.get_tool_id(cls) for cls in classes_np])
        
        return torch.tensor(tool_classes, device=classes_tensor.device, dtype=classes_tensor.dtype)

_mapper = None

def get_mapper(mapping_file=None):
    global _mapper
    if _mapper is None:
        if not mapping_file:
            raise ValueError("mapping_file is required for first-time mapper initialization")
        _mapper = TripletToToolMapper(mapping_file)
    return _mapper

def set_mapping_file(mapping_file):
    global _mapper
    _mapper = TripletToToolMapper(mapping_file)

def tool_based_non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=0,  # number of classes (optional)
    max_time_img=0.05,
    max_nms=30000,
    max_wh=7680,
    in_place=True,
    rotated=False,
    end2end=False,
    return_idxs=False,
    mapping_file=None,
):
    if end2end or (isinstance(prediction, (list, tuple)) and prediction[0].shape[-1] == 6):
        return ORIGINAL_NMS(
            prediction, conf_thres, iou_thres, classes, agnostic, multi_label, 
            labels, max_det, nc, max_time_img, max_nms, max_wh, in_place, 
            rotated, end2end, return_idxs
        )

    try:
        mapper = get_mapper(mapping_file)
    except ValueError as e:
        LOGGER.error(f"Tool-based NMS requires mapping file: {e}")
        # Fallback to original NMS if no mapping file provided
        return ORIGINAL_NMS(
            prediction, conf_thres, iou_thres, classes, agnostic, multi_label, 
            labels, max_det, nc, max_time_img, max_nms, max_wh, in_place, 
            rotated, end2end, return_idxs
        )

    if not hasattr(tool_based_non_max_suppression, '_logged'):
        tool_based_non_max_suppression._logged = True
    
    import torchvision

    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):  
        prediction = prediction[0]  
    if classes is not None:
        classes = torch.tensor(classes, device=prediction.device)

    bs = prediction.shape[0]  
    nc = nc or (prediction.shape[1] - 4)  
    nm = prediction.shape[1] - nc - 4  
    mi = 4 + nc  
    xc = prediction[:, 4:mi].amax(1) > conf_thres  
    xinds = torch.stack([torch.arange(len(i), device=prediction.device) for i in xc])[..., None]  

    time_limit = 2.0 + max_time_img * bs  
    multi_label &= nc > 1  

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    if not rotated:
        if in_place:
            prediction[..., :4] = ops.xywh2xyxy(prediction[..., :4]) 
        else:
            prediction = torch.cat((ops.xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), dim=-1) 

    t = time.time()
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    keepi = [torch.zeros((0, 1), device=prediction.device)] * bs
    
    for xi, (x, xk) in enumerate(zip(prediction, xinds)):  
        filt = xc[xi]
        x, xk = x[filt], xk[filt]

        if labels and len(labels[xi]) and not rotated:
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
            v[:, :4] = ops.xywh2xyxy(lb[:, 1:5]) 
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0 
            x = torch.cat((x, v), 0)

        if not x.shape[0]:
            continue

        box, cls, mask = x.split((4, nc, nm), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
            xk = xk[i]
        else:  
            conf, j = cls.max(1, keepdim=True)
            filt = conf.view(-1) > conf_thres
            x = torch.cat((box, conf, j.float(), mask), 1)[filt]
            xk = xk[filt]

        if classes is not None:
            filt = (x[:, 5:6] == classes).any(1)
            x, xk = x[filt], xk[filt]

        n = x.shape[0]  
        if not n:  
            continue
        if n > max_nms:  
            filt = x[:, 4].argsort(descending=True)[:max_nms]  
            x, xk = x[filt], xk[filt]

        original_classes = x[:, 5].clone()  
        tool_classes = mapper.convert_classes_to_tools(x[:, 5])  
        
        x[:, 5] = tool_classes.float()

        c = x[:, 5:6] * (0 if agnostic else max_wh)  
        scores = x[:, 4]  
        if rotated:
            boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)  # xywhr
            i = ops.nms_rotated(boxes, scores, iou_thres)
        else:
            boxes = x[:, :4] + c  
            i = torchvision.ops.nms(boxes, scores, iou_thres)  
        i = i[:max_det]

        x[i, 5] = original_classes[i]  
        
        output[xi], keepi[xi] = x[i], xk[i].reshape(-1)
        if (time.time() - t) > time_limit:
            LOGGER.warning(f"NMS time limit {time_limit:.3f}s exceeded")
            break  

    return (output, keepi) if return_idxs else output


def apply_patch(mapping_file=None):
    if mapping_file:
        set_mapping_file(mapping_file)

    import ultralytics.utils.ops as ops_module
    import ultralytics.utils as utils_module
    
    if not hasattr(ops_module, '_original_non_max_suppression'):
        ops_module._original_non_max_suppression = ops_module.non_max_suppression
    
    def patched_nms(*args, **kwargs):
        if 'mapping_file' not in kwargs and mapping_file:
            kwargs['mapping_file'] = mapping_file
        return tool_based_non_max_suppression(*args, **kwargs)

    ops_module.non_max_suppression = patched_nms
    utils_module.ops.non_max_suppression = patched_nms
    
    print("Tool-based NMS patch applied")
    return True


if __name__ == "__main__":
    default_mapping_file = "./triplet_maps_v2.txt"  # hard coded mapping file for testing
    
    mapper = TripletToToolMapper(default_mapping_file)
    print(f"Loaded {len(mapper.triplet_to_tool)} mappings")
    print(f"Max tool ID: {mapper.max_tool_id}")
    
    for triplet_id in [0, 1, 2, 5, 10]:
        tool_id = mapper.get_tool_id(triplet_id)
        print(f"Triplet {triplet_id} -> Tool {tool_id}")
    
    test_classes = torch.tensor([0, 1, 2, 5, 10])
    tool_classes = mapper.convert_classes_to_tools(test_classes)
    print(f"Tensor conversion: {test_classes.tolist()} -> {tool_classes.tolist()}")
        
