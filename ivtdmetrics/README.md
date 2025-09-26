# IVTDMetrics

An extended metrics package for surgical triplet detection.

## 📜 Overview

**IVTDMetrics** This package provides more accurate evaluation metrics for surgical triplet detection.

## ✨ Features

- **Recognition Metrics**: Same as ivtmetrics, compute Average Precision (AP) for multi-label classification tasks
- **Detection Metrics**: Evaluate object detection performance with IoU-based matching

<a id="ivtd_installation"></a>
## 🚀 Installation

### From source

```bash
cd ivtdmetrics
pip install -e ./
```

### 📋 Requirements

- Python >= 3.6
- numpy >= 1.21

## ⚡ Quick Start

#### 📥 Input Format

Detection data should be provided as lists of detections per frame:
- **List format (7 elements)**: `[[tripletID, toolID, Confidences, x, y, w, h], [tripletID, toolID, Confidences, x, y, w, h], ...]`
- **List format (9 elements)**: `[[tripletID, toolID, Confidences, verbID, targetID, x, y, w, h], [tripletID, toolID, Confidences, verbID, targetID, x, y, w, h], ...]`

**Note**: The system automatically detects the input format. Use 7-element format for IVT+I evaluation only, or 9-element format for full IVT+I+V+T evaluation.

#### 🧩 Supported Components

- `"ivt"`: Instrument-Verb-Target (full triplet)
- `"i"`: Instrument only
- `"v"`: Verb only
- `"t"`: Target only

<a id="ivtd_example"></a>
#### 🎬 Multi-Video Detection Evaluation Example

```python
from ivtdmetrics.detection import Detection
import numpy as np

# Test data
video1_targets = [
    [[0, 1, 1.0, 0.1, 0.1, 0.2, 0.2], [1, 2, 1.0, 0.5, 0.5, 0.2, 0.2]],
    [[2, 3, 1.0, 0.3, 0.3, 0.2, 0.2]]
]

video1_predictions = [
    [[0, 1, 0.9, 0.1, 0.1, 0.2, 0.2], [1, 2, 0.8, 0.7, 0.7, 0.1, 0.1], [3, 4, 0.7, 0.8, 0.8, 0.1, 0.1]],
    []
]

video2_targets = [
    [[2, 5, 1.0, 0.1, 0.1, 0.3, 0.3]],
    [[4, 5, 1.0, 0.2, 0.2, 0.3, 0.3]]
]

video2_predictions = [
    [[2, 5, 0.9, 0.1, 0.1, 0.3, 0.3], [6, 2, 0.5, 0.8, 0.8, 0.1, 0.1]],
    [[4, 5, 0.95, 0.2, 0.2, 0.3, 0.3], [7, 1, 0.4, 0.6, 0.6, 0.1, 0.1]]
]

video3_targets = [
    [[8, 0, 1.0, 0.0, 0.0, 0.3, 0.3], [9, 1, 1.0, 0.4, 0.4, 0.2, 0.2], [10, 2, 1.0, 0.7, 0.7, 0.2, 0.2]],
    [[11, 3, 1.0, 0.1, 0.1, 0.4, 0.4]]
]

video3_predictions = [
    [[8, 0, 0.9, 0.0, 0.0, 0.3, 0.3], [9, 1, 0.8, 0.45, 0.45, 0.15, 0.15], [12, 4, 0.6, 0.8, 0.8, 0.1, 0.1]],
    [[11, 3, 0.85, 0.15, 0.15, 0.35, 0.35], [11, 3, 0.7, 0.2, 0.2, 0.3, 0.3]]
]

all_targets = [video1_targets, video2_targets, video3_targets]
all_predictions = [video1_predictions, video2_predictions, video3_predictions]

detector = Detection(num_class=89, num_tool=7)

for video_idx, (targets, predictions) in enumerate(zip(all_targets, all_predictions)):
    detector.update(targets, predictions, format="list")
    detector.video_end()

# style = "coco" is default setting
video_ap_ivt = detector.compute_video_AP(component="ivt", style="coco")
global_ap_ivt = detector.compute_global_AP(component="ivt", style="coco")
video_ap_i = detector.compute_video_AP(component="i", style="coco")
global_ap_i = detector.compute_global_AP(component="i", style="coco")

print(f"IVT Video-wise mAP@50:    {video_ap_ivt['mAP']:.4f}")
print(f"IVT Global mAP@50:        {global_ap_ivt['mAP']:.4f}")
print(f"I Video-wise mAP@50:      {video_ap_i['mAP']:.4f}")
print(f"I Global mAP@50:          {global_ap_i['mAP']:.4f}") 
print(f"IVT Video-wise mAP@5095:  {video_ap_ivt['mAP_5095']:.4f}")
print(f"IVT Global mAP@5095:      {global_ap_ivt['mAP_5095']:.4f}")
print(f"I Video-wise mAP@5095:    {video_ap_i['mAP_5095']:.4f}")
print(f"I Global mAP@5095:        {global_ap_i['mAP_5095']:.4f}") 
```
### 🧮 Calculation ###

**Note**: The Detection module does not support component disentanglement features such as `iv` (instrument-verb) and `it` (instrument-target) pair evaluations, as their practical significance may be limited for surgical triplet detection tasks.

```python
# Only calculate mAP@50
detector = Detection(num_class=89, num_tool=7, enable_map5095=False)

# Calculate both mAP@50 and mAP@50_95
detector = Detection(num_class=89, num_tool=7)

....
....

# Use ultralytics AP calculation
results = detector.compute_video_AP(style="coco") # default

# Use orginal AP calculation
results = detector.compute_video_AP(style="11point")

....
....

# Other metrics (based on optimal global F1 threshold)
print(f"IVT Video-wise Rec: {video_ap_ivt['mRec']:.4f}") 
print(f"IVT Video-wise Pre: {video_ap_ivt['mPre']:.4f}")
print(f"IVT Video-wise F1:  {video_ap_ivt['mF1']:.4f}") 
print(f"IVT Video-wise AR:  {video_ap_ivt['mAR_5095']:.4f}") 
# LM, PLM.... are based on image-level conf ranking
```

#### 💻 Multi-Component Evaluation Example

To evaluate all components (IVT, I, V, T) simultaneously, use the 9-element format and compute metrics for each component:

```python
from ivtdmetrics.detection import Detection

detector = Detection(
    num_class=89,    
    num_tool=7,      
    num_verb=10,     
    num_target=10,  
    threshold=0.5
)

# Process your data (targets and predictions should use 9-element format)
# Format: [tripletID, toolID, confidence, verbID, targetID, x, y, w, h]
for targets, predictions in zip(all_targets, all_predictions):
    detector.update(targets, predictions, format="list")
    detector.video_end()

components = ["ivt", "i", "v", "t"]
results = {}

for component in components:
    results[f"video_{component}"] = detector.compute_video_AP(component=component, style="coco")
    results[f"global_{component}"] = detector.compute_global_AP(component=component, style="coco")

# Display comprehensive results
print("=" * 80)
print(f"{'Metric Type':<12} {'Component':<9} {'mAP@50':<8} {'mAP@50-95':<10} {'Precision':<10} {'Recall':<8} {'F1':<8} {'AR':<10}")
print("-" * 80)

for component in components:
    comp_name = component.upper()
    video_result = results[f"video_{component}"]
    global_result = results[f"global_{component}"]

    print(f"{'Video-wise':<12} {comp_name:<9} {video_result.get('mAP', 0):<8.4f} {video_result.get('mAP_5095', 0):<10.4f} {video_result.get('mPre', 0):<10.4f} {video_result.get('mRec', 0):<8.4f} {video_result.get('mF1', 0):<8.4f} {video_result.get('mAR_5095', 0):<10.4f}")
    print(f"{'Global':<12} {comp_name:<9} {global_result.get('mAP', 0):<8.4f} {global_result.get('mAP_5095', 0):<10.4f} {global_result.get('mPre', 0):<10.4f} {global_result.get('mRec', 0):<8.4f} {global_result.get('mF1', 0):<8.4f} {global_result.get('mAR_5095', 0):<10.4f}")

print("-" * 80)
```

## 🌟 Enhancements
**Global Confidence Ranking**: Implemented global confidence score ranking for mAP calculation instead of image-level ranking

**101-Point Interpolation**: Adopted 101-point interpolation for mAP calculation

**Pseudo-Detection Handling**: Fixed calculation errors when handling pseudo-detections for scenarios where ground truth lacks certain classes but predictions include them.

**Precision, Recall, and F1 Evaluation**: Added metrics based on a single optimal confidence threshold determined by maximizing F1 score. (this Recall differs from the AR calculation method)

**mAP50-95 Evaluation**: Added mAP50-95 result calculation.

**Support Component Verb and Target**: Added component Verb and Target calculation.

**AR@max_det Evaluation** (not recommended): Added Average Recall calculation. In the evaluation of **ultralytics**, mAP is often computed after filtering (NMS) predictions, whereas in the **COCO eval API**, the inputs are taken before filtering (score), from which AR@100 and similar metrics are then calculated. This inconsistency makes alignment difficult, so we recommend using **Recall** and **F1-score** instead of AR.

**Bug Fixes**: Fixed various bugs likse list2stack function.

## 🙏 Acknowledgments

This work is partily based on [ultralytics](https://github.com/ultralytics/yolov5), [ivtmetrics](https://github.com/CAMMA-public/ivtmetrics).

## 📞 Contact

For questions or issues, please open an issue on GitHub or contact the maintainer. 
