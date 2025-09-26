import os
import json
import yaml
import re
from pathlib import Path

def load_class_names(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data.get('names', [])

def parse_filename(filename):
    # support three formats:
    # 1. 5304_v1_unlabeled.txt -> (5304, v1_unlabeled)
    # 2. psiv1_302300.txt -> (302300, psiv1)
    # 3. psiv2_123124_2.txt -> (123124_2, psiv2)
    match1 = re.match(r'(\d+)_([^\.]+)\.txt', filename)
    match2 = re.match(r'([a-z]+\d+)_(\d+)\.txt', filename)
    match3 = re.match(r'([a-z]+\d+)_(\d+_\d+)\.txt', filename)
    
    if match1:
        base_name = match1.group(1)  # e.g. "5304"
        folder_name = match1.group(2)  # e.g. "v1_unlabeled"
        return base_name, folder_name
    elif match3:  # note: match3 (more specific pattern) should be matched first, then match2
        folder_name = match3.group(1)  # e.g. "psiv2"
        base_name = match3.group(2)  # e.g. "123124_2"
        return base_name, folder_name
    elif match2:
        folder_name = match2.group(1)  # e.g. "psiv1"
        base_name = match2.group(2)  # e.g. "302300"
        return base_name, folder_name
    else:
        return None, None

def yolo_to_labelme(yolo_bbox, img_width, img_height, class_id, class_names):
    # convert YOLO format bounding box to LabelMe format
    x_center, y_center, width, height = map(float, yolo_bbox)
    
    x_center = x_center * img_width
    y_center = y_center * img_height
    width = width * img_width
    height = height * img_height
    
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_width, x2)
    y2 = min(img_height, y2)
    
    class_name = class_names[int(class_id)] if int(class_id) < len(class_names) else f"class_{class_id}"
    
    labelme_annotation = {
        "label": class_name,
        "points": [[x1, y1], [x2, y2]],
        "group_id": None,
        "description": "",
        "shape_type": "rectangle",
        "flags": {},
        "attributes": {
            "Action": "null",
            "Target": "null"
        },
        "mask": None
    }
    
    return labelme_annotation

def create_labelme_json(image_filename, img_width, img_height, shapes=None):
    if shapes is None:
        shapes = []
    
    labelme_json = {
        "version": "5.6.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_filename,
        "imageData": None,
        "imageHeight": img_height,
        "imageWidth": img_width,
        "text": "",
        "description": ""
    }
    
    return labelme_json

def process_yolo_annotations(yolo_dir, output_dir, dataset_yaml, img_format="jpg"):
    img_width, img_height = 640, 640

    class_names = load_class_names(dataset_yaml)
    print(f"loaded class names: {class_names}")
    print(f"using image format: .{img_format}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    yolo_files = [f for f in os.listdir(yolo_dir) if f.endswith('.txt')]
    print(f"found {len(yolo_files)} YOLO annotation files")
    
    for i, yolo_file in enumerate(yolo_files):
        base_name, folder_name = parse_filename(yolo_file)
        if not base_name or not folder_name:
            print(f"warning: cannot parse file name {yolo_file}, skip")
            continue
        
        image_filename = f"{base_name}.{img_format}"
        yolo_path = os.path.join(yolo_dir, yolo_file)
        shapes = []
        with open(yolo_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    print(f"warning: annotation format is incorrect {line} in file {yolo_file}")
                    continue
                
                class_id = parts[0]
                yolo_bbox = parts[1:5]
                confidence = parts[5] if len(parts) > 5 else None
                
                labelme_annotation = yolo_to_labelme(yolo_bbox, img_width, img_height, class_id, class_names)
                
                if confidence:
                    labelme_annotation["description"] = f"confidence: {confidence}"
                
                shapes.append(labelme_annotation)
        
        labelme_json = create_labelme_json(image_filename, img_width, img_height, shapes)
        output_folder = os.path.join(output_dir, folder_name)
        os.makedirs(output_folder, exist_ok=True)
        
        output_path = os.path.join(output_folder, f"{base_name}.json")
        with open(output_path, 'w') as f:
            json.dump(labelme_json, f, indent=2)
        
        if (i + 1) % 100 == 0 or (i + 1) == len(yolo_files):
            print(f"processing: {i + 1}/{len(yolo_files)}")
    
    print(f"finished! output directory: {output_dir}")

def main():
    # all parameters are hardcoded
    yolo_dir = "/prostate/framework_yolo/tool_only/test_test/labels"
    output_dir = os.path.join(os.path.dirname(yolo_dir), "labels_labelme")
    dataset_yaml = '/prostate_track/ProstaTD_20fps/dataset_yolo_tool/split/dataset_tool.yaml'
    img_format = "jpg"
    
    process_yolo_annotations(yolo_dir, output_dir, dataset_yaml, img_format)

if __name__ == "__main__":
    main() 