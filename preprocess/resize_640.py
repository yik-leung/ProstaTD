import os
import json
from glob import glob
from PIL import Image, ImageEnhance
import shutil
import concurrent.futures
import threading
import math
from tqdm import tqdm

print_lock = threading.Lock()

def high_quality_resize(img, target_size):
    original_width, original_height = img.size
    target_width, target_height = target_size
    
    if max(original_width, original_height) < 1.5 * max(target_width, target_height):
        return img.resize(target_size, Image.Resampling.LANCZOS)
    
    steps = max(1, int(math.log(max(original_width/target_width, 
                                     original_height/target_height), 1.25)))
    
    current_img = img
    for i in range(steps):
        ratio = 1 - (i + 1) / steps
        current_w = int(original_width * ratio + target_width * (1 - ratio))
        current_h = int(original_height * ratio + target_height * (1 - ratio))
        
        if i > 0: 
            enhancer = ImageEnhance.Sharpness(current_img)
            current_img = enhancer.enhance(1.05)
            
        current_img = current_img.resize((current_w, current_h), Image.Resampling.LANCZOS)
    
    enhancer = ImageEnhance.Sharpness(current_img)
    current_img = enhancer.enhance(1.2)
    
    return current_img

def process_single_image(args):
    root, img_file, curr_output_dir, target_size = args
    try:
        img_path = os.path.join(root, img_file)
        base_name = os.path.splitext(img_file)[0] 
        ext = os.path.splitext(img_file)[1].lower() 
        json_file = base_name + '.json'
        json_path = os.path.join(root, json_file)
        
        with Image.open(img_path) as img:
            has_alpha = 'A' in img.getbands() if ext == '.png' else False
            original_mode = img.mode
            
            if img.mode != 'RGB' and not has_alpha:
                img = img.convert('RGB')
            elif has_alpha:
                img = img.convert('RGBA')
            
            original_width, original_height = img.size
            target_width, target_height = target_size
            
            square_size = max(original_width, original_height)
            
            if has_alpha:
                square_img = Image.new('RGBA', (square_size, square_size), (0, 0, 0, 0))
            else:
                square_img = Image.new('RGB', (square_size, square_size), (0, 0, 0))

            paste_x = (square_size - original_width) // 2
            paste_y = (square_size - original_height) // 2
            
            square_img.paste(img, (paste_x, paste_y))
            
            final_img = high_quality_resize(square_img, target_size)
            
            output_img_path = os.path.join(curr_output_dir, img_file)
            
            if ext == '.png':
                compression = 6  # 0-9, 9 is the highest compression rate but very slow
                final_img.save(output_img_path, 'PNG', compress_level=compression)
            else:
                final_img.save(output_img_path, 'JPEG', 
                               quality=95,
                               optimize=True,
                               subsampling=0)
        
        scale_factor = float(target_width) / float(square_size)
        
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            data["imageWidth"] = target_width
            data["imageHeight"] = target_height
            
            transform_info = {
                "originalWidth": original_width,
                "originalHeight": original_height,
                "squareSize": square_size,
                "pasteX": paste_x,
                "pasteY": paste_y,
                "scaleFactor": scale_factor,
                "targetWidth": target_width,
                "targetHeight": target_height
            }
            
            data["transformInfo"] = transform_info
            
            for shape in data["shapes"]:
                for i, point in enumerate(shape["points"]):
                    original_point = point.copy()
                    
                    x = (point[0] + paste_x) * scale_factor
                    y = (point[1] + paste_y) * scale_factor
                    
                    x = max(0, min(x, target_width))
                    y = max(0, min(y, target_height))
                    
                    shape["points"][i] = [x, y]
                    
                    if "originalPoints" not in shape:
                        shape["originalPoints"] = []
                    shape["originalPoints"].append(original_point)
        else:
            transform_info = {
                "originalWidth": original_width,
                "originalHeight": original_height,
                "squareSize": square_size,
                "pasteX": paste_x,
                "pasteY": paste_y,
                "scaleFactor": scale_factor,
                "targetWidth": target_width,
                "targetHeight": target_height
            }
            
            data = {
                "version": "4.5.6",
                "flags": {},
                "shapes": [],
                "imagePath": img_file,
                "imageData": None,
                "imageHeight": target_height,
                "imageWidth": target_width,
                "transformInfo": transform_info,
                "noAnnotation": True 
            }
        
        output_json_path = os.path.join(curr_output_dir, json_file)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        return True
    
    except Exception as e:
        with print_lock:
            print(f"  error processing file {img_file}: {str(e)}")
        return False

def restore_single_json(args):
    root, json_file, curr_output_dir = args
    try:
        json_path = os.path.join(root, json_file)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if "transformInfo" in data:
            transform_info = data["transformInfo"]
            original_width = transform_info["originalWidth"]
            original_height = transform_info["originalHeight"]
            paste_x = transform_info["pasteX"]
            paste_y = transform_info["pasteY"]
            scale_factor = transform_info["scaleFactor"]
        elif all(key in data for key in ["originalWidth", "originalHeight", "squareSize", "pasteX", "pasteY", "scaleFactor"]):
            original_width = data["originalWidth"]
            original_height = data["originalHeight"]
            paste_x = data["pasteX"]
            paste_y = data["pasteY"]
            scale_factor = data["scaleFactor"]
        else:
            with print_lock:
                print(f"Warning: {json_file} missing transformation information, cannot restore")
            return False
        
        data["imageWidth"] = original_width
        data["imageHeight"] = original_height
        
        for shape in data["shapes"]:
            if "originalPoints" in shape:
                for i, original_point in enumerate(shape["originalPoints"]):
                    if i < len(shape["points"]):
                        shape["points"][i] = original_point
                del shape["originalPoints"]
            else:
                for i, point in enumerate(shape["points"]):
                    x = float(point[0]) / float(scale_factor) - float(paste_x)
                    y = float(point[1]) / float(scale_factor) - float(paste_y)
                    
                    x = max(0, min(x, original_width))
                    y = max(0, min(y, original_height))
                    
                    shape["points"][i] = [x, y]
        
        if ("noAnnotation" not in data or not data["noAnnotation"]) and "transformInfo" in data:
            del data["transformInfo"]
        
        output_json_path = os.path.join(curr_output_dir, json_file)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        return True
    
    except Exception as e:
        with print_lock:
            print(f"  error processing JSON file {json_file}: {str(e)}")
        return False

def resize_image_and_annotation(input_dir, output_dir, target_size=(640, 640), num_workers=16):
    os.makedirs(output_dir, exist_ok=True)
    tasks = []

    for root, dirs, files in os.walk(input_dir):
        rel_path = os.path.relpath(root, input_dir)
        curr_output_dir = os.path.join(output_dir, rel_path)
        os.makedirs(curr_output_dir, exist_ok=True)
        
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if image_files: 
            print(f"found directory: {rel_path}, containing {len(image_files)} images")
            for img_file in image_files:
                tasks.append((root, img_file, curr_output_dir, target_size))
    
    total_images = len(tasks)
    print(f"found {total_images} images to process")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_single_image, task) for task in tasks]
        
        successful = 0
        with tqdm(total=total_images, desc="processing progress") as pbar:
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    successful += 1
                pbar.update(1)
    
    print(f"finished! {successful}/{total_images} images")

def restore_json_annotations(input_dir, output_dir, num_workers=16):
    os.makedirs(output_dir, exist_ok=True)
    
    tasks = []
    for root, dirs, files in os.walk(input_dir):
        rel_path = os.path.relpath(root, input_dir)
        curr_output_dir = os.path.join(output_dir, rel_path)
        os.makedirs(curr_output_dir, exist_ok=True)

        json_files = [f for f in files if f.lower().endswith('.json')]
        
        if json_files: 
            print(f"found directory: {rel_path}, containing {len(json_files)} JSON files")
            
            for json_file in json_files:
                tasks.append((root, json_file, curr_output_dir))
    
    total_jsons = len(tasks)
    print(f"found {total_jsons} JSON files to restore")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(restore_single_json, task) for task in tasks]
        successful = 0
        with tqdm(total=total_jsons, desc="processing progress") as pbar:
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    successful += 1
                pbar.update(1)
    
    print(f"finished! {successful}/{total_jsons} JSON files")

if __name__ == "__main__":
    # operation mode: resize or restore
    mode = "resize"  # change here: "resize" or "restore"
    
    # === resize mode configuration ===
    resize_input_dir = "./prostate_v2_no"
    resize_output_dir = "./prostate_v2"
    target_size = (640, 640)
    
    # === restore mode configuration ===
    # restore_input_dir = "./runs_EASD/predict/best_test/labels_labelme"      # restore input directory (contains already scaled annotations)
    # restore_output_dir = "./runs_EASD/predict/best_test/labels_restored" # restore output directory (contains restored annotations)

    # general configuration
    num_workers = 32  # number of threads, can be adjusted according to the number of CPU cores
    
    # select input and output directories according to the mode
    if mode == 'resize':
        input_dir = resize_input_dir
        output_dir = resize_output_dir
    else:  # restore mode
        input_dir = restore_input_dir
        output_dir = restore_output_dir
    
    if not os.path.exists(input_dir):
        print(f"error: input directory {os.path.abspath(input_dir)} does not exist!")
        exit(1)
    
    print(f"operation mode: {mode}")
    print(f"input directory: {os.path.abspath(input_dir)}")
    print(f"output directory: {os.path.abspath(output_dir)}")
    print(f"using {num_workers} threads to process")
    
    # execute different operations according to the mode
    if mode == 'resize':
        print(f"target size: {target_size}")
        resize_image_and_annotation(input_dir, output_dir, target_size, num_workers)
    else:  # restore mode
        print("only restore JSON annotations to the original size")
        restore_json_annotations(input_dir, output_dir, num_workers)
    
    if os.path.exists(output_dir):
        files_count = sum([len(files) for _, _, files in os.walk(output_dir)])
        print(f"finished! output directory contains {files_count} files")
    else:
        print("error: output directory not created successfully!")
