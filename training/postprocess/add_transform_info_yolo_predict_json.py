import os
import json
from pathlib import Path

def main():
    # hardcoded input paths
    original_dataset_dir = "/prostate_track/ProstaTD_20fps/dataset_640/"  # original dataset directory
    generated_labelme_dir = "/prostate/framework_yolo/tool_only/test_test/labels_labelme"  # generated LabelMe directory
    
    print(f"extract transformInfo from {original_dataset_dir}")
    print(f"add to JSON files in {generated_labelme_dir}")
    
    # get all subfolders in the generated LabelMe directory
    folder_count = 0
    file_count = 0
    updated_count = 0
    
    for root, dirs, files in os.walk(generated_labelme_dir):
        if root != generated_labelme_dir:
            folder_name = os.path.basename(root)
            folder_count += 1
            original_folder = os.path.join(original_dataset_dir, folder_name)
            
            if not os.path.exists(original_folder):
                print(f"warning: folder {folder_name} not found in original dataset")
                continue
            
            print(f"process folder: {folder_name}")
            
            for file in files:
                if file.endswith('.json'):
                    file_count += 1
                    generated_json_path = os.path.join(root, file)
                    original_json_path = os.path.join(original_folder, file)
                    
                    if not os.path.exists(original_json_path):
                        print(f"   warning: file {file} not found in original dataset")
                        continue
                    
                    try:
                        with open(generated_json_path, 'r') as f:
                            generated_json = json.load(f)
 
                        with open(original_json_path, 'r') as f:
                            original_json = json.load(f)
                        
                        if 'transformInfo' in original_json:
                            transform_info = original_json['transformInfo']
                            generated_json['transformInfo'] = transform_info
                            with open(generated_json_path, 'w') as f:
                                json.dump(generated_json, f, indent=2)
                            
                            updated_count += 1
                        else:
                            print(f"   warning: file {file} does not have transformInfo field")
                    
                    except Exception as e:
                        print(f"   error: error processing file {file}: {e}")
    
    print(f"\nfinished! processed {folder_count} folders, {file_count} files, updated {updated_count} files")

if __name__ == "__main__":
    main() 