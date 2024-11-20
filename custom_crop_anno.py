import os
import cv2
import json
import numpy as np
from pathlib import Path

option = 'val'  # train or val
# Set paths
current_dir = Path.cwd()
parent_directory = current_dir.parent / 'datasets' / 'dataset2_expanded_initial_split'

# Ensure necessary directories exist
if parent_directory.is_dir():
    create_folder = [
        parent_directory / 'crop',
        parent_directory / 'crop' / 'train',
        parent_directory / 'crop' / 'val',
        parent_directory / 'crop' / 'annotations'
    ]
    for folder in create_folder:
        folder_path = Path(folder)
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"Created folder: {folder_path}")
else:
    raise FileNotFoundError("Parent directory doesn't exist")

annotation_path = current_dir / 'annotations' / 'all_dataset2_Mingren_expanded.json'
image_directory = parent_directory / f'{option}'
output_directory = parent_directory / 'crop' / f'{option}'
output_annotation_path = parent_directory / 'crop' / 'annotations' / f'{option}.json'

# Parameters
crop_size = 512  # Output crop size
crop_times = 5  # Number of crops along each dimension

# Ensure output directory exists
output_directory.mkdir(parents=True, exist_ok=True)

# Load the JSON file with annotations
with open(annotation_path, 'r') as file:
    dataset = json.load(file)

# Track statistics
existing_image_files = set(os.listdir(image_directory))
total_images_before_crop = sum(1 for image in dataset['images'] if image['file_name'] in existing_image_files)
total_cropped_images = 0

def crop_image_with_sliding_window(image_path, name, crop_size, crop_times):
    """Crop the image using a sliding window and return the paths of cropped images."""
    image = cv2.imread(str(image_path / name))
    if image is None:
        return []  # Return an empty list if the image cannot be opened

    img_height, img_width, _ = image.shape
    slide_step = (img_width - crop_size) // (crop_times - 1)
    cropped_image_paths = []
    crop_count = 0

    for top in range(0, img_height - crop_size + 1, slide_step):
        for left in range(0, img_width - crop_size + 1, slide_step):
            cropped_image = image[top:top + crop_size, left:left + crop_size]
            crop_name = f"{name.split('.')[0]}_{crop_count + 1}.png"
            cropped_image_paths.append((cropped_image, crop_name, left + crop_size, top + crop_size))
            crop_count += 1

    return cropped_image_paths

def adjust_annotations_for_crop(annotation, crop_right, crop_bottom, crop_size):
    """Adjust segmentation and bbox for the cropped image."""
    crop_left = crop_right - crop_size
    crop_top = crop_bottom - crop_size
    new_segmentation = []

    for segment in annotation['segmentation']:
        adjusted_segment = []
        for i in range(0, len(segment), 2):
            x, y = segment[i], segment[i + 1]
            new_x, new_y = x - crop_left, y - crop_top

            if 0 <= new_x < crop_size and 0 <= new_y < crop_size:
                adjusted_segment.extend([new_x, new_y])

        if adjusted_segment:
            new_segmentation.append(adjusted_segment)

    if new_segmentation:
        points = np.concatenate(new_segmentation)
        min_x, min_y = np.min(points[::2]), np.min(points[1::2])
        max_x, max_y = np.max(points[::2]), np.max(points[1::2])
        new_bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
        new_area = (max_x - min_x) * (max_y - min_y)
    else:
        new_bbox, new_area = [0, 0, 0, 0], 0

    return {
        "id": annotation['id'],
        "category_id": annotation['category_id'],
        "iscrowd": annotation['iscrowd'],
        "segmentation": new_segmentation,
        "bbox": new_bbox,
        "area": new_area
    }

# Process images and annotations
cropped_annotations = {"images": [], "annotations": []}

for image_info in dataset['images']:
    image_id = image_info['id']
    image_name = image_info['file_name']
    image_output_dir = output_directory

    cropped_image_paths = crop_image_with_sliding_window(image_directory, image_name, crop_size, crop_times)
    
    if not cropped_image_paths:
        continue  # Skip if no valid crops are generated

    for crop_idx, (cropped_image, crop_name, crop_right, crop_bottom) in enumerate(cropped_image_paths):
        crop_path = image_output_dir / crop_name
        cv2.imwrite(str(crop_path), cropped_image)
        total_cropped_images += 1  # Increment total cropped images
        cropped_annotations["images"].append({
            "id": f"{image_id}_{crop_idx}",
            "file_name": crop_path.name,
            "width": crop_size,
            "height": crop_size
        })
        for annotation in dataset['annotations']:
            if annotation['image_id'] == image_id:
                cropped_annotation = adjust_annotations_for_crop(annotation, crop_right, crop_bottom, crop_size)
                cropped_annotation["image_id"] = f"{image_id}_{crop_idx}"
                cropped_annotations["annotations"].append(cropped_annotation)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
# Save the combined cropped annotations
with open(output_annotation_path, 'w') as out_file:
    json.dump(cropped_annotations, out_file, cls=NpEncoder)

# Output statistics
print(f"Saved cropped annotations to {output_annotation_path}")
print("\n--- Processing Summary ---")
print(f"Total images before crop: {total_images_before_crop}")
print(f"Total cropped images: {total_cropped_images}")
