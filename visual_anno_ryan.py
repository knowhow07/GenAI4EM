import json
import cv2
import os
from PIL import Image, ImageDraw
from pathlib import Path

option = 'val'  # train or val

# Set paths
current_dir = Path.cwd()
parent_directory = current_dir.parent / 'datasets' / 'dataset2_expanded_initial_split'

if parent_directory.is_dir():
    im_path = parent_directory / 'crop' / f'{option}'
    anno_path = parent_directory / 'crop' / 'annotations' / f'{option}.json'
else:
    raise FileNotFoundError("Parent directory doesn't exist")

# Load annotations
with open(anno_path, 'rb') as f:
    annos = json.load(f)

images = os.listdir(im_path)
class_colors = {1: 'yellow', 2: 'red', 3: 'blue'}

# Counters for each case
total_images = len(images)
no_annotation_count = 0
valid_images_count = 0
no_segmentation_count = 0

output_dir = current_dir / 'output_cropanno' / 'img_anno'
for file in output_dir.iterdir():
    if file.is_file():  # Check if it is a file
        file.unlink()  # Delete the file
        print(f"Deleted: {file}")

# Iterate through all images
for im_name in images:
    # Find the corresponding image ID in annotations
    image_id = None
    for im in annos['images']:
        if im['file_name'] == im_name:
            image_id = im['id']
            break
    
    if image_id is None:
        no_annotation_count += 1
        print(f"No matching annotation found for image {im_name}")
        continue

    # Get annotations corresponding to this image
    annos_thisimage = [
        anno for anno in annos['annotations'] if anno['image_id'] == image_id
    ]

    # Open the image
    im_raw_arr = cv2.imread(str(im_path / im_name))
    if im_raw_arr is None:
        print(f"Unable to open image: {im_name}")
        continue

    im_raw = Image.fromarray(cv2.cvtColor(im_raw_arr, cv2.COLOR_BGR2RGB))
    im1 = ImageDraw.Draw(im_raw)

    has_valid_segmentation = False

    # Draw bboxes and polygons on the image
    for anno in annos_thisimage:
        bbox = anno['bbox']
        cls = anno['category_id']
        polygons = anno.get('segmentation', [])

        # Draw polygons if they exist and have valid coordinates
        for polygon in polygons:
            if len(polygon) >= 6:  # A valid polygon must have at least 3 points (6 coordinates)
                try:
                    im1.polygon(polygon, outline=class_colors[cls])
                    has_valid_segmentation = True
                except Exception as e:
                    print(f"Error drawing polygon for image {im_name}: {e}")

        # Draw bounding boxes
        im1.rectangle(
            ((bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3])),
            outline=class_colors.get(cls, "white"),
        )

    if has_valid_segmentation:
        valid_images_count += 1
    else:
        no_segmentation_count += 1

    # Save the image with annotations
    
    output_dir.mkdir(parents=True, exist_ok=True)
    im_raw.save(output_dir / im_name)
    print(f"Processed and saved: {im_name}")

# Output the statistics
print("\n--- Processing Summary ---")
print(f"Total images: {total_images}")
print(f"Images with no annotations: {no_annotation_count}")
print(f"Images with valid annotations and segmentation: {valid_images_count}")
print(f"Images with annotations but no segmentation: {no_segmentation_count}")
