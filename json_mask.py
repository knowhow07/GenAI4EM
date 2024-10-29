import json
import os
import numpy as np

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
 
# Load the JSON file
dir = '/Users/nuohaoliu/Library/CloudStorage/OneDrive-UW-Madison/laps_example/other/GenAI/MaskRCNNObjectDetection/annotations/'
json_file = dir + 'all_dataset2_Mingren_expanded.json'

with open(json_file) as f:

    data = json.load(f)
 
# Define the function to create segmentation masks with different values for different IDs
def create_segmentation_mask_with_ids(image_data, annotations, image_shape):
    # Create a blank mask
    mask = Image.new('L', (image_shape[1], image_shape[0]), 0)
    # Loop through annotations for the given image
    for annotation in annotations:
        if annotation['image_id'] == image_data['id']:
            # Get segmentation points and the ID of the annotation
            segmentation = annotation['segmentation']
            annotation_id = annotation['category_id']
            print(annotation_id)
            # Flatten the list of tuples into a list of coordinates
            for segment in segmentation:
                flattened_segmentation = [(segment[i], segment[i + 1]) for i in range(0, len(segment), 2)]
 
                # Draw and fill the polygon with the annotation ID value
                ImageDraw.Draw(mask).polygon(flattened_segmentation, outline=annotation_id, fill=annotation_id)
 
    # Convert the mask to a numpy array
    mask_array = np.array(mask)
    return mask_array
 
# Iterate over all images in the dataset and create their segmentation masks
for image in data['images']:
    image_shape = (image['height'], image['width'])
    mask_array = create_segmentation_mask_with_ids(image, data['annotations'], image_shape)
    # Save the mask as an image
    mask_image = Image.fromarray(mask_array)
    mask_image.save(f"{dir}masks/{image['file_name'].replace('.jpg', '_mask.png')}")
  






 