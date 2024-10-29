import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Define your directories
# dirname = os.getcwd()
# image_directory = os.path.join(dirname, 'image/')  #relative dir
# output_directory = os.path.join(dirname, 'output/')
image_directory = "/Users/nuohaoliu/Library/CloudStorage/OneDrive-UW-Madison/laps_example/other/GenAI/crop/image"
mask_directory = "/Users/nuohaoliu/Library/CloudStorage/OneDrive-UW-Madison/laps_example/other/GenAI/MaskRCNNObjectDetection/annotations/mask/"

# Parameters
crop_size = 512 # the output size
crop_times = 2  # Customize the crop times along each side, by sliding windows, ranges from 2 to 1000

def crop_image_with_sliding_window(image_path, name, crop_size, crop_times):
    # Open the original image
    image = cv2.imread(os.path.join(image_path, name))
    if image is None:
        print(f"Could not open {name}, skipping.")
        return
    img_height, img_width, _ = image.shape
    
    # Ensure the output directory exists
    os.makedirs(image_path, exist_ok=True)
    
    assert crop_times >= 0, "crop_times ranges from 2 to 1000"  
    slide_times = crop_times -1 # moveing twice will get 3 crop
     
    
    # Create a subfolder for cropped images by crop size
    subfolder = f"{crop_size}_crop_{crop_times}"
    path = os.path.join(image_path, subfolder)
    os.makedirs(path, exist_ok=True)
    

    crop_count = 0
    upper = img_width - crop_size
    for top in range(0, upper + 1, int(upper/slide_times)):
        for left in range(0, upper + 1, int(upper/slide_times)):
            # Define the right and bottom boundaries for cropping
            right = min(left + crop_size, img_width)
            bottom = min(top + crop_size, img_height)
            
            # Crop and save the image
            cropped_image = image[top:bottom, left:right]
            output_path = os.path.join(path, f"{name.split('.')[0]}_{crop_count + 1}.png")
            cv2.imwrite(output_path, cropped_image)
            crop_count += 1
            print(f"Saved: {output_path}")


def crop_mask_with_sliding_window(image_path, name, crop_size, crop_times):
    # Open the original image
    image = Image.open(os.path.join(image_path, name))
    image_array = np.array(image)
    
    # image = cv2.imread(os.path.join(image_path, name))
    if image is None:
        print(f"Could not open {name}, skipping.")
        return
    img_height, img_width= image_array.shape
    

    
    assert crop_times >= 0, "crop_times ranges from 2 to 1000"  
    slide_times = crop_times -1 # moveing twice will get 3 crop
     
    
    # Create a subfolder for cropped images by crop size
    subfolder = f"{crop_size}_crop_{crop_times}"
    path = os.path.join(image_path, subfolder)
    os.makedirs(path, exist_ok=True)
    
    crop_count = 0
    upper = img_width - crop_size
    for top in range(0, upper + 1, int(upper/slide_times)):
        for left in range(0, upper + 1, int(upper/slide_times)):
            # Define the right and bottom boundaries for cropping
            right = min(left + crop_size, img_width)
            bottom = min(top + crop_size, img_height)
            
            # Crop and save the image
            cropped_image_array = image_array[top:bottom, left:right]
            output_path = os.path.join(path, f"{name.split('.')[0]}_{crop_count + 1}.png")
            # cv2.imwrite(output_path, cropped_image)
            mask_image = Image.fromarray(cropped_image_array)
            mask_image.save(output_path)
            crop_count += 1
            print(f"Saved: {output_path}")

# Apply the sliding window crop to every image in the image folder
for filename in os.listdir(image_directory):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        print(f"Processing {filename} with sliding window...")
        crop_image_with_sliding_window(image_directory, filename, crop_size, crop_times)
    else:
        print(f"Skipping non-image file: {filename}")
        

# Apply the sliding window crop to every mask in the masks folder
for filename in os.listdir(mask_directory):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        print(f"Processing {filename} with sliding window...")
        crop_mask_with_sliding_window(mask_directory, filename, crop_size, crop_times)
    else:
        print(f"Skipping non-image file: {filename}")
