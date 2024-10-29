import json
import os
import numpy as np

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
 
# Load the JSON file
dir = '/Users/nuohaoliu/Library/CloudStorage/OneDrive-UW-Madison/laps_example/other/GenAI/MaskRCNNObjectDetection/annotations/mask/512_crop_2/'

for filename in os.listdir(dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        print(filename)
        image = Image.open(dir+"3ROI_100kx_4100CL_foil1_mask_1.png")
        
        image_array = np.array(image)
        print(image_array.shape)
        plt.imshow(image)
        plt.axis('off')  # Turn off axis labels
        plt.show()

    else:
        print(f"Skipping non-image file: {filename}")

# image = Image.open(dir+'512_crop_2/2ROI_100kx_4100CL_foil1_mask_1.png')
# image = Image.open(dir+'1ROI_100kx_4100CL_foil1_mask.png')
# image_array = np.array(image)
# print(image_array.shape)
# # np.savetxt("my_array.txt", image_array, delimiter="/n") 
    
# plt.imshow(image_array)
# plt.axis('off')  # Turn off axis labels
# plt.show()



# Display the image


 