import pickle
import numpy as np
import cv2
import tkinter as tk
import numpy as np
from PIL import Image
import glob
import os
import matplotlib.pyplot as plt

# Create a function that will be called when a color is selected in a dialog box
def on_color_select(event):
    global color_index
    item = canvas.find_closest(event.x, event.y)[0]
    color_index = item - 1
    top.destroy()

if __name__ == '__main__':
    
    streamlinesFolder = r'D:\MUSE processing\PyFibers\pickle files\comparison-151-iodine\\ST 3x DS\\'
    streamlinesFile = glob.glob(streamlinesFolder + 'streamlines*.pkl')[0]
    colorsFile = os.path.dirname(streamlinesFile) + '\\colors_' + os.path.basename(streamlinesFile)[12:]
    validation_masks = r'D:\\MUSE_datasets\\Tractography\\151-iodine\\Manual segmentations 3x\\'
    
    ds_factor = 3
    section_thickness = 3
    normalize_wrt_slice_physical_distance_microns_array_index = 0
        
    # fast-stain-thin-3mm
    # slice_physical_distance_microns = np.array([0, 300, 600, 902]) * section_thickness
    # image_width = int(2196 / ds_factor)
    # image_height = int(2380 / ds_factor)
    # pixel_size = (0.74 * ds_factor)
    # color_names = ['Blue', 'Light green', 'Dark green', 'Pink', 'Orange'] # check from color ordering from GUI
    
    # 151-iodine
    slice_physical_distance_microns = np.array([0, 499, 999, 1499]) * section_thickness
    image_width = int(4000 / ds_factor)
    image_height = int(3000 / ds_factor)
    pixel_size = (0.9 * ds_factor)
    color_names = ['Blue', 'Light green', 'Pink', 'Orange'] # check from color ordering from GUI

    check_color_ordering = False
    
    if check_color_ordering:
        with open(colorsFile, 'rb') as f:
            colors = pickle.load(f)       
        
        unique_colors = np.unique(colors, axis = 0)
            
        # Create the dialog box
        top = tk.Tk()
        top.title("Colors in the tractogram")

        # Create a canvas to display the colors
        canvas = tk.Canvas(top, width=200, height=150)
        canvas.pack()

        # Create rectangles with the colors on the canvas
        for i in range(len(unique_colors)):
            r, g, b = np.uint8(unique_colors[i] * 255)        
            hex_color = "#{:02x}{:02x}{:02x}".format(r, g, b)
            x0, y0, x1, y1 = i * 33, 0, (i + 1) * 33, 33
            rect = canvas.create_rectangle(x0, y0, x1, y1, fill=hex_color, outline=hex_color)
            canvas.tag_bind(rect, '<Button-1>', on_color_select)

        # Run the dialog box so the user can identify the colors, these colors sequentially go into color_names above
        top.mainloop()
    
    else:   

        iou = np.zeros((len(color_names), len(slice_physical_distance_microns)))
        dice = np.zeros((len(color_names),len(slice_physical_distance_microns)))   
                
        for k in range(len(color_names)):
            
            with open(colorsFile, 'rb') as f:
                colors = pickle.load(f)       

            unique_colors = np.unique(colors, axis = 0)
            color_of_interest = unique_colors[k]
        
            validation_masks_folder = validation_masks + color_names[k]
            validation_masks_list = glob.glob(validation_masks_folder + '\\*.png')    
                        
            with open(streamlinesFile, 'rb') as f:
                streamlines = pickle.load(f)

            # Convert streamline array into point cloud arrays
            # Get a list of all unique z coordinates across all numpy arrays in the list
            unique_z_values = np.unique(np.concatenate([arr[:, 2] for arr in streamlines]))
                
            # Create an empty list to hold the x and y coordinates of points at the current z value
            points = [[] for _ in range(len(slice_physical_distance_microns))]
            
            for index_z_val, z_val in enumerate(slice_physical_distance_microns):
                
                # Iterate over each 2D numpy array in the list
                for index, arr in enumerate(streamlines):
                    if not np.array_equiv(color_of_interest, colors[index]):
                        continue
                    
                    # Find the indices of all points in the current array with z coordinate equal to `z_val`
                    indices = np.where(arr[:, 2] == z_val)[0]
                    
                    # Extract the x and y coordinates of those points and add them to `points`
                    points[index_z_val].extend(arr[indices, :2])
                    
                points_in_streamlines = np.array(points[index_z_val])
            
                # Need to go to image space and flip y axis coordinates
                points_in_streamlines = points_in_streamlines/pixel_size    
                points_in_streamlines[:,1] = image_height - points_in_streamlines[:,1]
                
                mask_image = Image.open(validation_masks_list[index_z_val])
                mask_image = np.array(mask_image)
                
                streamline_mask_image = np.zeros((image_height, image_width))
                y_indices = points_in_streamlines[:,1].astype(int)
                x_indices = points_in_streamlines[:,0].astype(int)
                
                streamline_mask_image[y_indices, x_indices] = 255
                
                # Convert the images to binary masks using thresholding
                _, mask1 = cv2.threshold(mask_image, 0, 1, cv2.THRESH_BINARY)
                _, mask2 = cv2.threshold(streamline_mask_image, 0, 1, cv2.THRESH_BINARY)

                mask2 = mask2.astype('uint8')
                
                assert mask1.shape == mask2.shape
                        
                # Compute the Dice coefficient between the two masks
                intersection = mask1 & mask2
                union = mask1 | mask2
                iou[k, index_z_val] = np.sum(intersection) / np.sum(union)
                
                intersection = np.sum(mask1 * mask2)
                dice[k, index_z_val] = (2. * intersection) / (np.sum(mask1) + np.sum(mask2))
                
        dice_color_averaged = np.mean(dice, axis=0)
        
    print('Dice: ', dice_color_averaged)
    
    normalized_dice = dice_color_averaged/dice_color_averaged[normalize_wrt_slice_physical_distance_microns_array_index]
    normalized_dice = np.delete(normalized_dice, normalize_wrt_slice_physical_distance_microns_array_index)
    print('Normalized Dice: ', normalized_dice)
    print('Mean and sd: ', np.round(np.mean(normalized_dice),2), np.round(np.std(normalized_dice),2))
    
    print('Mean Dice_norm: ', np.mean(normalized_dice))