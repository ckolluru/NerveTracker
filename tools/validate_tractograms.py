import pickle
import numpy as np
import cv2
import tkinter as tk
import numpy as np
from PIL import Image
import glob
import os
import matplotlib.pyplot as plt
import sys
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QLabel, QFormLayout

class PopUpWindow(QDialog):
	def __init__(self, parent=None, unique_colors=None):
		super(PopUpWindow, self).__init__(parent)
		self.initUI(unique_colors)

	def initUI(self, unique_colors):
		self.setWindowTitle('Validate tractograms')

		label = QLabel()
		label.setText('Enter sub-folder names in the manual segmentation folder corresponding to color')
		label_layout = QHBoxLayout()
		label_layout.addWidget(label)
  
		# Create a list to store line edit values
		self.line_edit_values = []
		
		# Create the layout and add the widgets to it
		layout = QFormLayout()
  
		# List to contain all line edit items
		self.line_edits = []

		# Create a horizontal layout for each row
		for i in range(unique_colors.shape[0]):

			color = unique_colors[i,:] 
			
			# Create a label with the color
			color_label = QLabel('')
			color_label.setStyleSheet(f"background-color: rgb(" + str(color[0]) + ", " + str(color[1]) + "," + str(color[2]) + ")")     
			
			# Create a line edit item
			line_edit = QLineEdit()
			self.line_edits.append(line_edit)
			layout.addRow(color_label, line_edit)

		self.ok_button = QPushButton('OK', self)
		self.ok_button.clicked.connect(self.accept)	
		buttons_layout = QHBoxLayout()
		buttons_layout.addStretch()
		buttons_layout.addWidget(self.ok_button)
		buttons_layout.addStretch()

		main_layout = QVBoxLayout(self)
		main_layout.addLayout(label_layout)
		main_layout.addLayout(layout)
		main_layout.addLayout(buttons_layout)
  
		self.setLayout(main_layout)

	def get_color_names(self):

		color_names = [line_edit.text() for line_edit in self.line_edits]
		return color_names
		
def validate(streamlinesFilePath, colorsFile, validation_masks, validationMetadata):
	
	streamlinesFile = streamlinesFilePath
  
	ds_factor = validationMetadata['ds_factor']
	normalize_wrt_slice_physical_distance_microns_array_index = validationMetadata['normalize_wrt_slice_physical_distance_microns_array_index']

	slice_physical_distance_microns = np.array(validationMetadata['slice_numbers_to_evaluate']) * validationMetadata['section_thickness']
	image_width = int(validationMetadata['image_width'] / ds_factor)
	image_height = int(validationMetadata['image_height']  / ds_factor)
	pixel_size = (validationMetadata['pixel_size'] * ds_factor)

	with open(colorsFile, 'rb') as f:
		colors = pickle.load(f)       
		
	unique_colors = np.unique(colors * 255, axis = 0)

	popup = PopUpWindow(unique_colors = unique_colors)
	if popup.exec_() == QDialog.Accepted:
		color_names = popup.get_color_names()   

	iou = np.zeros((len(color_names), len(slice_physical_distance_microns)))
	dice = np.zeros((len(color_names),len(slice_physical_distance_microns)))  
 
	false_positives = np.zeros((len(color_names), len(slice_physical_distance_microns)))
	true_positives = np.zeros((len(color_names), len(slice_physical_distance_microns)))
	false_negatives = np.zeros((len(color_names), len(slice_physical_distance_microns)))
	true_negatives = np.zeros((len(color_names), len(slice_physical_distance_microns)))
			
	for k in range(len(color_names)):
		
		with open(colorsFile, 'rb') as f:
			colors = pickle.load(f)       

		unique_colors = np.unique(colors, axis = 0)
		color_of_interest = unique_colors[k]
	
		validation_masks_folder = validation_masks + '\\' + color_names[k]
		validation_masks_list = glob.glob(validation_masks_folder + '\\*.png')    
					
		with open(streamlinesFilePath, 'rb') as f:
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
   
			false_positives[k, index_z_val] = np.sum((mask2 == 1) & (mask1 == 0))
			true_positives[k, index_z_val] = np.sum((mask2 == 1) & (mask1 == 1))
			false_negatives[k, index_z_val] = np.sum((mask2 == 0) & (mask1 == 1))
			true_negatives[k, index_z_val] = np.sum((mask2 == 0) & (mask1 == 0))
				
		dice_color_averaged = np.mean(dice, axis=0)		
 
	normalized_dice = dice_color_averaged/dice_color_averaged[normalize_wrt_slice_physical_distance_microns_array_index]
	normalized_dice = np.delete(normalized_dice, normalize_wrt_slice_physical_distance_microns_array_index)
 
	return dice_color_averaged, normalized_dice,  false_positives, true_positives, false_negatives, true_negatives