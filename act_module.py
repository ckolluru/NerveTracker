# Import required libraries
import numpy as np
from tqdm import tqdm
from PyQt5 import QtCore
from PIL import Image

class ACTClass(QtCore.QThread):
	
	progressSignal = QtCore.pyqtSignal(int)
	progressMinimumSignal = QtCore.pyqtSignal(int)
	progressMaximumSignal = QtCore.pyqtSignal(int)
	completeSignal = QtCore.pyqtSignal(int) 
	statusBarSignal = QtCore.pyqtSignal(str)
  		
	def __init__(self, num_images_to_process, streamlines_image_coords, fascicle_label_filelist):

		super(ACTClass, self).__init__(None)
		self.num_images_to_process = num_images_to_process
		self.streamlines_image_coords = streamlines_image_coords
		self.fascicle_label_filelist = fascicle_label_filelist
		
	# Clip streamlines if they go out of the fascicle mask
	def run(self):
	 
		self.statusBarSignal.emit('Iterating over label images and clipping if streamlines are moving out.')
		self.progressMinimumSignal.emit(0)
		self.progressMaximumSignal.emit(self.num_images_to_process)
  
		# iterate over the set of label images, ignore the first slice
		for k in tqdm(np.arange(1, self.num_images_to_process)):
			
			self.progressSignal.emit(k)
			label = Image.open(self.fascicle_label_filelist[k])
			mask = np.array(label)
			mask[mask != 0] = 1

			# Go to each streamline and check if there is a point on the current slice index
			for i in np.arange(len(self.streamlines_image_coords)):
				current_streamline = self.streamlines_image_coords[i]
				streamline_length = len(current_streamline)
				z_index_of_first_point = current_streamline[0][2]

				# Check if the z index of the first point in the streamline is greater than the current slice index
				if z_index_of_first_point > k:
					continue
				else:
					index_within_streamline_with_current_slice_index = k - z_index_of_first_point

					# If a point exists in the current streamline with the same z index we are currently looking at
					if index_within_streamline_with_current_slice_index < streamline_length:
						x_coordinate = current_streamline[index_within_streamline_with_current_slice_index][0]
						y_coordinate = current_streamline[index_within_streamline_with_current_slice_index][1]

						if mask[y_coordinate, x_coordinate]:
							continue
						else:
							self.streamlines_image_coords[i] = current_streamline[:index_within_streamline_with_current_slice_index]
	   
		self.progressSignal.emit(0)
		self.statusBarSignal.emit('Anatomically constrained tractography complete.')
		self.completeSignal(1)
   
	def get_streamlines_image_coords(self):
     
		return self.streamlines_image_coords

	def terminate_thread(self):
		
		self.quit()
		self.wait()