import numpy as np
import time
import glob
from tqdm import tqdm
import math
from skimage.measure import block_reduce
import matplotlib.pyplot as plt
from tractogram_functions import find_seed_points, tractogram
from PyQt5 import QtCore

# CPU option
from structure_tensor import eig_special_3d, structure_tensor_3d

# GPU option
#from structure_tensor.cp import eig_special_3d, structure_tensor_3d

def get_vectors_from_structure_tensors(volume, sigma = 1.5, rho = 5):	
	
	# Create structure tensor, do eigen decomposition
	S = structure_tensor_3d(volume, sigma, rho, truncate = 2.0)	
	val, vec = eig_special_3d(S)	
 
	# Reformat as needed
	vec = np.transpose(vec, axes=[1, 2, 3, 0])    
	
	# Convert vec to XYZ from ZYX (x, y, z, 1, 3)
	vec = vec[:, :, :, [1, 2, 0]]   
	  
	return val, vec

class StructureTensorClass(QtCore.QThread):
	progressSignal = QtCore.pyqtSignal(int)
	progressMinimumSignal = QtCore.pyqtSignal(int)
	progressMaximumSignal = QtCore.pyqtSignal(int)
	completeSignal = QtCore.pyqtSignal(int)
	statusBarSignal = QtCore.pyqtSignal(str)
 		
	def __init__(self, folderImagesPath, mask_image, affine, metadata, neighborhoodScale, noiseScale, seedsPerPixel, downsampleFactor, startSliceIndex = 0, forwardTrackingFlag = True, backwardTrackingFlag = False):

		super(StructureTensorClass, self).__init__(None)
		self.folderImagesPath = folderImagesPath
		self.mask_image = mask_image
		self.affine = affine
		self.metadata = metadata
		self.neighborhoodScale = neighborhoodScale
		self.noiseScale = noiseScale
		self.seedsPerPixel = seedsPerPixel
		self.downsampleFactor = downsampleFactor
		self.startSliceIndex = startSliceIndex
  
		self.streamlines_phys_coords = None
		self.color = None
  
		self.forwardTrackingFlag = forwardTrackingFlag
		self.backwardTrackingFlag = backwardTrackingFlag
  
		# Find seed point coordinates, colors
		self.seed_point_coordinates, self.color = find_seed_points(self.mask_image, self.seedsPerPixel)
  
	def run(self):

		streamlines_forward = []
		streamlines_backward = []
  
		compute_forward = self.forwardTrackingFlag and (self.startSliceIndex != self.metadata['num_images_to_read'] - 1)
		compute_backward = self.backwardTrackingFlag and (self.startSliceIndex != 0)
  
		if compute_forward:  
      
			self.progressMinimumSignal.emit(self.startSliceIndex)
			self.progressMaximumSignal.emit(self.metadata['num_images_to_read'] - 1)
			self.statusBarSignal.emit('Tracking in forward direction..')
          
			streamlines_forward = self.get_streamlines(self.startSliceIndex, self.metadata['num_images_to_read'] - 1, 1)

			self.progressSignal.emit(self.startSliceIndex)
   
		if compute_backward:
      
			self.progressMinimumSignal.emit(0)
			self.progressMaximumSignal.emit(self.startSliceIndex)
			self.statusBarSignal.emit('Tracking in backward direction..')
      
			streamlines_backward = self.get_streamlines(self.startSliceIndex, 0, -1)
   
			self.progressSignal.emit(0) 
     
		# Combine both streamlines (reverse the order of points in backward streamlines prior)
		# Remove the first set of points in the forward streamlines to avoid duplication on startSliceIndex
		for k in range(len(streamlines_backward)):
			streamlines_backward[k].reverse()

		self.statusBarSignal.emit('Converting streamlines to physical coordinates..')
  
		if compute_forward and compute_backward:   
			for k in range(len(streamlines_backward)):
				streamlines_backward[k].extend(streamlines_forward[k][1:])
			
			self.streamlines_phys_coords = tractogram(streamlines_backward, self.affine, self.metadata['y_size_pixels'],
                                             		  self.progressMinimumSignal, self.progressMaximumSignal, self.progressSignal, self.statusBarSignal)

		elif not compute_forward and compute_backward:
			self.streamlines_phys_coords = tractogram(streamlines_backward, self.affine, self.metadata['y_size_pixels'],
                                             		  self.progressMinimumSignal, self.progressMaximumSignal, self.progressSignal, self.statusBarSignal)
   
		elif compute_forward and not compute_backward:
			self.streamlines_phys_coords = tractogram(streamlines_forward, self.affine, self.metadata['y_size_pixels'],
                                             		  self.progressMinimumSignal, self.progressMaximumSignal, self.progressSignal, self.statusBarSignal)    			

		self.statusBarSignal.emit('Tracking complete.')		
		self.completeSignal.emit(2) 
  	
	def get_streamlines(self, startSliceIndex, stopSliceIndex, direction):
     
		# Streamlines variable (image space coordinates)
		streamlines = [[] for _ in range(self.seed_point_coordinates.shape[0])]
			
		# Streamline coordinates on the first slice are simply the seed point coordinates
		for k in range(self.seed_point_coordinates.shape[0]):
			streamlines[k].append((self.seed_point_coordinates[k,0], self.seed_point_coordinates[k,1], startSliceIndex))
		
		# Status of tracking for each streamline
		tracking_status = np.ones((len(streamlines)))  
  
		# Get filenames
		image_filelist = glob.glob(self.folderImagesPath + "\\*" + self.metadata['image_type'])
		
		# Get a chunk of the data
		step_size = self.metadata['step_size']
		larger_sigma = self.noiseScale if self.noiseScale > self.neighborhoodScale else self.neighborhoodScale
		overlap = int((2* larger_sigma + 0.5))
		ds_factor = self.downsampleFactor if self.downsampleFactor != 0 else 1
		
		# Getting this value since multiplying is faster than division
		us_factor = 1 / ds_factor
  
		# Angle threshold (75 degrees) in pixels, all ST computation is down after downsampling in XY to get near isotropic voxels
		angle_threshold_pixels = int(np.tan(np.deg2rad(75)) * self.metadata['section_thickness'] / (self.metadata['pixel_size_xy'] * ds_factor))
		
		stack_finished_flag = False

		# Chunk - a possibly larger piece of volume with overlap on each end with previous chunks
		# Compute - contiguous region inside a chunk in which ST vectors are used
		for i in tqdm(np.arange(self.startSliceIndex, stopSliceIndex + 1, step_size*direction)):
			
			if direction == 1:
				if i + step_size > stopSliceIndex:
					stop_chunk = stopSliceIndex
					stop_compute  = stop_chunk
					stack_finished_flag = True
				
				else:
					stop_chunk = min(i + step_size + overlap, stopSliceIndex)
					stop_compute = i + step_size
     
				if i - overlap < self.startSliceIndex:
					start_chunk = i
					start_compute = i
				else:
					start_chunk = i - overlap
					start_compute = i
     
			else:
				if i - step_size < stopSliceIndex:
					stop_chunk = stopSliceIndex
					stop_compute = stopSliceIndex
					stack_finished_flag = True
				else:
					stop_chunk = max(i - step_size - overlap, stopSliceIndex)
					stop_compute = i - step_size
     
				if i + overlap > self.startSliceIndex:
					start_chunk = i
					start_compute = i
				else:
					start_chunk = i + overlap
					start_compute = i
   
			volume = np.zeros((int(self.metadata['y_size_pixels'] * us_factor), int(self.metadata['x_size_pixels'] * us_factor), abs(start_chunk - stop_chunk)), dtype = 'uint8')
			image_stack = np.zeros((int(self.metadata['y_size_pixels']), int(self.metadata['x_size_pixels']), abs(start_chunk - stop_chunk)), dtype=np.uint8)
   			
			try:
				for j in np.arange(abs(start_chunk - stop_chunk)):
					image_stack[:,:,j] = (plt.imread(image_filelist[start_chunk + (j*direction)])* 255).astype('uint8')
			except ValueError:
				print('Could not read image files, possible error in metadata file for image size fields or images not present in specified path')
				return None, None
			
			volume = block_reduce(image_stack, (ds_factor, ds_factor, 1), np.mean)
			
			val, vec = get_vectors_from_structure_tensors(volume.astype('float32'), sigma = self.noiseScale, rho = self.neighborhoodScale)

			# fa = np.empty((val.shape[1], val.shape[2], val.shape[3]))    
			# fa = np.sqrt(0.5 * ((val[0,] - val[1,]) ** 2 + (val[1,] - val[2,]) ** 2 + (val[2,] - val[0,]) ** 2) / 
            #      ((val[0,] * val[0,]) + (val[1,] * val[1,]) + (val[2,] * val[2,])))
   
			# line_measure = np.empty((val.shape[1], val.shape[2], val.shape[3])) 
			# line_measure = (val[1,] - val[0,]) / val[2,]
   
			for k in np.arange(abs(start_compute - stop_compute)):
				
				vector_field = vec[:, :, abs(start_compute - start_chunk) + k]
				
				for l in np.arange(len(streamlines)):
					
					if not tracking_status[l]:
						continue
						
					old_x_coordinate = streamlines[l][-1][0]
					old_y_coordinate = streamlines[l][-1][1]

					# Ensure the index is not out of bounds
					if int(math.floor(old_y_coordinate * us_factor)) >= vector_field.shape[0]:
						y_index = vector_field.shape[0] - 1
					else:
						y_index = int(math.floor(old_y_coordinate * us_factor))

					if int(math.floor(old_x_coordinate * us_factor)) >= vector_field.shape[1]:
						x_index = vector_field.shape[1] - 1
					else:
						x_index = int(math.floor(old_x_coordinate * us_factor))
						
					current_vector = vector_field[y_index, x_index]
     
					# current_fa = fa[y_index, x_index, k]
					# current_line_measure = line_measure[y_index, x_index, k]
     
					# If fractional anisotropy is under a threshold, stop tracking
					# Or line measure
					# if current_line_measure < 0.1:
					# 	tracking_status[l] = 0
					# 	continue
     
					# If the z component of the vector is zero, indicates that the 
					# new location is in the same slice, which we do not consider for now.
					if current_vector[2] == 0:
						tracking_status[l] = 0
						continue

					# Flip z direction if needed, always move from one slice to the next
					if current_vector[2] < 0:
						current_vector = - current_vector
      					
					# Scale the vector by dividing with the z component, so that it becomes equal to 1
					x_coordinate_change = current_vector[0] / current_vector[2]
					y_coordinate_change = current_vector[1] / current_vector[2]

					# Large step in one direction compared to other, probably at a stitching slice or artifact					
					# Keep the same coordinates as previous
					if abs(abs(x_coordinate_change) - abs(y_coordinate_change)) > angle_threshold_pixels:
						streamlines[l].append((old_x_coordinate, old_y_coordinate, i + ((k+1) * direction)))
						continue  

					# Check if new location would be within the angle threshold at original resolution
					if abs(x_coordinate_change) > angle_threshold_pixels or abs(y_coordinate_change) > angle_threshold_pixels:
						tracking_status[l] = 0
						continue
    									
					new_x_coordinate = (old_x_coordinate * us_factor) + x_coordinate_change
					new_y_coordinate = (old_y_coordinate * us_factor) + y_coordinate_change
					
					streamlines[l].append((new_x_coordinate * ds_factor, new_y_coordinate * ds_factor, i + ((k+1)*direction)))
					
			self.progressSignal.emit(i if direction == 1 else (startSliceIndex - i))
   
			if stack_finished_flag:
				break

		return streamlines

	def get_streamlines_and_colors(self):
     
		return self.streamlines_phys_coords, self.color	

	def terminate_thread(self):
		
		self.quit()
		self.wait()