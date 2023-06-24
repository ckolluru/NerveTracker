# Import required libraries
import cv2
import numpy as np

from scipy.ndimage import gaussian_filter
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

from tractogram_functions import find_seed_points, tractogram
from PyQt5 import QtCore
import zarr

class OpticFlowClass(QtCore.QThread):
    
	progressSignal = QtCore.pyqtSignal(int)
	progressMinimumSignal = QtCore.pyqtSignal(int)
	progressMaximumSignal = QtCore.pyqtSignal(int)
	completeSignal = QtCore.pyqtSignal(int) 
	statusBarSignal = QtCore.pyqtSignal(str)
  		
	def __init__(self, folderImagesPath, mask_image, affine, metadata, windowSize, maxLevel, seedsPerPixel, gaussianSigma, startSliceIndex = 0, forwardTrackingFlag = True, backwardTrackingFlag = False):

		super(OpticFlowClass, self).__init__(None)
		self.folderImagesPath = folderImagesPath
		self.mask_image = mask_image
		self.affine = affine
		self.metadata = metadata
		self.windowSize = windowSize
		self.maxLevel = maxLevel
		self.seedsPerPixel = seedsPerPixel
		self.gaussianSigma = gaussianSigma
		self.startSliceIndex = startSliceIndex
  
		self.streamlines_phys_coords = None
		self.color = None
		self.forwardTrackingFlag = forwardTrackingFlag
		self.backwardTrackingFlag = backwardTrackingFlag
  
  		# Parameters for lucas kanade optical flow
		self.lk_params = dict(winSize  = (self.windowSize, self.windowSize),
						maxLevel = self.maxLevel,
						criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
  
  		# Find seed point coordinates, colors
		self.seed_point_coordinates, self.color = find_seed_points(self.mask_image, self.seedsPerPixel)
		
	# Optical flow using the Lucas Kanade algorithm
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
		self.completeSignal.emit(1)   
   
	# Run the algorithm from one slice to another
	def get_streamlines(self, startSliceIndex, stopSliceIndex, direction):
     
     	# Streamlines variable (image space coordinates)
		streamlines = [[] for _ in range(self.seed_point_coordinates.shape[0])]
			
		# Streamline coordinates on the first slice are simply the seed point coordinates
		for k in range(self.seed_point_coordinates.shape[0]):
			streamlines[k].append((self.seed_point_coordinates[k,0], self.seed_point_coordinates[k,1], startSliceIndex))   

		# Tracking status (1 to continue tracking, 0 to stop)
		tracking_status = np.ones((len(streamlines)))
   
		# Get filenames
		if self.metadata['image_type'] == '.png':
			image_filelist = glob.glob(self.folderImagesPath + "\\*.png")
		else:
			dataset = zarr.open(self.folderImagesPath)
			muse_dataset = dataset['muse']
  
		# Angle threshold (75 degrees) in pixels
		angle_threshold_pixels = int(np.tan(np.deg2rad(75)) * self.metadata['section_thickness'] / self.metadata['pixel_size_xy'])
		
		# Loop through each frame, update list of points to track, streamlines
		for i in tqdm(np.arange(startSliceIndex, stopSliceIndex, direction)):
			
			# Blur the images a bit to have a better gradient, else will be noisy
			if i == self.startSliceIndex:
				try:
					if self.metadata['image_type'] == '.png':
						image = (plt.imread(image_filelist[self.startSliceIndex])* 255).astype('uint8')
					else:
						image = np.squeeze(np.array(muse_dataset[self.startSliceIndex, 0, :, :]))
						image = image.astype('uint8')
				except ValueError:
					print('Could not read image files, possible error in metadata file for image size fields or images not present in specified path')
					return None
				current_image = gaussian_filter(image, sigma = self.gaussianSigma)
			else:
				current_image = next_image
			
			try:
				if self.metadata['image_type'] == '.png':
					image = (plt.imread(image_filelist[i+direction])* 255).astype('uint8')
				else:
					image = np.squeeze(np.array(muse_dataset[i+direction, 0, :, :]))
					image = image.astype('uint8')					
			except ValueError:
				print('Could not read image files, possible error in metadata file for image size fields or images not present in specified path')
				return None
			
			next_image = gaussian_filter(image, sigma = self.gaussianSigma)

			# OpenCV needs points in the form (n, 1, 2), just the XY coordinates
			points_to_track = np.float32([tr[-1] for tr in streamlines]).reshape(-1, 1, 3)[:,:,:2]
			
			# Calculate optical flow
			new_location_of_tracked_points, status_fw, err_fw = cv2.calcOpticalFlowPyrLK(current_image, next_image, points_to_track, None, **self.lk_params)
			
			new_tracks = []
			for tr, (x, y), status, index in zip(streamlines, new_location_of_tracked_points.reshape(-1, 2), status_fw, np.arange(len(streamlines))):
				
				if not status:
					tracking_status[index] = 0
					new_tracks.append(tr)
					continue
 
				if tracking_status[index]:
					diff_in_x_coordinate = abs(tr[-1][0] - x)
					diff_in_y_coordinate = abs(tr[-1][1] - y)    

					# If there is a big jump in one direction compared to other, then that would probably be at a stitching slice or cutting artifact.
					# Keep the same coordinates as previous
					if abs(diff_in_x_coordinate - diff_in_y_coordinate) > angle_threshold_pixels:
						tr.append((tr[-1][0], tr[-1][1], i + direction))

					# Check if the new points are within a 75 degree angle
					if (diff_in_x_coordinate < angle_threshold_pixels) and (diff_in_y_coordinate < angle_threshold_pixels):
						tr.append((x, y, i + direction))
					else:
						tracking_status[index] = 0
					new_tracks.append(tr)
				else:
					new_tracks.append(tr)
     
			streamlines = new_tracks

			self.progressSignal.emit(i if direction == 1 else (startSliceIndex - i))
   
		return streamlines
	
	# Return streamlines and colors back to the main window for display
	def get_streamlines_and_colors(self):
     
		return self.streamlines_phys_coords, self.color

	# Stop the thread
	def terminate_thread(self):
		
		self.quit()
		self.wait()