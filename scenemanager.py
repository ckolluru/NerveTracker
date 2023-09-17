import vtk
import datetime
import numpy as np
from visualization.vtk_helper_functions import lines_to_vtk_polydata
import pickle
import os
import math
import zarr
from vtkmodules.util import numpy_support
from skimage import exposure
import cv2
from PIL import Image

class SceneManager:

	def __init__(self, vtkWidget=None):
  
		# The renderers, render window and interactor
		renderers = list()
		self.window = vtkWidget.GetRenderWindow()
		
		# The first renderer 
		renderers.append(vtk.vtkRenderer())
		self.window.AddRenderer(renderers[0])
  
		# The second renderer
		self.ren = vtk.vtkRenderer()
		renderers.append(self.ren)
		self.window.AddRenderer(self.ren)

		# Interactor style
		self.iren = self.window.GetInteractor()
		self.iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
		self.iren.Initialize()  

		# Layer 0 - background not transparent
		colors = vtk.vtkNamedColors()
		renderers[0].SetBackground(colors.GetColor3d("White"))
		renderers[0].SetLayer(0)
  
		# Layer 1 - the background is transparent, so we only see the layer 0 background color
		renderers[1].SetLayer(1)

		#  We have two layers
		self.window.SetNumberOfLayers(2)    

		# Variables
		self.streamlines = None
		self.color = None
		self.streamlineClusters = None
		self.streamlineClustersColors = None
  
		# Actors
		self.clustersActor = None		
		self.streamlinesActor = None
		self.bbActor = None
		self.XYSliceActor = None
		self.streamline_poly_mapper = None
     
		# Visualization, metadata
		self.parallelView = 0  
		self.extent_x = None
		self.extent_y = None
		self.extent_z = None
		self.pixel_size_xy = None
		self.image_slice_thickness = None

		# Create orientation axes
		axes = vtk.vtkAxesActor()
		axes.SetShaftTypeToCylinder()

		self.orient = vtk.vtkOrientationMarkerWidget()
		self.orient.SetOrientationMarker(axes)
		self.orient.SetInteractor(self.iren)
		self.orient.SetViewport(0.0, 0.0, 0.2, 0.2)
		self.orient.SetEnabled(1)		
		self.orient.InteractiveOff()
		self.orient.SetEnabled(0)
  
		# Contouor widget, interactor object for window/level adjustments
		self.contourWidget = None
		self.vtkInteractorStyleImageObject = None

	# Set view to XY
	def SetViewXY(self):		
		camera = self.ren.GetActiveCamera()
  
		if self.pixel_size_xy is None:
			return False
   
		if self.parallelView:
			camera.SetFocalPoint(0, 0, 0)
			camera.SetViewUp(0, 1, 0)

			yd = self.extent_y * self.pixel_size_xy
			d = camera.GetDistance()
			camera.SetParallelScale(0.5*yd)
			camera.SetPosition(0, 0, d)
		else:
			camera.SetFocalPoint(0, 0, 0)
			camera.SetViewUp(0, 1, 0)
   
			d = camera.GetDistance()
			h = 0.5 * self.extent_y * self.pixel_size_xy
			camera.SetViewAngle(30)
			d = h / (math.tan(math.pi/12))
			camera.SetPosition(0, 0, d)
        
		camera.UpdateViewport(self.ren)
		self.iren.ReInitialize()   
		self.window.Render()

		return True

	# Set view to XZ
	def SetViewXZ(self):     
     
		camera = self.ren.GetActiveCamera()
   
		if self.parallelView:
			camera.SetFocalPoint(0, 0, 0)
			camera.SetViewUp(0, 0, 1)

			zd = self.extent_z * self.image_slice_thickness
			d = camera.GetDistance()
			camera.SetParallelScale(0.5*zd)
			camera.SetPosition(0, d, 0)
		else:
			camera.SetFocalPoint(0, 0, 0)
			camera.SetViewUp(0, 0, 1)
   
			d = camera.GetDistance()
			h = 0.5 * self.extent_z * self.image_slice_thickness
			camera.SetViewAngle(30)
			d = h / (math.tan(math.pi/12))
			camera.SetPosition(0, d, 0)
        
		camera.UpdateViewport(self.ren)
		self.iren.ReInitialize()   
		self.window.Render()

	# Set view to YZ
	def SetViewYZ(self):

		camera = self.ren.GetActiveCamera()

		if self.parallelView:
			camera.SetFocalPoint(0, 0, 0)
			camera.SetViewUp(0, 0, 1)

			zd = self.extent_z * self.image_slice_thickness
			d = camera.GetDistance()
			camera.SetParallelScale(0.5*zd)
			camera.SetPosition(d, 0, 0)
		else:
			camera.SetFocalPoint(0, 0, 0)
			camera.SetViewUp(0, 0, 1)

			d = camera.GetDistance()
			h = 0.5 * self.extent_z * self.image_slice_thickness
			camera.SetViewAngle(30)
			d = h / (math.tan(math.pi/12))
			camera.SetPosition(d, 0, 0)

		camera.UpdateViewport(self.ren)
		self.iren.ReInitialize() 
		self.window.Render()
  
		# camera = self.ren.GetActiveCamera()
		# halfViewAngleRadians = camera.GetViewAngle() * np.pi / 360
		# hypotenuse = self.extent_y / (2 * np.sin(halfViewAngleRadians))
		# distance = np.cos(halfViewAngleRadians) * hypotenuse
		# camera.SetPosition(distance,0.0,0.0)
		# camera.SetViewUp(0.0,0.0,1.0)
		# camera.SetFocalPoint(0.0,0.0,0)
		# self.window.Render()

	# Take a snapshot of the viewer and save to disk
	def Snapshot(self):
		wintoim=vtk.vtkWindowToImageFilter()
		self.window.Render()
		wintoim.SetInput(self.window)
		wintoim.Update()

		snapshot = vtk.vtkPNGWriter()
		filenamesnap = "snapshots\\snapshot_" + datetime.datetime.now().strftime("%H%M%S_%m%d%Y") + ".png"
		snapshot.SetFileName(filenamesnap)
		snapshot.SetInputConnection(0,wintoim.GetOutputPort())
		snapshot.Write()

	# Visualize axis based on checkbox value
	def ToggleVisualizeAxis(self, visible):
     
     	# Needed to set InteractiveOff
		self.orient.SetEnabled(1)		
		self.orient.InteractiveOff()
		self.orient.SetEnabled(visible)
		self.window.Render()

	# Create the actor displaying the XY slice
	def createXYSliceActor(self, imagesPath, pixel_size_xy, image_slice_thickness, x_size_pixels, y_size_pixels, num_images_to_read, image_type):
		
		# Update metadata
		self.extent_x = x_size_pixels - 1
		self.extent_y = y_size_pixels - 1
		self.extent_z = num_images_to_read - 1

		self.pixel_size_xy = pixel_size_xy
		self.image_slice_thickness = image_slice_thickness
		self.image_type = image_type

		if self.image_type == '.png':
			# PNG dataasets
			reader = vtk.vtkPNGReader()
			reader.SetFilePrefix(imagesPath + '\\') 
			reader.SetFilePattern('%sImage_%05d.png')
			reader.SetDataExtent(0, self.extent_x, 0, self.extent_y, 0, self.extent_z)
			reader.SetDataSpacing(self.pixel_size_xy, self.pixel_size_xy, self.image_slice_thickness)
			reader.SetDataScalarTypeToUnsignedChar()
			reader.SetNumberOfScalarComponents(1)
		
			# Extract VOI (first slice in stack)
			self.imageXY = vtk.vtkExtractVOI()
			self.imageXY.SetInputConnection(reader.GetOutputPort())
			self.imageXY.SetVOI(0, self.extent_x, 0, self.extent_y, 0, 0)
			self.imageXY.Update()
		else:
			# Zarr datasets
			dataset = zarr.open(imagesPath)
			self.muse_dataset = dataset['muse']
			image = np.flipud(np.squeeze(self.muse_dataset[0,:,:,:]))
			image = self.adjust_contrast_zarr(image)
			self.vtk_img = vtk.vtkImageData()
			self.vtk_img.SetSpacing(self.pixel_size_xy, self.pixel_size_xy, self.image_slice_thickness)
			self.vtk_img.SetDimensions(image.shape[1], image.shape[0], 1)
			self.vtk_img.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
			self.vtk_img.GetPointData().GetScalars().DeepCopy(numpy_support.numpy_to_vtk(np.reshape(image, (image.shape[1]*image.shape[0], 1))))

		# Set up an imageActor
		self.XYSliceActor = vtk.vtkImageActor()
		self.XYSliceActor.SetPosition(-self.extent_x*self.pixel_size_xy / 2, -self.extent_y*self.pixel_size_xy / 2,  (-self.extent_z *self.image_slice_thickness / 2))

		if self.image_type == '.png':
			self.XYSliceActor.GetMapper().SetInputConnection(self.imageXY.GetOutputPort())
		else:
			self.XYSliceActor.GetMapper().SetInputData(self.vtk_img)
		
		# Set look up table and properties
		ip = vtk.vtkImageProperty()
		ip.SetColorWindow(255)
		ip.SetColorLevel(128)
		ip.SetAmbient(0.0)
		ip.SetDiffuse(1.0)
		ip.SetOpacity(1.0)
		ip.SetInterpolationTypeToLinear()
		
		# Update the actor, ready for display
		self.XYSliceActor.SetProperty(ip)
		self.XYSliceActor.Update()
  
  	# Remove the XY slice from the renderer	
	
	# Remove the XY slice actor from the renderer if exists
	def removeXYSliceActor(self):
	
		if self.XYSliceActor is not None:		 
			self.ren.RemoveActor(self.XYSliceActor)
  
		self.window.Render() 
	
	# Visualize the XY slice based on checkbox value and xy slice number from GUI
	def visualizeXYSlice(self, value, isChecked):	

		if self.image_type == '.png':	
			self.imageXY.SetVOI(0, self.extent_x, 0, self.extent_y, value, value)
		else:			
			image = np.flipud(np.squeeze(self.muse_dataset[value,:,:,:]))
			image = self.adjust_contrast_zarr(image)
			self.vtk_img.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
			self.vtk_img.GetPointData().GetScalars().DeepCopy(numpy_support.numpy_to_vtk(np.reshape(image, (image.shape[1]*image.shape[0], 1))))
			self.XYSliceActor.SetPosition(-self.extent_x*self.pixel_size_xy / 2, -self.extent_y*self.pixel_size_xy / 2, (-self.extent_z *self.image_slice_thickness / 2) + (value*self.image_slice_thickness))

		if self.image_type == '.png':
			self.XYSliceActor.GetMapper().SetInputConnection(self.imageXY.GetOutputPort())
		else:
			self.XYSliceActor.GetMapper().SetInputData(self.vtk_img)

		if isChecked:
			self.ren.AddActor(self.XYSliceActor)
		else:
			self.ren.RemoveActor(self.XYSliceActor)

		self.window.Render()
		self.iren.ReInitialize()

	# Manipulate XY slice opacity
	def opacityXYSlice(self, value):
     
		if self.XYSliceActor is not None:
			self.XYSliceActor.GetProperty().SetOpacity(value / 100)

		self.window.Render()
      
	# Create the actor displaying the streamlines
	def createStreamlinesActor(self):

		# Poly data with lines and colors
		poly_data, color_is_scalar = lines_to_vtk_polydata(self.streamlines, self.color)

		self.streamline_poly_mapper = vtk.vtkPolyDataMapper()
		self.streamline_poly_mapper.SetInputData(poly_data)
		self.streamline_poly_mapper.ScalarVisibilityOn()
		self.streamline_poly_mapper.SetScalarModeToUsePointFieldData()
		self.streamline_poly_mapper.SelectColorArray("colors")
		self.streamline_poly_mapper.Update()

		self.streamlinesActor = vtk.vtkLODActor()
		self.streamlinesActor.SetNumberOfCloudPoints(10000)
		self.streamlinesActor.GetProperty().SetPointSize(3)
		self.streamlinesActor.SetPosition(-self.extent_x*self.pixel_size_xy / 2,
										  -self.extent_y*self.pixel_size_xy / 2,
										  -self.extent_z*self.image_slice_thickness / 2)
		
		self.streamlinesActor.SetMapper(self.streamline_poly_mapper)
		self.streamlinesActor.GetProperty().SetLineWidth(1)
		self.streamlinesActor.GetProperty().SetOpacity(0.7)

	# Remove the streamlines from the renderer if exists
	def removeStreamlinesActor(self):
	
		if self.streamlinesActor is not None:		 
			self.ren.RemoveActor(self.streamlinesActor)
   
		self.window.Render() 

	# Visualize streamlines based on checkbox value
	def visualizeStreamlines(self, isChecked):

		if isChecked:
			self.ren.AddActor(self.streamlinesActor)
		else:
			self.ren.RemoveActor(self.streamlinesActor)
		
		self.window.Render()
		self.iren.ReInitialize()

	# Manipulate streamline opacity
	def opacityStreamlines(self, value):

		if self.streamlinesActor is not None: 
			self.streamlinesActor.GetProperty().SetOpacity(value / 100)

		self.window.Render()
       
	# Utility function to get the most common color in a list of colors
	def mode_rows(self, array):
		array = np.ascontiguousarray(array)
		void_dt = np.dtype((np.void, array.dtype.itemsize * np.prod(array.shape[1:])))
		_,ids, count = np.unique(array.view(void_dt).ravel(), \
									return_index=1,return_counts=1)
		largest_count_id = ids[count.argmax()]
		most_frequent_row = array[largest_count_id]
		return most_frequent_row

	# Create an actor for the clusters
	def createClustersActor(self):

		if self.streamlineClusters is not None:
			# Poly data with lines and colors
			poly_data, color_is_scalar = lines_to_vtk_polydata(self.streamlineClusters.centroids, self.streamlineClustersColors)

			self.clusters_poly_mapper = vtk.vtkPolyDataMapper()
			self.clusters_poly_mapper.SetInputData(poly_data)
			self.clusters_poly_mapper.ScalarVisibilityOn()
			self.clusters_poly_mapper.SetScalarModeToUsePointFieldData()
			self.clusters_poly_mapper.SelectColorArray("colors")
			self.clusters_poly_mapper.Update()

			self.clustersActor = vtk.vtkLODActor()
			self.clustersActor.SetNumberOfCloudPoints(10000)
			self.clustersActor.GetProperty().SetPointSize(3)
			self.clustersActor.SetPosition(-self.extent_x*self.pixel_size_xy / 2,
							-self.extent_y*self.pixel_size_xy / 2,
							-self.extent_z*self.image_slice_thickness / 2)

			self.clustersActor.SetMapper(self.clusters_poly_mapper)
			self.clustersActor.GetProperty().SetLineWidth(3)
			self.clustersActor.GetProperty().SetOpacity(1)
	
	# Remove the clusters from the renderer if exists
	def removeClustersActor(self):
	
		if self.clustersActor is not None:		 
			self.ren.RemoveActor(self.clustersActor)   
   
		self.window.Render() 

	# Visualize clusters based on checkbox value
	def visualizeClusters(self, isChecked):

		if self.clustersActor is not None:
			if isChecked:
				self.ren.AddActor(self.clustersActor)
			else:
				self.ren.RemoveActor(self.clustersActor)
		
		self.window.Render()
		self.iren.ReInitialize()
     
	# Manipulate cluster opacity
	def opacityClusters(self, value):

		if self.clustersActor is not None:
			self.clustersActor.GetProperty().SetOpacity(value / 100)

		self.window.Render() 
  
	# Clip streamlines and clusters from current z slice
	def clipStreamlines(self, isChecked, value, z, ieTabSelected):

		if ieTabSelected:
			isChecked = True
			value = 5
   
		if self.streamlinesActor is not None:

			if isChecked:
				clipping_plane_origin_offset = -self.extent_z*self.image_slice_thickness / 2
				bottom_clipping_plane = vtk.vtkPlane()
				bottom_clipping_plane.SetOrigin(0, 0, clipping_plane_origin_offset + (np.max([0, z - value]) *self.image_slice_thickness))
				bottom_clipping_plane.SetNormal(0, 0, 1)

				top_clipping_plane = vtk.vtkPlane()
				top_clipping_plane.SetOrigin(0, 0, clipping_plane_origin_offset + (np.min([self.extent_z, z + value]) * self.image_slice_thickness)) 
				top_clipping_plane.SetNormal(0, 0, -1)

				self.streamlinesActor.GetMapper().RemoveAllClippingPlanes()
				self.streamlinesActor.GetMapper().AddClippingPlane(bottom_clipping_plane)
				self.streamlinesActor.GetMapper().AddClippingPlane(top_clipping_plane)		
			else:
				self.streamlinesActor.GetMapper().RemoveAllClippingPlanes()
   
		if self.clustersActor is not None:

			if isChecked:
				clipping_plane_origin_offset = -self.extent_z*self.image_slice_thickness / 2
				bottom_clipping_plane = vtk.vtkPlane()
				bottom_clipping_plane.SetOrigin(0, 0, clipping_plane_origin_offset + (z *self.image_slice_thickness))
				bottom_clipping_plane.SetNormal(0, 0, 1)

				top_clipping_plane = vtk.vtkPlane()
				top_clipping_plane.SetOrigin(0, 0, clipping_plane_origin_offset + (np.min([self.extent_z, z + value]) * self.image_slice_thickness)) 
				top_clipping_plane.SetNormal(0, 0, -1)

				self.clustersActor.GetMapper().RemoveAllClippingPlanes()
				self.clustersActor.GetMapper().AddClippingPlane(bottom_clipping_plane)
				self.clustersActor.GetMapper().AddClippingPlane(top_clipping_plane)		
			else:
				self.clustersActor.GetMapper().RemoveAllClippingPlanes()
   
		self.window.Render()
		self.iren.ReInitialize()

	# Sync variables from UI to SceneManager
	def updateStreamlinesAndColors(self, streamlines, color):
		self.streamlines = streamlines
		self.color = color
 
	# Sync variables from UI to SceneManager	
	def updateClusters(self, clusters, color):
		self.streamlineClusters = clusters
		self.streamlineClustersColors = color

	# Visualize streamlines by color
	def visualizeStreamlinesByColor(self, selected_colors, isChecked, streamlinesVisibilityCheckbox, opacity):
		
		if isChecked:			
			streamline_indices = [None] * selected_colors.shape[0]
			for i in np.arange(selected_colors.shape[0]):
				streamline_indices[i] = np.where(np.all(self.color == selected_colors[i], axis=1))

			indices_to_render = np.concatenate(streamline_indices, axis = 1)
			selected_streamlines = [self.streamlines[i] for i in list(indices_to_render[0])]

			poly_data, _ = lines_to_vtk_polydata(selected_streamlines, self.color[tuple(indices_to_render[0]),:])

			self.streamline_poly_mapper.SetInputData(poly_data)
			self.streamline_poly_mapper.Update()
   
			self.opacityStreamlines(opacity)

		self.visualizeStreamlines(streamlinesVisibilityCheckbox)

	# Visualize clusters by color
	def visualizeClustersByColor(self, selected_colors, isChecked, opacity):
   
		if isChecked and selected_colors is not None:   
			cluster_indices = [None] * selected_colors.shape[0]
			for i in np.arange(selected_colors.shape[0]):
				cluster_indices[i] = np.where(np.all(self.streamlineClustersColors == selected_colors[i], axis=1))

			indices_to_render = np.concatenate(cluster_indices, axis = 1)
			selected_clusters = [self.streamlineClusters.centroids[i] for i in list(indices_to_render[0])]     

			# Poly data with lines and colors
			poly_data, _ = lines_to_vtk_polydata(selected_clusters, self.streamlineClustersColors[tuple(indices_to_render[0]),:])

			self.clusters_poly_mapper.SetInputData(poly_data)
			self.clusters_poly_mapper.Update()
   
			self.opacityClusters(opacity)
    
		self.window.Render()

	# Export streamlines and colors to pickle file
	def exportStreamlinesAndColorsLK(self, windowSize, maxLevel, seedsPerPixel, gaussianSigma, sample_name):
	 
		pickle_files_save_folder = 'pickle files\\' + sample_name
		if not os.path.exists(pickle_files_save_folder):
			os.makedirs(pickle_files_save_folder)
   
		streamlinesFileName = pickle_files_save_folder + '\\streamlines_lk_' + datetime.datetime.now().strftime("%H%M%S_%m%d%Y") + "_win_" + str(windowSize) + "_maxLev_" + str(maxLevel) + "_spp_" + str(seedsPerPixel) + "_blur_" + str(gaussianSigma) + ".pkl"
		colorsFileName = pickle_files_save_folder + '\\colors_lk_' + datetime.datetime.now().strftime("%H%M%S_%m%d%Y") + "_win_" + str(windowSize) + "_maxLev_" + str(maxLevel) + "_spp_" + str(seedsPerPixel) + "_blur_" + str(gaussianSigma) + ".pkl"

		with open(streamlinesFileName, 'wb') as file:
			pickle.dump(self.streamlines, file)
		
		with open(colorsFileName, 'wb') as file:
			pickle.dump(self.color, file)
	
	# Export streamlines and colors to pickle file   
	def exportStreamlinesAndColorsST(self, neighborhoodScale, noiseScale, seedsPerPixel, downsampleFactor, sample_name):
	 	
		pickle_files_save_folder = 'pickle files\\' + sample_name
		if not os.path.exists(pickle_files_save_folder):
			os.makedirs(pickle_files_save_folder)
   
		streamlinesFileName = pickle_files_save_folder + '\\streamlines_st_' + datetime.datetime.now().strftime("%H%M%S_%m%d%Y") + "_neighborscale_" + str(neighborhoodScale) + "_noiseScale_" + str(noiseScale) + "_spp_" + str(seedsPerPixel) + "_ds_factor_" + str(downsampleFactor) + ".pkl"
		colorsFileName = pickle_files_save_folder + '\\colors_st_' + datetime.datetime.now().strftime("%H%M%S_%m%d%Y") + "_neighborscale_" + str(neighborhoodScale) + "_noiseScale_" + str(noiseScale) + "_spp_" + str(seedsPerPixel) + "_ds_factor_" + str(downsampleFactor) + ".pkl"

		with open(streamlinesFileName, 'wb') as file:
			pickle.dump(self.streamlines, file)
		
		with open(colorsFileName, 'wb') as file:
			pickle.dump(self.color, file)
   
	# Export clusters to pickle file
	def exportStreamlinesClustersLK(self, windowSize, maxLevel, seedsPerPixel, gaussianSigma, clusteringThreshold, sample_name):
		pickle_files_save_folder = 'pickle files\\' + sample_name
		if not os.path.exists(pickle_files_save_folder):
			os.makedirs(pickle_files_save_folder)
   
		streamlinesFileName = pickle_files_save_folder + '\\streamlinesClusters_lk_' + datetime.datetime.now().strftime("%H%M%S_%m%d%Y") + "_win_" + str(windowSize) + "_maxLev_" + str(maxLevel) + "_spp_" + str(seedsPerPixel) + "_blur_" + str(gaussianSigma) + "_thresh_" + str(clusteringThreshold) + ".pkl"
		colorsFileName = pickle_files_save_folder + '\\streamlinesClustersColors_lk_' + datetime.datetime.now().strftime("%H%M%S_%m%d%Y") + "_win_" + str(windowSize) + "_maxLev_" + str(maxLevel) + "_spp_" + str(seedsPerPixel) + "_blur_" + str(gaussianSigma) + "_thresh_" + str(clusteringThreshold) + ".pkl"

		with open(streamlinesFileName, 'wb') as file:
			pickle.dump(self.streamlineClusters.centroids, file)
		
		with open(colorsFileName, 'wb') as file:
			pickle.dump(self.streamlineClustersColors, file)

	# Export clusters to pickle file   
	def exportStreamlinesClustersST(self, neighborhoodScale, noiseScale, seedsPerPixel, downsampleFactor, clusteringThreshold, sample_name):
	 	
		pickle_files_save_folder = 'pickle files\\' + sample_name
		if not os.path.exists(pickle_files_save_folder):
			os.makedirs(pickle_files_save_folder)
   
		streamlinesFileName = pickle_files_save_folder + '\\streamlinesClusters_st_' + datetime.datetime.now().strftime("%H%M%S_%m%d%Y") + "_neighborscale_" + str(neighborhoodScale) + "_noiseScale_" + str(noiseScale) + "_spp_" + str(seedsPerPixel) + "_ds_factor_" + str(downsampleFactor) + "_thresh_" + str(clusteringThreshold) + ".pkl"
		colorsFileName = pickle_files_save_folder + '\\streamlinesClustersColors_st_' + datetime.datetime.now().strftime("%H%M%S_%m%d%Y") + "_neighborscale_" + str(neighborhoodScale) + "_noiseScale_" + str(noiseScale) + "_spp_" + str(seedsPerPixel) + "_ds_factor_" + str(downsampleFactor) + "_thresh_" + str(clusteringThreshold) + ".pkl"

		with open(streamlinesFileName, 'wb') as file:
			pickle.dump(self.streamlineClusters.centroids, file)
		
		with open(colorsFileName, 'wb') as file:
			pickle.dump(self.streamlineClustersColors, file)
   
   # Camera perspective toggle
	def toggleCameraParallelPerspective(self):
	   
		self.parallelView = 1 - self.parallelView
		camera = self.ren.GetActiveCamera()

		if self.parallelView:
			d = camera.GetDistance()
			a = camera.GetViewAngle()
			camera.SetParallelScale(d*math.tan(0.5*(a*math.pi/180)))
			camera.ParallelProjectionOn()
		else:
			d = camera.GetDistance()
			h = camera.GetParallelScale()
			camera.SetViewAngle(2.0*math.atan(h/d)*180/math.pi)
			camera.ParallelProjectionOff()

		self.window.Render()
  
	# Create BB actor
	def createBBActor(self):
		
		x_max = self.extent_x * self.pixel_size_xy
		y_max = self.extent_y * self.pixel_size_xy
		z_max = self.extent_z * self.image_slice_thickness

		bounding_box_coordinates = [np.asarray([[0, 0, 0],[x_max, 0, 0],[x_max, y_max, 0],[0, y_max, 0],[0, 0, 0]]),
							  		np.asarray([[0, 0, z_max],[x_max, 0, z_max],[x_max, y_max, z_max],[0, y_max, z_max],[0, 0, z_max]]),
							  		np.asarray([[0, 0, 0],[0, 0, z_max],[0, y_max, z_max],[0, y_max, 0],[0, 0, 0]]),
							  		np.asarray([[x_max, 0, 0],[x_max, 0, z_max],[x_max, y_max, z_max],[x_max, y_max, 0],[x_max, 0, 0]]),
		   							np.asarray([[0, y_max, 0],[x_max, y_max, 0],[x_max, y_max, z_max],[0, y_max, z_max],[0, y_max, 0]])]
		
		# Poly data with lines and colors
		bb_poly_data, _ = lines_to_vtk_polydata(bounding_box_coordinates, np.array([1, 0.64, 0]))

		bb_poly_mapper = vtk.vtkPolyDataMapper()
		bb_poly_mapper.SetInputData(bb_poly_data)
		bb_poly_mapper.ScalarVisibilityOn()
		bb_poly_mapper.SetScalarModeToUsePointFieldData()
		bb_poly_mapper.SelectColorArray("colors")
		bb_poly_mapper.Update()

		self.bbActor= vtk.vtkLODActor()
		self.bbActor.GetProperty().SetPointSize(3)
		self.bbActor.SetPosition(-self.extent_x*self.pixel_size_xy / 2,
							-self.extent_y*self.pixel_size_xy / 2,
							-self.extent_z*self.image_slice_thickness / 2)
		
		self.bbActor.SetMapper(bb_poly_mapper)
		self.bbActor.GetProperty().SetLineWidth(1)
		self.bbActor.GetProperty().SetOpacity(0.7)

	# Remove the BB actor from the renderer if exists
	def removeBBActor(self):
	
		if self.bbActor is not None:		 
			self.ren.RemoveActor(self.bbActor)
   		
		self.window.Render() 
   
	# Visualize BB actor
	def visualizeBoundingBox(self, enabled):
	 
		if self.bbActor is not None:
			if enabled:
				self.ren.AddActor(self.bbActor)
			else:
				self.ren.RemoveActor(self.bbActor)
		
		self.window.Render() 
   
	# Change camera properties and view when interactive editing tab is selected
	def interactiveEditingTabSelected(self):
       
		# We want parallel view in this window, the toggle function will toggle it.
		self.parallelView = 0
		self.toggleCameraParallelPerspective() 
  
  		# Visualize in the XY view
		self.SetViewXY()
  
  		# Change interactor style to image view
		self.iren.SetInteractorStyle(vtk.vtkInteractorStyleImage())  

	# When an ROI is drawn, save to disk
	def ROIDrawComplete(self, caller, event = None, calldata = None):
		
		# Get user drawn ROI as path
		path = self.contourWidget.GetContourRepresentation().GetContourRepresentationAsPolyData()			
		
		# Convert ROI to mask
		polyDataToImageStencil = vtk.vtkPolyDataToImageStencil()
		polyDataToImageStencil.SetTolerance(0)
		polyDataToImageStencil.SetInputData(path)
		polyDataToImageStencil.SetOutputOrigin(-self.extent_x* self.pixel_size_xy / 2, -self.extent_y * self.pixel_size_xy  / 2, 0)
		polyDataToImageStencil.SetOutputSpacing(self.pixel_size_xy, self.pixel_size_xy, self.image_slice_thickness)
		polyDataToImageStencil.SetOutputWholeExtent(0, self.extent_x, 0, self.extent_y, 0, 0)
		polyDataToImageStencil.Update()
		
		imageStencilToImage = vtk.vtkImageStencilToImage()
		imageStencilToImage.SetInputConnection(polyDataToImageStencil.GetOutputPort())
		imageStencilToImage.SetInsideValue(255)
		imageStencilToImage.Update()
		
		# Write user ROI to mask image, we will read from here into numpy array for later
		writer = vtk.vtkPNGWriter()
		writer.SetFileName('.\\masks\\user-selection.png')
		writer.SetInputConnection(imageStencilToImage.GetOutputPort())
		writer.Write()
 
	# When an ROI is drawn for a mask, update the image on disk	
	def MaskDrawUpdate(self, caller, event = None, calldata = None):
		
		# Get user drawn ROI as path
		path = self.contourWidget.GetContourRepresentation().GetContourRepresentationAsPolyData()			
		
		# Convert ROI to mask
		polyDataToImageStencil = vtk.vtkPolyDataToImageStencil()
		polyDataToImageStencil.SetTolerance(0)
		polyDataToImageStencil.SetInputData(path)
		polyDataToImageStencil.SetOutputOrigin(-self.extent_x* self.pixel_size_xy / 2, -self.extent_y * self.pixel_size_xy  / 2, 0)
		polyDataToImageStencil.SetOutputSpacing(self.pixel_size_xy, self.pixel_size_xy, self.image_slice_thickness)
		polyDataToImageStencil.SetOutputWholeExtent(0, self.extent_x, 0, self.extent_y, 0, 0)
		polyDataToImageStencil.Update()
		
		imageStencilToImage = vtk.vtkImageStencilToImage()
		imageStencilToImage.SetInputConnection(polyDataToImageStencil.GetOutputPort())
		imageStencilToImage.SetInsideValue(255)
		imageStencilToImage.Update()
		
		# Write user ROI to mask image, we will read from here into numpy array for later
		writer = vtk.vtkPNGWriter()
  
		# If the file doesn't exist, create one, else read from existing mask and update
		mask_path = '.\\masks\\mask-image-slice-' + str(self.mask_slice_index) + '.png'
  
		if not os.path.exists(mask_path):
			writer.SetFileName(mask_path)
			writer.SetInputConnection(imageStencilToImage.GetOutputPort())
			writer.Write()
   
		else:
			prev_image = np.array(Image.open(mask_path)).astype(np.uint8)
   
			temp = numpy_support.vtk_to_numpy(imageStencilToImage.GetOutput().GetPointData().GetScalars())
			dims = imageStencilToImage.GetOutput().GetDimensions()
			current_mask_roi = temp.reshape(dims[1], dims[0])
			current_mask_roi = np.flipud(current_mask_roi)
   
			updated_image = prev_image + current_mask_roi
			updated_image = Image.fromarray(updated_image)
			updated_image.save(mask_path)

	# Set up a contour tracing widget
	def tracerWidget(self, drawing_type, slice_index):
  
		# Set a variable to track which image slice is being updated (used for drawing mask, not ROI)
		self.mask_slice_index = slice_index

		# Change interactor style to image view
		self.iren.SetInteractorStyle(vtk.vtkInteractorStyleImage())  
  
		# Set up contour widget
		self.contourWidget = vtk.vtkContourWidget()
		contourRepresentation = vtk.vtkOrientedGlyphContourRepresentation()
		contourRepresentation.GetLinesProperty().SetColor([1, 0, 0])
		contourRepresentation.SetAlwaysOnTop(1)
		self.contourWidget.SetRepresentation(contourRepresentation)
  
		if drawing_type == 'roi':
			self.contourWidget.AddObserver(vtk.vtkCommand.EndInteractionEvent, self.ROIDrawComplete)	

		else:
			self.contourWidget.AddObserver(vtk.vtkCommand.EndInteractionEvent, self.MaskDrawUpdate)
   
		self.contourWidget.SetInteractor(self.iren) 

		# Switch on the contour widget
		self.contourWidget.On()
  
		# Render window
		#self.window.Render()

	# Remove a contour on exit from interactive editing tab
	def removeContour(self):
		self.contourWidget.Off()
		self.window.Render()

	# Change camera properties and view when visualization tab is selected
	def visualizationTabSelected(self):
	 
		# Change interactor style back to camera trackball
		self.iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

		# Turn off display of contour widget if it is on
		if self.contourWidget is not None:
			self.contourWidget.Off()
   
   		# We want perspective view in this window, the toggle function will toggle it.
		self.parallelView = 1
		self.toggleCameraParallelPerspective()
  
	# Window level adjustments from UI (left button mouse press and slide up/down and left/right)
	def windowLevelAdjustments(self, value, keepCurrentWinLev=False):
     
		if value:
			# We want parallel view in this window, the toggle function will toggle it.
			self.parallelView = 0
			self.toggleCameraParallelPerspective() 
	
			# Visualize in the XY view
			self.SetViewXY()

			# An object for the Interactor Style Image class, usable if we want its window level property in the future.
			# GetWindowLevelCurrentPosition for example
			self.vtkInteractorStyleImageObject = vtk.vtkInteractorStyleImage()
   
			# Change interactor style to image view
			self.iren.SetInteractorStyle(self.vtkInteractorStyleImageObject)  
   
		else:		
			
			if not keepCurrentWinLev:
				ip = vtk.vtkImageProperty()
				ip.SetColorWindow(255)
				ip.SetColorLevel(128)
				ip.SetAmbient(0.0)
				ip.SetDiffuse(1.0)
				ip.SetOpacity(1.0)
				ip.SetInterpolationTypeToLinear()

				self.XYSliceActor.SetProperty(ip)
				self.XYSliceActor.Update()
   
			# Change interactor style back to camera trackball
			self.iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

			# We want perspective view in this window, the toggle function will toggle it.
			self.parallelView = 1
			self.toggleCameraParallelPerspective()
   
			self.SetViewXY()

	def adjust_contrast_zarr(self, image):
		gamma = 0.75
		vmin = 0
		vmax = 2500

		image[image > vmax] = vmax
		image[image < vmin] = vmin

		image = cv2.normalize(image, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
		image = exposure.adjust_gamma(image, gamma = gamma) 
		image = np.floor(cv2.normalize(image, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)).astype('uint8')

		return image
