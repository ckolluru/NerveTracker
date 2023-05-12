import vtk
from vtkmodules.util import numpy_support
import datetime
import numpy as np
from vtk_helper_functions import numpy_to_vtk_points, numpy_to_vtk_cells, numpy_to_vtk_colors, lines_to_vtk_polydata
import pickle
import os
from PIL import Image
import math
import glob

class SceneManager:

	def __init__(self, vtkWidget=None):
		# self.ren = vtk.vtkRenderer()
		# self.window = vtkWidget.GetRenderWindow()
		# self.window.AddRenderer(self.ren)
		# self.iren = self.window.GetInteractor()
		# self.iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
		# self.iren.Initialize()  
  
		# The renderers, render window and interactor
		renderers = list()
		self.window = vtkWidget.GetRenderWindow()
		
		# The first renderer 
		renderers.append(vtk.vtkRenderer())
		self.window.AddRenderer(renderers[0])
  
		self.ren = vtk.vtkRenderer()
		renderers.append(self.ren)
		self.window.AddRenderer(self.ren)

		self.iren = self.window.GetInteractor()
		self.iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
		self.iren.Initialize()  

		# Layer 0 - background not transparent
		colors = vtk.vtkNamedColors()
		renderers[0].SetBackground(colors.GetColor3d("White"))
		renderers[0].SetLayer(0)
  
		# Layer 1 - the background is transparent
		#           so we only see the layer 0 background color
		renderers[1].SetLayer(1)

		#  We have two layers
		self.window.SetNumberOfLayers(2)    

		self.streamlinesActor = None

		self.streamlines = None
		self.color = None
		self.streamlineClusters = None
		self.streamlineClustersColors = None
		self.clustersActor = None
		self.parallelView = 0
  
		self.extent_x = None
		self.extent_y = None
		self.extent_z = None
		self.bbACtor = None

		# Create orientation axes
		axes = vtk.vtkAxesActor()
		axes.SetShaftTypeToCylinder()

		self.orient = vtk.vtkOrientationMarkerWidget()
		self.orient.SetOrientationMarker( axes )
		self.orient.SetInteractor( self.iren )
		self.orient.SetViewport( 0.0, 0.0, 0.2, 0.2 )
		self.orient.SetEnabled(1)		# Needed to set InteractiveOff
		self.orient.InteractiveOff()
		self.orient.SetEnabled(0)
  
		self.contourWidget = None

	def SetViewXY(self):		
		camera = self.ren.GetActiveCamera()
   
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

	def SetViewXZ(self):     
     
		camera = self.ren.GetActiveCamera()
   
		if self.parallelView:
			camera.SetFocalPoint(0, 0, 0)
			camera.SetViewUp(0, 0, 1)

			zd = self.extent_z * self.section_thickness
			d = camera.GetDistance()
			camera.SetParallelScale(0.5*zd)
			camera.SetPosition(0, d, 0)
		else:
			camera.SetFocalPoint(0, 0, 0)
			camera.SetViewUp(0, 0, 1)
   
			d = camera.GetDistance()
			h = 0.5 * self.extent_z * self.section_thickness
			camera.SetViewAngle(30)
			d = h / (math.tan(math.pi/12))
			camera.SetPosition(0, d, 0)
        
		camera.UpdateViewport(self.ren)
		self.iren.ReInitialize()   
		self.window.Render()

	def SetViewYZ(self):

		camera = self.ren.GetActiveCamera()

		if self.parallelView:
			camera.SetFocalPoint(0, 0, 0)
			camera.SetViewUp(0, 0, 1)

			zd = self.extent_z * self.section_thickness
			d = camera.GetDistance()
			camera.SetParallelScale(0.5*zd)
			camera.SetPosition(d, 0, 0)
		else:
			camera.SetFocalPoint(0, 0, 0)
			camera.SetViewUp(0, 0, 1)

			d = camera.GetDistance()
			h = 0.5 * self.extent_z * self.section_thickness
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

	def ToggleVisualizeAxis(self, visible):
		self.orient.SetEnabled(1)		# Needed to set InteractiveOff
		self.orient.InteractiveOff()
		self.orient.SetEnabled(visible)
		self.window.Render()

	def ToggleVisibility(self, visibility):
		# iterate through and set each visibility
		props = self.ren.GetViewProps()
		props.InitTraversal()
		for i in range(props.GetNumberOfItems()):
			props.GetNextProp().SetVisibility(visibility)
		
		self.window.Render()

	def addImageDataActor(self, imagesPath, pixel_size_xy, section_thickness, x_size_pixels, y_size_pixels, num_images_to_read):
		
		self.extent_x = x_size_pixels - 1
		self.extent_y = y_size_pixels - 1
		self.extent_z = num_images_to_read - 1

		self.pixel_size_xy = pixel_size_xy
		self.section_thickness = section_thickness

		# PNG (working)
		reader = vtk.vtkPNGReader()
		reader.SetFilePrefix(imagesPath + '\\') 
		reader.SetFilePattern('%sImage_%05d.png')
		reader.SetDataExtent(0, self.extent_x, 0, self.extent_y, 0, self.extent_z)
		reader.SetDataSpacing(pixel_size_xy, pixel_size_xy, section_thickness)
		reader.SetDataScalarTypeToUnsignedChar()
		reader.SetNumberOfScalarComponents(1)
  
		#VTK-HDF experiment
		# reader = vtk.vtkHDFReader()
		# reader.SetFileName(imagesPath + '\\stack.hdf')
		# reader.Update()    
		# outDS = reader.GetOutput()
		# outDS.GetPointData().SetScalars(outDS.GetPointData().GetArray("PNGImage"))

		#NRRD
		# filename = glob.glob(imagesPath + '\*.nrrd')
		# reader = vtk.vtkNrrdReader()
		# reader.SetFileName(filename[0]) 
		# reader.SetDataExtent(0, self.extent_x, 0, self.extent_y, 0, self.extent_z)
		# reader.SetDataSpacing(pixel_size_xy, pixel_size_xy, section_thickness)
		# reader.SetDataScalarTypeToUnsignedChar()
		# reader.SetNumberOfScalarComponents(1)
		# reader.SetFileDimensionality(3)
		# reader.FileLowerLeftOn()
   
		#NIFTI
		# reader = vtk.vtkNIFTIImageReader()
		# reader.SetFileName(imagesPath[0])
		# reader.SetDataExtent(0, self.extent_x, 0, self.extent_y, 0, self.extent_z)
		# reader.SetDataSpacing(pixel_size_xy, pixel_size_xy, section_thickness)
		# reader.FileLowerLeftOff()
		# reader.SetDataScalarTypeToUnsignedChar()
		# reader.SetNumberOfScalarComponents(1)
		# reader.SetFileDimensionality(3)

		# data_extent = reader.GetDataExtent()
		# dimensionality = reader.GetFileDimensionality()
		# type = reader.GetDataScalarType()
  
  
		#TIFF
		# reader = vtk.vtkTIFFReader()
		# reader.SetFileName(imagesPath[0])
		# reader.SetDataExtent(0, self.extent_x, 0, self.extent_y, 0, self.extent_z)
		# reader.SetDataSpacing(pixel_size_xy, pixel_size_xy, section_thickness)
		# reader.SpacingSpecifiedFlagOn()
		# reader.OriginSpecifiedFlagOn()
		# reader.SetOrientationType(3)
	
		self.imageXY = vtk.vtkExtractVOI()
		self.imageXY.SetInputConnection(reader.GetOutputPort())
		self.imageXY.SetVOI(0, self.extent_x, 0, self.extent_y, 0, 0)
		self.imageXY.Update()
  
		self.XYSliceActor = vtk.vtkImageActor()
		self.XYSliceActor.SetPosition(-self.extent_x*pixel_size_xy / 2, -self.extent_y*pixel_size_xy / 2, -self.extent_z*section_thickness / 2)
		self.XYSliceActor.GetMapper().SetInputConnection(self.imageXY.GetOutputPort())
		  
		ip = vtk.vtkImageProperty()
		ip.SetColorWindow(255)
		ip.SetColorLevel(128)
		ip.SetAmbient(0.0)
		ip.SetDiffuse(1.0)
		ip.SetOpacity(1.0)
		ip.SetInterpolationTypeToLinear()

		self.XYSliceActor.SetProperty(ip)
		self.XYSliceActor.Update()
	
	def addStreamlinesActor(self, streamlines, color):

		# Poly data with lines and colors
		poly_data, color_is_scalar = lines_to_vtk_polydata(streamlines, color)

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
										  -self.extent_z*self.section_thickness / 2)
		
		self.streamlinesActor.SetMapper(self.streamline_poly_mapper)
		self.streamlinesActor.GetProperty().SetLineWidth(1)
		self.streamlinesActor.GetProperty().SetOpacity(0.7)

		self.streamlines = streamlines
		self.color = color

	def removeStreamlinesActor(self):
	
		if self.streamlinesActor is not None:
		 
			self.ren.RemoveActor(self.streamlinesActor)
   
	def mode_rows(self, array):
		array = np.ascontiguousarray(array)
		void_dt = np.dtype((np.void, array.dtype.itemsize * np.prod(array.shape[1:])))
		_,ids, count = np.unique(array.view(void_dt).ravel(), \
									return_index=1,return_counts=1)
		largest_count_id = ids[count.argmax()]
		most_frequent_row = array[largest_count_id]
		return most_frequent_row

	def visualizeClusters(self, clusters, color, isChecked, streamlinesVisibilityChecked):
	 
		if isChecked:   
			clusterColors = np.empty((len(clusters.centroids), 3))

			for k in np.arange(len(clusters.centroids)):
				clusterColors[k,:] = self.mode_rows(color[clusters[k].indices, :])

			self.streamlineClusters = clusters.centroids
			self.streamlineClustersColors = clusterColors
   
			# Poly data with lines and colors
			poly_data, color_is_scalar = lines_to_vtk_polydata(clusters.centroids, clusterColors)

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
							-self.extent_z*self.section_thickness / 2)

			self.clustersActor.SetMapper(self.clusters_poly_mapper)
			self.clustersActor.GetProperty().SetLineWidth(3)
			self.clustersActor.GetProperty().SetOpacity(1)

			self.clustersActor.GetProperty().SetRenderLinesAsTubes(1)
			
			self.ren.AddActor(self.clustersActor)

		else:
			if self.clustersActor is not None:		 
				self.ren.RemoveActor(self.clustersActor)
		
		self.visualizeStreamlines(streamlinesVisibilityChecked)
		self.window.Render()
  
	def visualizeXYSlice(self, value, isChecked):		
		self.imageXY.SetVOI(0, self.extent_x, 0, self.extent_y, value, value)

		if isChecked:
			self.ren.AddActor(self.XYSliceActor)
		else:
			self.ren.RemoveActor(self.XYSliceActor)

		self.window.Render()
		self.window.Render()
		self.iren.ReInitialize()

	def opacityXYSlice(self, value):
		self.XYSliceActor.GetProperty().SetOpacity(value / 100)

		self.window.Render()

	def visualizeStreamlines(self, value):

		if value:
			self.ren.AddActor(self.streamlinesActor)
		else:
			self.ren.RemoveActor(self.streamlinesActor)
		
		self.window.Render()

	def opacityStreamlines(self, value):

		if self.streamlinesActor is not None: 
			self.streamlinesActor.GetProperty().SetOpacity(value / 100)

		self.window.Render()

	def opacityClusters(self, value):

		if self.clustersActor is not None:
			self.clustersActor.GetProperty().SetOpacity(value / 100)

		self.window.Render()

	def clipStreamlines(self, isChecked, value, z):

		if self.streamlinesActor is not None:

			if isChecked:
				clipping_plane_origin_offset = -self.extent_z*self.section_thickness / 2
				bottom_clipping_plane = vtk.vtkPlane()
				bottom_clipping_plane.SetOrigin(0, 0, clipping_plane_origin_offset + (np.max([0, z - value]) *self.section_thickness))
				bottom_clipping_plane.SetNormal(0, 0, 1)

				top_clipping_plane = vtk.vtkPlane()
				top_clipping_plane.SetOrigin(0, 0, clipping_plane_origin_offset + (np.min([self.extent_z, z + value]) * self.section_thickness)) 
				top_clipping_plane.SetNormal(0, 0, -1)

				self.streamlinesActor.GetMapper().RemoveAllClippingPlanes()
				self.streamlinesActor.GetMapper().AddClippingPlane(bottom_clipping_plane)
				self.streamlinesActor.GetMapper().AddClippingPlane(top_clipping_plane)		
			else:
				self.streamlinesActor.GetMapper().RemoveAllClippingPlanes()
	
		if self.clustersActor is not None:

			if isChecked:
				clipping_plane_origin_offset = -self.extent_z*self.section_thickness / 2
				bottom_clipping_plane = vtk.vtkPlane()
				bottom_clipping_plane.SetOrigin(0, 0, clipping_plane_origin_offset + (z *self.section_thickness))
				bottom_clipping_plane.SetNormal(0, 0, 1)

				top_clipping_plane = vtk.vtkPlane()
				top_clipping_plane.SetOrigin(0, 0, clipping_plane_origin_offset + (np.min([self.extent_z, z + value]) * self.section_thickness)) 
				top_clipping_plane.SetNormal(0, 0, -1)

				self.clustersActor.GetMapper().RemoveAllClippingPlanes()
				self.clustersActor.GetMapper().AddClippingPlane(bottom_clipping_plane)
				self.clustersActor.GetMapper().AddClippingPlane(top_clipping_plane)		
			else:
				self.clustersActor.GetMapper().RemoveAllClippingPlanes()

		self.window.Render()

	def updateStreamlinesAndColors(self, streamlines, colors):
		self.streamlines = streamlines
		self.color = colors
		
		poly_data, _ = lines_to_vtk_polydata(self.streamlines, self.color)
		self.streamline_poly_mapper.SetInputData(poly_data)
		self.streamline_poly_mapper.Update()

		self.window.Render()
  
	def showAllTracks(self, streamlinesVisibilityCheckbox):

		poly_data, _ = lines_to_vtk_polydata(self.streamlines, self.color)

		self.streamline_poly_mapper.SetInputData(poly_data)
		self.streamline_poly_mapper.Update()

		self.visualizeStreamlines(streamlinesVisibilityCheckbox)		

	def visualizeTracksByColor(self, selected_colors, isChecked, streamlinesVisibilityCheckbox):
		
		if not isChecked:
			self.visualizeStreamlines(streamlinesVisibilityCheckbox)
		
		else:			
			streamline_indices = [None] * selected_colors.shape[0]
			for i in np.arange(selected_colors.shape[0]):
				streamline_indices[i] = np.where(np.all(self.color == selected_colors[i], axis=1))

			indices_to_render = np.concatenate(streamline_indices, axis = 1)
			selected_streamlines = [self.streamlines[i] for i in list(indices_to_render[0])]

			poly_data, _ = lines_to_vtk_polydata(selected_streamlines,
															   self.color[tuple(indices_to_render[0]),:])

			self.streamline_poly_mapper.SetInputData(poly_data)
			self.streamline_poly_mapper.Update()

			self.visualizeStreamlines(streamlinesVisibilityCheckbox)

	def visualizeClustersByColor(self, selected_colors, isChecked, clustersVisibilityCheckbox):
     
		if not isChecked:
			if self.clustersActor is not None:		 
				self.ren.RemoveActor(self.clustersActor)
    
			self.window.Render()
   
		else:   
			if self.clustersActor is not None:
				cluster_indices = [None] * selected_colors.shape[0]
				for i in np.arange(selected_colors.shape[0]):
					cluster_indices[i] = np.where(np.all(self.streamlineClustersColors == selected_colors[i], axis=1))

				indices_to_render = np.concatenate(cluster_indices, axis = 1)
				selected_Clusters = [self.streamlineClusters[i] for i in list(indices_to_render[0])]     

				# Poly data with lines and colors
				poly_data, color_is_scalar = lines_to_vtk_polydata(selected_Clusters, self.streamlineClustersColors[tuple(indices_to_render[0]),:])

				self.clusters_poly_mapper.SetInputData(poly_data)
				self.clusters_poly_mapper.Update()
    
				if clustersVisibilityCheckbox:
					self.ren.AddActor(self.clustersActor)
				else:
					self.ren.RemoveActor(self.clustersActor)
     
				self.window.Render()

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
   
	def exportStreamlinesClustersLK(self, windowSize, maxLevel, seedsPerPixel, gaussianSigma, sample_name):
		pickle_files_save_folder = 'pickle files\\' + sample_name
		if not os.path.exists(pickle_files_save_folder):
			os.makedirs(pickle_files_save_folder)
   
		streamlinesFileName = pickle_files_save_folder + '\\streamlinesClusters_lk_' + datetime.datetime.now().strftime("%H%M%S_%m%d%Y") + "_win_" + str(windowSize) + "_maxLev_" + str(maxLevel) + "_spp_" + str(seedsPerPixel) + "_blur_" + str(gaussianSigma) + ".pkl"
		colorsFileName = pickle_files_save_folder + '\\streamlinesClustersColors_lk_' + datetime.datetime.now().strftime("%H%M%S_%m%d%Y") + "_win_" + str(windowSize) + "_maxLev_" + str(maxLevel) + "_spp_" + str(seedsPerPixel) + "_blur_" + str(gaussianSigma) + ".pkl"

		with open(streamlinesFileName, 'wb') as file:
			pickle.dump(self.streamlineClusters, file)
		
		with open(colorsFileName, 'wb') as file:
			pickle.dump(self.streamlineClustersColors, file)
   
	def exportStreamlinesClustersST(self, neighborhoodScale, noiseScale, seedsPerPixel, downsampleFactor, sample_name):
	 	
		pickle_files_save_folder = 'pickle files\\' + sample_name
		if not os.path.exists(pickle_files_save_folder):
			os.makedirs(pickle_files_save_folder)
   
		streamlinesFileName = pickle_files_save_folder + '\\streamlinesClusters_st_' + datetime.datetime.now().strftime("%H%M%S_%m%d%Y") + "_neighborscale_" + str(neighborhoodScale) + "_noiseScale_" + str(noiseScale) + "_spp_" + str(seedsPerPixel) + "_ds_factor_" + str(downsampleFactor) + ".pkl"
		colorsFileName = pickle_files_save_folder + '\\streamlinesClustersColors_st_' + datetime.datetime.now().strftime("%H%M%S_%m%d%Y") + "_neighborscale_" + str(neighborhoodScale) + "_noiseScale_" + str(noiseScale) + "_spp_" + str(seedsPerPixel) + "_ds_factor_" + str(downsampleFactor) + ".pkl"

		with open(streamlinesFileName, 'wb') as file:
			pickle.dump(self.streamlineClusters, file)
		
		with open(colorsFileName, 'wb') as file:
			pickle.dump(self.streamlineClustersColors, file)
   
	def updateStreamlines(self, streamlines):
		self.streamlines = streamlines

	def updateColors(self, color):
		self.color = color
		self.addStreamlinesActor(self.streamlines, self.color)
   
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
  
	def addBoundingBox(self, metadata):
		
		x_max = self.extent_x * self.pixel_size_xy
		y_max = self.extent_y * self.pixel_size_xy
		z_max = self.extent_z * self.section_thickness

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
							-self.extent_z*self.section_thickness / 2)
		
		self.bbActor.SetMapper(bb_poly_mapper)
		self.bbActor.GetProperty().SetLineWidth(1)
		self.bbActor.GetProperty().SetOpacity(0.7)
  
		self.ren.AddActor(self.bbActor)
		
		self.window.Render()
  
	def interactiveEditingTabSelected(self):
       
		# We want parallel view in this window, the toggle function will toggle it.
		self.parallelView = 0
		self.toggleCameraParallelPerspective() 
  
  		# Visualize in the XY view
		self.SetViewXY()
  
  		# Change interactor style to image view
		self.iren.SetInteractorStyle(vtk.vtkInteractorStyleImage())  

	def ROIDrawComplete(self, caller, event = None, calldata = None):
		
		# Get user drawn ROI as path
		path = self.contourWidget.GetContourRepresentation().GetContourRepresentationAsPolyData()			
		
		# Convert ROI to mask
		polyDataToImageStencil = vtk.vtkPolyDataToImageStencil()
		polyDataToImageStencil.SetTolerance(0)
		polyDataToImageStencil.SetInputData(path)
		polyDataToImageStencil.SetOutputOrigin(-self.extent_x* self.pixel_size_xy / 2, -self.extent_y * self.pixel_size_xy  / 2, 0)
		polyDataToImageStencil.SetOutputSpacing(self.pixel_size_xy, self.pixel_size_xy, self.section_thickness)
		polyDataToImageStencil.SetOutputWholeExtent(0, self.extent_x - 1, 0, self.extent_y - 1, 0, 0)
		polyDataToImageStencil.Update()
		
		imageStencilToImage = vtk.vtkImageStencilToImage()
		imageStencilToImage.SetInputConnection(polyDataToImageStencil.GetOutputPort())
		imageStencilToImage.SetInsideValue(255)
		imageStencilToImage.Update()
		
		# Write user ROI to mask image, we will read from here into numpy array for later
		writer = vtk.vtkPNGWriter()
		writer.SetFileName('user-selection.png')
		writer.SetInputConnection(imageStencilToImage.GetOutputPort())
		writer.Write()
 
	def tracerWidget(self):
  
		# Change interactor style to image view
		self.iren.SetInteractorStyle(vtk.vtkInteractorStyleImage())  
  
		# Set up contour widget
		self.contourWidget = vtk.vtkContourWidget()
		contourRepresentation = vtk.vtkOrientedGlyphContourRepresentation()
		contourRepresentation.GetLinesProperty().SetColor([1, 0, 0])
		contourRepresentation.SetAlwaysOnTop(1)
		self.contourWidget.SetRepresentation(contourRepresentation)
		self.contourWidget.AddObserver(vtk.vtkCommand.EndInteractionEvent, self.ROIDrawComplete)	
		self.contourWidget.SetInteractor(self.iren) 

		# Switch on the contour widget
		self.contourWidget.On()
  
		# Render window
		#self.window.Render()

	def removeContour(self):
		self.contourWidget.Off()
		self.window.Render()

	def visualizationTabSelected(self):
	 
		# Change interactor style back to camera trackball
		self.iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

		# Turn off display of contour widget if it is on
		if self.contourWidget is not None:
			self.contourWidget.Off()
   
   		# We want perspective view in this window, the toggle function will toggle it.
		self.parallelView = 1
		self.toggleCameraParallelPerspective()
  
	def windowLevelAdjustments(self, value):
     
		if value:
			# We want parallel view in this window, the toggle function will toggle it.
			self.parallelView = 0
			self.toggleCameraParallelPerspective() 
	
			# Visualize in the XY view
			self.SetViewXY()
	
			# Change interactor style to image view
			self.iren.SetInteractorStyle(vtk.vtkInteractorStyleImage())  
   
		else:		
			# Change interactor style back to camera trackball
			self.iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

			# We want perspective view in this window, the toggle function will toggle it.
			self.parallelView = 1
			self.toggleCameraParallelPerspective()
  
	def visualizeBoundingBox(self, enabled):
	 
		if self.bbActor is not None:
			if enabled:
				self.ren.AddActor(self.bbActor)
			else:
				self.ren.RemoveActor(self.bbActor)
		
		self.window.Render()