import sys
import os
import glob

from PyQt5.QtWidgets import QApplication, QWidget,QMainWindow,QFileDialog,QMessageBox,QSlider,QLineEdit,QCheckBox,QListWidget,QListWidgetItem,QAbstractItemView, QInputDialog, QTabWidget, QStyledItemDelegate, QPushButton, QRadioButton,QProgressBar
from PyQt5.QtWidgets import QApplication, QDialog, QLineEdit, QPushButton
from PyQt5.QtGui import QColor, QIcon, QPen
from PyQt5.QtWidgets import QStyle
from PyQt5.uic import loadUiType
from PyQt5.QtGui import QIntValidator
from PyQt5.QtCore import Qt

from scenemanager import SceneManager

import numpy as np
import scipy
import matplotlib.pyplot as plt
import zarr
from tqdm import tqdm

from modules.optical_flow_module import OpticFlowClass
from modules.structure_tensor_module import StructureTensorClass
from visualization.dialog_box_module import MetadataDialogBox, ValidationMetadataDialogBox
from tools.validate_tractograms import validate
from modules.act_module import ACTClass
from visualization.flythrough_module import MovieClass

import pickle
from dipy.segment.clustering import QuickBundles

import distinctipy
from PIL import Image
from tools.compare_tractograms import compare_tractogram_clusters
import ctypes

# Load GUI
Ui_MainWindow = loadUiType("mainwindow.ui")[0]

# Needed for drawing a border around selected items in a list of colors
class ColorDelegate(QStyledItemDelegate):
	def paint(self, painter, option, index):
		painter.save()

		# Call the base paint method to draw the default item background and foreground
		super().paint(painter, option, index)

		# If the item is selected, draw a red border around it
		if option.state & QStyle.State_Selected:
			color_rect = option.rect.adjusted(1, 1, -1, -1)
			painter.setPen(QPen(QColor(255, 0, 0), 3))
			painter.drawRect(color_rect)

		painter.restore()

# MainWindow class inheriting QMainWindow
class MainWindow(QMainWindow, Ui_MainWindow):

	def __init__(self, parent=None):
		super(MainWindow, self).__init__(parent)
		self.setupUi(self)
		self.initVTK()

		# Initialize variables
		self.imagesPath = None
		self.maskImageForSeedsFilePath = None
		self.fascicleSegmentationsPath = None
		self.metadata = dict()
		self.affine = None
		self.streamlines = None
		self.color = None
		self.maskImage = None
		self.trackingAlgoLK = 0
		self.trackingAlgoST = 0
		self.streamlineClusters = None
		self.streamlineClustersColors = None
		self.userSelectedColor = None
		self.startSliceIndex = None

		# Set current tab
		self.tabWidget.setCurrentWidget(self.visualizationTab)
		self.ieTabSelected = 0

		# Disable interactive editing tab until streamlines are created/loaded
		self.tabWidget.setTabEnabled(1, False)
  
		# Threads for optic flow analysis and structure tensor analysis
		self.opticFlowThread = None
		self.structureTensorThread = None
  
		# Validate inputs
		winSizeValidator = QIntValidator(3, 1000, self.windowSizeEdit)
		self.windowSizeEdit.setValidator(winSizeValidator)
	
	# We need to call QVTKWidget's show function before initializing the interactor
	def initVTK(self):

		self.show()		
		self.SceneManager = SceneManager(self.vtkContext)

	# Save the folder path to the images
	def OpenFolderImages(self):
		title = "Open Folder with PNG images"
		flags = QFileDialog.ShowDirsOnly
		self.imagesPath = QFileDialog.getExistingDirectory(self,
															title,
															os.path.expanduser("."),
															flags)

		if self.imagesPath == '':
			self.imagesPath = None
			return
		
		self.statusBar().showMessage('Image folder location saved', 2000)

	# Open and load the mask image to memory
	def OpenMaskImageForSeeds(self):
		title = "Open Mask Image with Regions For Seeds (corresponds to an arbitrary slice in image stack)"
		self.maskImageForSeedsFilePath = QFileDialog.getOpenFileName(self,
																	title,
																	os.path.expanduser("."),
																	"Mask File (*.png *.PNG)")
  
		if self.maskImageForSeedsFilePath[0] == '':
			self.maskImageForSeedsFilePath = None
			return

		self.LoadMaskImage()
		self.statusBar().showMessage('Mask image location saved', 2000)

	# Read metadata from a dialog box or pre-defined XML file
	def OpenImageMetadataXML(self):

		dialog = MetadataDialogBox()
		if dialog.exec_() == QDialog.Accepted:
			self.metadata = dialog.get_metadata()

		self.readMetadata()

	# Read metadata and create image data and bounding box actors
	def readMetadata(self):

		if self.imagesPath is None:
			msgBox = QMessageBox()
			msgBox.setText("Folder path for images is not set, please set it first.")
			msgBox.exec()
			return None

		# Read the first image in the directory and infer the image shape
		if self.metadata['image_type'] == '.png':
			filelist = glob.glob(self.imagesPath + '\\*' + self.metadata['image_type'])
			pil_image = Image.open(filelist[0])
			image = np.asarray(pil_image)
		
		else:
			dataset = zarr.open(self.imagesPath)
			muse_dataset = dataset['muse']
			image = np.squeeze(np.array(muse_dataset[0, 0, :, :]))   

		self.metadata['x_size_pixels'] = image.shape[1]
		self.metadata['y_size_pixels'] = image.shape[0]
  
		self.statusBar().showMessage('Reading metadata, creating XY image plane object for display..')
		self.progressBar.setMinimum(0)
		self.progressBar.setMaximum(self.metadata['num_images_to_read'])
  
		self.progressBar2.setMinimum(0)
		self.progressBar2.setMaximum(self.metadata['num_images_to_read'])
		
		self.affine = np.eye(4)
		self.affine[0, 0] = self.metadata['pixel_size_xy']
		self.affine[1, 1] = self.metadata['pixel_size_xy']
		self.affine[2, 2] = self.metadata['section_thickness']
			
		# Update slice view sliders and streamline clip slider
		self.xySlider.setMaximum(self.metadata['num_images_to_read'] - 1)
		self.xySlider2.setMaximum(self.metadata['num_images_to_read'] - 1)
		self.clipStreamlinesSlider.setMinimum(1)
		self.clipStreamlinesSlider.setMaximum(100)

		# Remove existing actors
		self.SceneManager.removeXYSliceActor()
		self.SceneManager.removeBBActor()
  
		# Add new actors for XY slice and BB
		self.SceneManager.createXYSliceActor(self.imagesPath, self.metadata['pixel_size_xy'],
											self.metadata['section_thickness'], self.metadata['x_size_pixels'],
											self.metadata['y_size_pixels'], self.metadata['num_images_to_read'], self.metadata['image_type'])
  
		self.SceneManager.createBBActor()

		validator = QIntValidator(0, self.metadata['num_images_to_read'] - 1, self.tracksStartingSliceIndex)
		self.tracksStartingSliceIndex.setValidator(validator)
  
		self.downsampleFactor = str(int(round(self.metadata['section_thickness'] / self.metadata['pixel_size_xy'])))
  
		self.statusBar().showMessage('Metadata reading complete', 2000)
  
	# Save path to the fascicle segmentation masks
	def OpenFascicleSegmentationsFolder(self):
		title = "Open Folder with Fascicle Segmentation Images"
		flags = QFileDialog.ShowDirsOnly
		self.fascicleSegmentationsPath = QFileDialog.getExistingDirectory(self,
																		title,
																		os.path.expanduser("."),
																		flags)
		if self.fascicleSegmentationsPath == '':
			self.fascicleSegmentationsPath = None
			return
		
		self.statusBar().showMessage('Fascicle segmentation folder location saved', 2000)		
  
	# Load the select mask image into memory
	def LoadMaskImage(self):

		if self.maskImageForSeedsFilePath is None:
			msgBox = QMessageBox()
			msgBox.setText("Mask Image path is not set, please set it first.")
			msgBox.exec()
			return None

		if self.metadata is None:
			msgBox = QMessageBox()
			msgBox.setText("Image metadata should be provided in a .xml file, please load it first.")
			msgBox.exec()
			return None

		self.maskImage = (plt.imread(self.maskImageForSeedsFilePath[0])* 255).astype('uint8')
		self.maskImage = self.maskImage.astype('uint8')
		self.maskImage[self.maskImage > 0] = 255
		
		self.statusBar().showMessage('Finished reading mask image for seeds into memory.', 2000)

	# Save streamlines and colors to the disk (pickle file)
	def exportStreamlinesAndColors(self):
	
		text, ok = QInputDialog.getText(self, 'Export streamlines and colors', 'Provide sample name')
		if ok:
			try:
				sample_name = str(text)
			except:
				msgBox = QMessageBox()
				msgBox.setText("Could not convert value entered into string, please try again.")
				msgBox.exec()
				return None

		# Sync variables first, then write
		self.SceneManager.updateStreamlinesAndColors(self.streamlines, self.color)
  
		if self.trackingAlgoLK:
			self.SceneManager.exportStreamlinesAndColorsLK(self.windowSizeEdit.text(), self.maxLevelEdit.text(), self.seedsPerPixelEdit.text(), self.blurEdit.text(), sample_name)
		if self.trackingAlgoST:
			self.SceneManager.exportStreamlinesAndColorsST(self.neighborhoodScaleEdit.text(), self.noiseScaleEdit.text(), self.seedsPerPixelEdit.text(), self.downsampleFactor, sample_name)

		self.statusBar().showMessage('Saved streamlines and colors data into pickle files folder.', 2000)

	# Save streamline cluster to the disk (pickle file)
	def exportStreamlinesClusters(self):

		text, ok = QInputDialog.getText(self, 'Export streamlines clusters', 'Provide sample name')
		if ok:
			try:
				sample_name = str(text)
			except:
				msgBox = QMessageBox()
				msgBox.setText("Could not convert value entered into string, please try again.")
				msgBox.exec()
				return None
 
		# Sync variables first, then write
		self.SceneManager.updateClusters(self.streamlineClusters, self.streamlineClustersColors)
  
		if self.trackingAlgoLK:
			self.SceneManager.exportStreamlinesClustersLK(self.windowSizeEdit.text(), self.maxLevelEdit.text(), self.seedsPerPixelEdit.text(), self.blurEdit.text(), self.clusteringThresholdEdit.text(), sample_name)
		if self.trackingAlgoST:
			self.SceneManager.exportStreamlinesClustersST(self.neighborhoodScaleEdit.text(), self.noiseScaleEdit.text(), self.seedsPerPixelEdit.text(), self.downsampleFactor, self.clusteringThresholdEdit.text(), sample_name)

		self.statusBar().showMessage('Saved streamline clusters into pickle files folder.', 2000)
	
	# Load streamlines from pickle file
	def loadStreamlines(self):

		title = "Open Streamlines Pickle file, needs colors pickle file in the same folder."
		self.streamlinesPickleFile = QFileDialog.getOpenFileName(self,
										title,
										os.path.expanduser("."),
										"Pickle Files (*.pkl)")

		if self.streamlinesPickleFile[0] == '':
			self.streamlinesPickleFile = None
			return

		# Find the colors pickle file in the same folder
		colorsFileName = 'colors_' + os.path.basename(self.streamlinesPickleFile[0])[12:]
		self.colorsPickleFile = os.path.dirname(self.streamlinesPickleFile[0]) + '\\' + colorsFileName
  
		if not os.path.exists(self.colorsPickleFile):
			msgBox = QMessageBox()
			msgBox.setText("Did not find the colors pickle file in the same folder, please verify and try again.")
			msgBox.exec()
			return None

		# Read the streamline pickle file and sync variables here and in SceneManager
		with open(self.streamlinesPickleFile[0], 'rb') as f:
			self.streamlines = pickle.load(f) 
  
		# Read the colors pickle file and sync variables here and in SceneManager
		with open(self.colorsPickleFile, 'rb') as f:
			self.color = pickle.load(f)

		self.SceneManager.removeStreamlinesActor()
		self.SceneManager.updateStreamlinesAndColors(self.streamlines, self.color)
		self.SceneManager.createStreamlinesActor()
		self.SceneManager.visualizeStreamlines(self.streamlinesVisibilityCheckbox.isChecked())
		self.statusBar().showMessage('Loading streamlines complete', 2000)  
  
		# Clear and update the colors list widget
		self.selectTracksListWidget.clear()
		self.unique_colors = np.unique(self.color, axis = 0)
  
		for i in np.arange(self.unique_colors.shape[0]):
			listItem = QListWidgetItem('')
			listItem.setBackground(QColor(int(self.unique_colors[i][0] * 255), int(self.unique_colors[i][1] * 255),
										  int(self.unique_colors[i][2] * 255), 200))
			self.selectTracksListWidget.addItem(listItem)

		self.selectTracksListWidget.setItemDelegate(ColorDelegate(self.selectTracksListWidget))
		self.selectTracksListWidget.setSelectionMode(QAbstractItemView.MultiSelection)
  
		self.statusBar().showMessage('Loading colors complete, can display streamlines', 2000) 
 
 		# Enable the interactive editing tab and update visualization
		if self.color is not None and self.streamlines is not None:

			self.tabWidget.setTabEnabled(1, True)  
   
			self.uncheckStreamlinesUIElements()
  
		# Set flags to identify which tracking algorithm was used
		if '_st_' in self.streamlinesPickleFile[0]:
			self.trackingAlgoST = 1
			self.trackingAlgoLK = 0
		else:
			self.trackingAlgoLK = 1 
			self.trackingAlgoST = 0

	# Uncheck streamline specific UI elements to start afresh
	def uncheckStreamlinesUIElements(self):
	 
		self.clipStreamlinesCheckbox.setChecked(False)
		self.selectTracksByColorCheckbox.setChecked(False)
		self.clusterCheckbox.setChecked(False)
   
	# Visualize bounding box, checkbox signal from UI
	def visualizeBoundingBox(self, value):
		
		self.SceneManager.visualizeBoundingBox(value)
  
	# Connect the forward/backward radio button group's buttonReleased() signal to this function
 	# Checks if at least one box is checked
	def trackingDirection(self):
		if not self.forwardTrackingButton.isChecked() and not self.backwardTrackingButton.isChecked():
			self.sender().setChecked(True)
   
		if not self.forwardTrackingButton2.isChecked() and not self.backwardTrackingButton2.isChecked():
			self.sender().setChecked(True)
   
	# Cluster the streamlines and render them in the window
	def clusterStreamlines(self, isChecked):
	 	
		if isChecked: 
			if self.streamlines is None:
				msgBox = QMessageBox()
				msgBox.setText("Need streamlines to be computed/loaded prior to clusering.")
				msgBox.exec()
				return None    

			self.statusBar().showMessage('Running quick bundles clustering..')   

			# Do not consider streamlines that do not cover full length of the stack, else short streamlines will get their own clusters
			# Computational modeling considers streamlines over the full length of the stack
			streamline_lengths = [streamline.shape[0] for streamline in self.streamlines]

			# Length of the full stack is the most commonly occuring streamline length
			full_length = scipy.stats.mode(streamline_lengths, keepdims=False)[0]
   
			# Create a new streamlines variable that only contains streamlines across the full length.
			streamlines_for_clustering = []
			colors_for_clustering = []
			for k in range(len(self.streamlines)):
				if self.streamlines[k].shape[0] == full_length:
					streamlines_for_clustering.append(self.streamlines[k])
					colors_for_clustering.append(self.color[k])
   
			colors_for_clustering = np.array(colors_for_clustering)
   
			# Larger threshold, fewer clusters
			qb = QuickBundles(threshold = float(self.clusteringThresholdEdit.text()))
			self.streamlineClusters = qb.cluster(streamlines_for_clustering)
			self.streamlineClustersColors = np.empty((len(self.streamlineClusters.centroids), 3))
   
			for k in np.arange(len(self.streamlineClusters.centroids)):
				self.streamlineClustersColors[k,:] = self.SceneManager.mode_rows(colors_for_clustering[self.streamlineClusters[k].indices, :])

			# Synchronize variables, remove actor, create actor, visualize
			self.SceneManager.removeClustersActor()
			self.SceneManager.updateClusters(self.streamlineClusters, self.streamlineClustersColors)
			self.SceneManager.createClustersActor()
   
		self.SceneManager.visualizeClusters(isChecked)
		self.statusBar().showMessage('Creating and visualizing cluster bundles complete.', 2000)
	
	# Progress bar update from threads
	def progressUpdate(self, value):
		self.progressBar.setValue(value)  
		self.progressBar2.setValue(value)  
 
	# Progress bar minimum value update from threads 
	def progressMinimum(self, value):
		self.progressBar.setMinimum(value)
		self.progressBar2.setMinimum(value)
  
	# Progress bar maximum value update from threads 
	def progressMaximum(self, value):
		self.progressBar.setMaximum(value)
		self.progressBar2.setMaximum(value)
  
	# Set status bar from threads
	def statusBarMessage(self, string):
		self.statusBar().showMessage(string)

	# Tracking complete signal slot mechanism from threads, sync variables with Scenemanager, update actors
	def trackingComplete(self, value):
	 
		if value:
			if value == 1:
				self.streamlines, self.color = self.opticFlowThread.get_streamlines_and_colors()
				self.opticFlowThread.terminate_thread()
			if value == 2:
				self.streamlines, self.color = self.structureTensorThread.get_streamlines_and_colors()
				self.structureTensorThread.terminate_thread()
  		
			if self.streamlines is None and self.color is None:
				msgBox = QMessageBox()
				msgBox.setText('Could not read image files, possible error in metadata file for image size fields or images not present in specified path')
				msgBox.exec()
				return None
			
			self.SceneManager.removeStreamlinesActor()
			self.SceneManager.updateStreamlinesAndColors(self.streamlines, self.color)
			self.SceneManager.createStreamlinesActor()
			self.SceneManager.visualizeStreamlines(self.streamlinesVisibilityCheckbox.isChecked())

			self.unique_colors = np.unique(self.color, axis = 0)
			self.selectTracksListWidget.clear()
	
			for i in np.arange(self.unique_colors.shape[0]):
				listItem = QListWidgetItem('')
				listItem.setBackground(QColor(int(self.unique_colors[i][0] * 255), int(self.unique_colors[i][1] * 255),
											int(self.unique_colors[i][2] * 255), 200))
				self.selectTracksListWidget.addItem(listItem)

			self.selectTracksListWidget.setItemDelegate(ColorDelegate(self.selectTracksListWidget))
			self.selectTracksListWidget.setSelectionMode(QAbstractItemView.MultiSelection)
			self.statusBar().showMessage('Tracks calculated', 5000)
	
			if value == 1:
				self.trackingAlgoST = 0
				self.trackingAlgoLK = 1
			else:
				self.trackingAlgoLK = 0
				self.trackingAlgoST = 1
	
			# Enable the interactive editing tab 
			if self.color is not None and self.streamlines is not None:
				self.tabWidget.setTabEnabled(1, True)
	
			# Uncheck UI elements to start afresh
			self.uncheckStreamlinesUIElements()
   
			# Enable the push buttons back   
			self.computeTracksLKButton.setEnabled(True)
			self.computeTracksSTButton.setEnabled(True)
			self.anatomicallyConstrainStreamlinesButton.setEnabled(True)

	# Compute tracks with Lucas Kanade slot
	def computeTracksLK(self):
		try:
			windowSize = int(self.windowSizeEdit.text())
			maxLevel = int(self.maxLevelEdit.text())
			seedsPerPixel = float(self.seedsPerPixelEdit.text())
			blur = float(self.blurEdit.text())
		except:	
			msgBox = QMessageBox()
			msgBox.setText("Incorrect value for window size and max. level and seeds per pixel, expect integers/floats.")
			msgBox.exec()
			return None

		if self.maskImage is None:
			msgBox = QMessageBox()
			msgBox.setText("Need to load mask image into memory prior to computing tracks, please click that button.")
			msgBox.exec()
			return None

		if self.imagesPath is None:
			msgBox = QMessageBox()
			msgBox.setText("Folder path for images is not set, please set it first.")
			msgBox.exec()
			return None

		if len(self.metadata) == 0:
			msgBox = QMessageBox()
			msgBox.setText("Metadata is not set, please set it first.")
			msgBox.exec()
			return None

		self.opticFlowThread = OpticFlowClass(self.imagesPath, self.maskImage,
											  self.affine, self.metadata, windowSize, maxLevel,
											  seedsPerPixel, blur, int(self.tracksStartingSliceIndex.text()),
			 								  self.forwardTrackingButton.isChecked(), self.backwardTrackingButton.isChecked())
		self.opticFlowThread.progressSignal.connect(self.progressUpdate)
		self.opticFlowThread.progressMinimumSignal.connect(self.progressMinimum)
		self.opticFlowThread.progressMaximumSignal.connect(self.progressMaximum)
		self.opticFlowThread.completeSignal.connect(self.trackingComplete)
		self.opticFlowThread.statusBarSignal.connect(self.statusBarMessage)
		self.computeTracksLKButton.setEnabled(False)
		self.computeTracksSTButton.setEnabled(False)
		self.anatomicallyConstrainStreamlinesButton.setEnabled(False)
		self.opticFlowThread.start()
  
	# Compute tracks with Structure tensor slot
	def computeTracksST(self):
		try:
			neighborhoodScale = int(self.neighborhoodScaleEdit.text())
			noiseScale = float(self.noiseScaleEdit.text())
			seedsPerPixel = float(self.seedsPerPixelEdit.text())
			downsampleFactor = int(self.downsampleFactor)
		except:	
			msgBox = QMessageBox()
			msgBox.setText("Incorrect value for window size and max. level and seeds per pixel, expect integers/floats.")
			msgBox.exec()
			return None

		if self.maskImage is None:
			msgBox = QMessageBox()
			msgBox.setText("Need to load mask image into memory prior to computing tracks, please click that button.")
			msgBox.exec()
			return None

		if self.imagesPath is None:
			msgBox = QMessageBox()
			msgBox.setText("Folder path for images is not set, please set it first.")
			msgBox.exec()
			return None

		if len(self.metadata) == 0:
			msgBox = QMessageBox()
			msgBox.setText("Metadata is not set, please set it first.")
			msgBox.exec()
			return None
 
		self.structureTensorThread = StructureTensorClass(self.imagesPath, self.maskImage,
														self.affine, self.metadata, neighborhoodScale, noiseScale,
														seedsPerPixel, downsampleFactor, int(self.tracksStartingSliceIndex.text()),
														self.forwardTrackingButton.isChecked(), self.backwardTrackingButton.isChecked())
		self.structureTensorThread.progressSignal.connect(self.progressUpdate)
		self.structureTensorThread.progressMinimumSignal.connect(self.progressMinimum)
		self.structureTensorThread.progressMaximumSignal.connect(self.progressMaximum)
		self.structureTensorThread.completeSignal.connect(self.trackingComplete)
		self.structureTensorThread.statusBarSignal.connect(self.statusBarMessage)
		self.computeTracksLKButton.setEnabled(False)
		self.computeTracksSTButton.setEnabled(False)
		self.anatomicallyConstrainStreamlinesButton.setEnabled(False)
		self.structureTensorThread.start()
	
	# Visualize streamlines checkbox slot
	def visualizeStreamlines(self, value):

		if self.SceneManager.streamlinesActor is None:
			msgBox = QMessageBox()
			msgBox.setText("Streamlines not created, click compute tracks button or load existing streamlines file.")
			msgBox.exec()
			self.streamlinesVisibilityCheckbox.setChecked(False)
			return None

		self.SceneManager.visualizeStreamlines(value)
	
	# Opacity of streamlines modification slot
	def streamlinesOpacity(self, value):

		if self.SceneManager.streamlinesActor is None:
			msgBox = QMessageBox()
			msgBox.setText("Streamlines not created, click compute tracks button or load existing streamlines file.")
			msgBox.exec()
			return None

		self.SceneManager.opacityStreamlines(value)
		self.SceneManager.opacityClusters(value)

	# Streamline clipping slot
	def clipStreamlines(self, isChecked):

		if self.SceneManager.streamlinesActor is None:
			msgBox = QMessageBox()
			msgBox.setText("Streamlines not created, click compute tracks button or load existing streamlines file.")
			msgBox.exec()
			return None

		self.SceneManager.clipStreamlines(isChecked, self.clipStreamlinesSlider.value(), self.xySlider.value(), self.ieTabSelected)

	# Streamline clipping slider update slot
	def clipStreamlinesSliderUpdate(self, value):
		self.clipStreamlinesEdit.setText(str(value))

		self.SceneManager.clipStreamlines(self.clipStreamlinesCheckbox.isChecked(), value, self.xySlider.value(), self.ieTabSelected)

	# Streamline clipping edit slot
	def clipStreamlinesEdit_changed(self):
		try:
			self.clipStreamlinesSlider.setValue(int(self.clipStreamlinesEdit.text()))
		except:			
			msgBox = QMessageBox()
			msgBox.setText("Incorrect value for clip streamlines, expect integer")
			msgBox.exec()
			return None

		self.SceneManager.clipStreamlines(self.clipStreamlinesCheckbox.isChecked(), int(self.clipStreamlinesEdit.text()), self.xySlider.value(), self.ieTabSelected)

	# Visualize streamlines by selected colors
	def visualizeStreamlinesByColor(self, isChecked):

		if isChecked:
			selected_indices = self.selectTracksListWidget.selectionModel().selectedIndexes()

			if len(selected_indices) == 0:	
				msgBox = QMessageBox()
				msgBox.setWindowTitle("Error")
				msgBox.setText("Select streamline colors to keep from list")
				msgBox.exec()
				self.selectTracksByColorCheckbox.setChecked(False)
				return None

			unique_colors = np.unique(self.color, axis = 0)
		
			selected_colors = np.zeros((len(selected_indices), 3))

			for i in np.arange(len(selected_indices)):
				selected_colors[i,:] = unique_colors[int(selected_indices[i].row()), :]    
	
			self.SceneManager.visualizeStreamlinesByColor(selected_colors, isChecked, self.streamlinesVisibilityCheckbox.isChecked(), self.streamlinesOpacitySlider.value())
			self.SceneManager.visualizeClustersByColor(selected_colors, self.clusterCheckbox.isChecked(), self.streamlinesOpacitySlider.value())
			
		else:
			self.SceneManager.removeStreamlinesActor()
			self.SceneManager.updateStreamlinesAndColors(self.streamlines, self.color)
			self.SceneManager.createStreamlinesActor()
			self.SceneManager.visualizeStreamlines(self.streamlinesVisibilityCheckbox.isChecked())
   
			self.SceneManager.removeClustersActor()
			self.SceneManager.updateClusters(self.streamlineClusters, self.streamlineClustersColors)
			self.SceneManager.createClustersActor()
			self.SceneManager.visualizeClusters(self.clusterCheckbox.isChecked())

			self.selectTracksListWidget.selectionModel().clear()
  
	# Update color list widget in interactive editing tab, sync XY slice, show streamlines and clip
	def interactiveEditingTabSelect(self, tabIndex):
	 
		if tabIndex == 1:
			self.ieTabSelected = True
			self.SceneManager.interactiveEditingTabSelected()	
   
			self.pickColorsListWidget.clear()
	
			for i in np.arange(self.unique_colors.shape[0]):
				listItem = QListWidgetItem('')
				listItem.setBackground(QColor(int(self.unique_colors[i][0] * 255), int(self.unique_colors[i][1] * 255),
											int(self.unique_colors[i][2] * 255), 200))
				self.pickColorsListWidget.addItem(listItem)

			self.pickColorsListWidget.setSelectionMode(QAbstractItemView.SingleSelection)

			current_slice_num = int(self.XYSliceEdit.text())
			self.streamlinesVisibilityCheckbox.setChecked(True)
			self.SceneManager.visualizeXYSlice(current_slice_num, self.streamlinesVisibilityCheckbox.isChecked())	
			self.SceneManager.visualizeStreamlines(self.streamlinesVisibilityCheckbox.isChecked())	
			self.clusterCheckbox.setChecked(False)
			self.selectTracksByColorCheckbox.setChecked(False)
   
			# Always clip streamlines by 5 slices in interactive editing tab
			self.SceneManager.clipStreamlines(self.streamlinesVisibilityCheckbox.isChecked(), 5, current_slice_num, self.ieTabSelected)
   
		if tabIndex == 0:
			self.ieTabSelected = False
			self.SceneManager.visualizationTabSelected()
			current_slice_num = int(self.XYSliceEdit2.text())
   
			# Clip as desired by the user
			self.SceneManager.clipStreamlines(self.clipStreamlinesCheckbox.isChecked(), self.clipStreamlinesSlider.value(), current_slice_num, self.ieTabSelected)

	# Draw ROI slot, creates the tracer widget
	def drawROI(self):
		self.startSliceIndex = int(self.XYSliceEdit2.text())
		self.SceneManager.tracerWidget()
		self.statusBar().showMessage('Draw a closed contour.', 3000)	

	# Create a new color and add to list
	def addNewColorROI(self):
		
		# Get a new color that is different from all existing colors
		unique_colors_transpose = self.unique_colors.T
		current_colors_as_list_of_tuples = list(zip(unique_colors_transpose[0], unique_colors_transpose[1], unique_colors_transpose[2]))
		current_colors_as_list_of_tuples.append((0, 0, 0))
		current_colors_as_list_of_tuples.append((1, 1, 1))
  
		self.userSelectedColor = distinctipy.get_colors(1, current_colors_as_list_of_tuples)	
		self.statusBar().showMessage('New color created, track with LK or ST!', 3000)	
  
	# Pick a color for streamlines from the ROI
	def pickColorROI(self, listWidgetItem):
		 
		selected_indices = self.pickColorsListWidget.selectionModel().selectedIndexes()

		if not len(selected_indices) == 1:	
			msgBox = QMessageBox()
			msgBox.setText("More than one color picked from list, please select only one.")
			msgBox.exec()
			return None

		self.userSelectedColor = np.expand_dims(self.unique_colors[int(selected_indices[0].row()), :], axis = 1).T
		self.statusBar().showMessage('Color pick complete, track with LK or ST!', 3000)	
  
	# Tracking ROI complete signal
	def trackingROIComplete(self, value):

		if value:
			if value == 1:
				streamlines, _ = self.opticFlowThread.get_streamlines_and_colors()
				self.opticFlowThread.terminate_thread()
			if value == 2:
				streamlines, _  = self.structureTensorThread.get_streamlines_and_colors()
				self.structureTensorThread.terminate_thread()
  		
			if self.streamlines is None and self.color is None:
				msgBox = QMessageBox()
				msgBox.setText('Could not read image files, possible error in metadata file for image size fields or images not present in specified path')
				msgBox.exec()
				return None
			
			# Concatenate new streamlines to existing
			self.streamlines.extend(streamlines)

			# Get user selected color and update self.color
			color = np.repeat(self.userSelectedColor, len(streamlines), axis = 0)
			self.color = np.concatenate((self.color, color), axis = 0)
	
			self.SceneManager.updateStreamlinesAndColors(self.streamlines, self.color)
			self.SceneManager.removeStreamlinesActor()
			self.SceneManager.createStreamlinesActor()
			self.SceneManager.visualizeStreamlines(self.streamlinesVisibilityCheckbox.isChecked())

			self.unique_colors = np.unique(self.color, axis = 0)

			self.selectTracksListWidget.clear()
	
			for i in np.arange(self.unique_colors.shape[0]):
				listItem = QListWidgetItem('')
				listItem.setBackground(QColor(int(self.unique_colors[i][0] * 255), int(self.unique_colors[i][1] * 255),
											int(self.unique_colors[i][2] * 255), 200))
				self.selectTracksListWidget.addItem(listItem)

			self.selectTracksListWidget.setItemDelegate(ColorDelegate(self.selectTracksListWidget))
			self.selectTracksListWidget.setSelectionMode(QAbstractItemView.MultiSelection)

			self.pickColorsListWidget.clear()
		
			for i in np.arange(self.unique_colors.shape[0]):
				listItem = QListWidgetItem('')
				listItem.setBackground(QColor(int(self.unique_colors[i][0] * 255), int(self.unique_colors[i][1] * 255),
											int(self.unique_colors[i][2] * 255), 200))
				self.pickColorsListWidget.addItem(listItem)

			self.pickColorsListWidget.setSelectionMode(QAbstractItemView.SingleSelection)
			self.statusBar().showMessage('Tracks calculated', 5000)
		
			if value == 1:
				self.trackingAlgoST = 0
				self.trackingAlgoLK = 1
			else:
				self.trackingAlgoLK = 0
				self.trackingAlgoST = 1

			# Remove contour
			self.SceneManager.removeContour()

			# Visualize streamlines
			self.SceneManager.visualizeStreamlines(self.streamlinesVisibilityCheckbox.isChecked())
			
			# Enable push buttons back again
			self.trackROI_LKButton.setEnabled(True)
			self.trackROI_STButton.setEnabled(True)
   
	# Track with optic flow, ROI only
	def trackROI_LK(self):

		try:
			windowSize = int(self.windowSizeEdit.text())
			maxLevel = int(self.maxLevelEdit.text())
			seedsPerPixel = float(self.seedsPerPixelEdit.text())
			blur = float(self.blurEdit.text())
		except:	
			msgBox = QMessageBox()
			msgBox.setText("Incorrect value for window size and max. level and seeds per pixel, expect integers/floats.")
			msgBox.exec()
			return None

		if self.imagesPath is None:
			msgBox = QMessageBox()
			msgBox.setText("Folder path for images is not set, please set it first.")
			msgBox.exec()
			return None

		if len(self.metadata) == 0:
			msgBox = QMessageBox()
			msgBox.setText("Metadata is not set, please set it first.")
			msgBox.exec()
			return None

		if self.userSelectedColor is None:
			msgBox = QMessageBox()
			msgBox.setText("Color not selected, please select from list or click Add a new color button.")
			msgBox.exec()
			return None    
  
		# Start tracking from this slice and use image ROI as mask
		mask_image = Image.open('user-selection.png')
		mask_image = np.asarray(mask_image)

		self.opticFlowThread = OpticFlowClass(self.imagesPath, mask_image,
											  self.affine, self.metadata, windowSize, maxLevel,
											  seedsPerPixel, blur, self.startSliceIndex,
											  self.forwardTrackingButton2.isChecked(), self.backwardTrackingButton2.isChecked())
		self.opticFlowThread.progressSignal.connect(self.progressUpdate)
		self.opticFlowThread.progressMinimumSignal.connect(self.progressMinimum)
		self.opticFlowThread.progressMaximumSignal.connect(self.progressMaximum)
		self.opticFlowThread.completeSignal.connect(self.trackingROIComplete)
		self.opticFlowThread.statusBarSignal.connect(self.statusBarMessage)
		self.trackROI_LKButton.setEnabled(False)
		self.trackROI_STButton.setEnabled(False)
		self.opticFlowThread.start()

	# Track with structure tensor analysis, ROI only
	def trackROI_ST(self):
		try:
			neighborhoodScale = int(self.neighborhoodScaleEdit.text())
			noiseScale = float(self.noiseScaleEdit.text())
			seedsPerPixel = float(self.seedsPerPixelEdit.text())			
			downsampleFactor = int(self.downsampleFactor)
		except:	
			msgBox = QMessageBox()
			msgBox.setText("Incorrect value for window size and max. level and seeds per pixel, expect integers/floats.")
			msgBox.exec()
			return None

		if self.imagesPath is None:
			msgBox = QMessageBox()
			msgBox.setText("Folder path for images is not set, please set it first.")
			msgBox.exec()
			return None

		if len(self.metadata) == 0:
			msgBox = QMessageBox()
			msgBox.setText("Metadata is not set, please set it first.")
			msgBox.exec()
			return None

		if self.userSelectedColor is None:
			msgBox = QMessageBox()
			msgBox.setText("Color not selected, please select from list or click Add a new color button.")
			msgBox.exec()
			return None    

		# Start tracking from this slice and use image ROI as mask
		mask_image = Image.open('user-selection.png')
		mask_image = np.asarray(mask_image)
  
		self.structureTensorThread = StructureTensorClass(self.imagesPath, mask_image,
														self.affine, self.metadata, neighborhoodScale, noiseScale,
														seedsPerPixel, downsampleFactor, self.startSliceIndex,
														self.forwardTrackingButton2.isChecked(), self.backwardTrackingButton2.isChecked())
		self.structureTensorThread.progressSignal.connect(self.progressUpdate)
		self.structureTensorThread.progressMinimumSignal.connect(self.progressMinimum)
		self.structureTensorThread.progressMaximumSignal.connect(self.progressMaximum)
		self.structureTensorThread.completeSignal.connect(self.trackingROIComplete)
		self.structureTensorThread.statusBarSignal.connect(self.statusBarMessage)
		self.trackROI_LKButton.setEnabled(False)
		self.trackROI_STButton.setEnabled(False)
		self.structureTensorThread.start()

	# Remove tracks through ROI
	def removeTracksThroughROI(self):
		sliceZcoordinates_physical = self.startSliceIndex * self.metadata['section_thickness']
		mask_image = Image.open('user-selection.png')
		mask_image = np.asarray(mask_image)
  
		trackIndicesToDelete = []
		
		for i in np.arange(len(self.streamlines)):
			for j in np.arange(len(self.streamlines[i])):
				if self.streamlines[i][j][2] == sliceZcoordinates_physical:
					y_index = self.metadata['y_size_pixels'] - self.streamlines[i][j][1]/self.metadata['pixel_size_xy']
					x_index = self.streamlines[i][j][0]/self.metadata['pixel_size_xy']
	 
					if mask_image[int(y_index), int(x_index)] == 255:
						trackIndicesToDelete.append(int(i)) 
		
		# Delete select indices
		self.streamlines = [ele for idx, ele in enumerate(self.streamlines) if idx not in trackIndicesToDelete]	
		self.color = np.delete(self.color, trackIndicesToDelete, axis = 0)

		# Update variables in SceneManager, update actors
		self.SceneManager.removeStreamlinesActor()
		self.SceneManager.updateStreamlinesAndColors(self.streamlines, self.color)
		self.SceneManager.createStreamlinesActor()

		self.unique_colors = np.unique(self.color, axis = 0)

		self.selectTracksListWidget.clear()
  
		for i in np.arange(self.unique_colors.shape[0]):
			listItem = QListWidgetItem('')
			listItem.setBackground(QColor(int(self.unique_colors[i][0] * 255), int(self.unique_colors[i][1] * 255),
										  int(self.unique_colors[i][2] * 255), 200))
			self.selectTracksListWidget.addItem(listItem)

		self.selectTracksListWidget.setItemDelegate(ColorDelegate(self.selectTracksListWidget))
		self.selectTracksListWidget.setSelectionMode(QAbstractItemView.MultiSelection)

		self.pickColorsListWidget.clear()
	
		for i in np.arange(self.unique_colors.shape[0]):
			listItem = QListWidgetItem('')
			listItem.setBackground(QColor(int(self.unique_colors[i][0] * 255), int(self.unique_colors[i][1] * 255),
										int(self.unique_colors[i][2] * 255), 200))
			self.pickColorsListWidget.addItem(listItem)

		self.pickColorsListWidget.setSelectionMode(QAbstractItemView.SingleSelection)
  
		self.SceneManager.visualizeStreamlines(self.streamlinesVisibilityCheckbox.isChecked())
		self.clusterCheckbox.setChecked(False)
		self.selectTracksByColorCheckbox.setChecked(False)

		self.statusBar().showMessage('Deleted tracks passing through selected ROI', 3000)
  
  		# Remove contour
		self.SceneManager.removeContour()
	
	# Remove tracks from ROI
	def removeTracksFromROI(self):
		sliceZcoordinates_physical = self.startSliceIndex * self.metadata['section_thickness']		
		mask_image = Image.open('user-selection.png')
		mask_image = np.asarray(mask_image)
		
		trackIndicesToDelete = []
		locationToDeleteFrom = []
		
		for i in np.arange(len(self.streamlines)):
			for j in np.arange(len(self.streamlines[i])):
				if self.streamlines[i][j][2] == sliceZcoordinates_physical:
					y_index = self.metadata['y_size_pixels'] - self.streamlines[i][j][1]/self.metadata['pixel_size_xy']
					x_index = self.streamlines[i][j][0]/self.metadata['pixel_size_xy']
	 
					if mask_image[int(y_index), int(x_index)] == 255:
						trackIndicesToDelete.append(int(i)) 
						locationToDeleteFrom.append(int(j))
		
		for i in np.arange(len(trackIndicesToDelete)):
			currentTrackIndexToClip = trackIndicesToDelete[int(i)]            
			full_streamline = self.streamlines[currentTrackIndexToClip]           
			self.streamlines[currentTrackIndexToClip] = full_streamline[:locationToDeleteFrom[int(i)]]

		# Sync variables, update actors, visualize
		self.SceneManager.removeStreamlinesActor()
		self.SceneManager.updateStreamlinesAndColors(self.streamlines, self.color)
		self.SceneManager.createStreamlinesActor()
		self.SceneManager.visualizeStreamlines(self.streamlinesVisibilityCheckbox.isChecked())
  
		self.SceneManager.visualizeStreamlines(self.streamlinesVisibilityCheckbox.isChecked())
		self.clusterCheckbox.setChecked(False)
		self.selectTracksByColorCheckbox.setChecked(False)

		self.statusBar().showMessage('Deleted tracks arising from selected ROI', 3000)       
  
  		# Remove contour
		self.SceneManager.removeContour()          
	
	# Processing if the anatomically constrain streamlines button is pressed
	def anatomicallyConstrainStreamlines(self):
		if self.fascicleSegmentationsPath is None:
			msgBox = QMessageBox()
			msgBox.setText("Fascicle segmentations folder path is not set, please set it first.")
			msgBox.setWindowTitle("Error")
			msgBox.exec()
			return None

		filelist = glob.glob(self.fascicleSegmentationsPath + '\\*.png')

		if len(filelist) < self.metadata['num_images_to_read']:
			msgBox = QMessageBox()
			msgBox.setText("Number of masks in fascicles segmentations path is less than num_images_to_read field in metadata XML file.")
			msgBox.setWindowTitle("Error")
			msgBox.exec()
			return None	

		reply = QMessageBox()
		reply.setWindowTitle('Warning')
		reply.setText("This operation cannot be undone. Save current streamlines to file if needed. Continue?")
		reply.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
		x = reply.exec()

		if x == QMessageBox.StandardButton.Yes:
			self.constrainStreamlines()
   
	# Constrain streamlines into provided mask
	def constrainStreamlines(self):
		
		num_images_to_process = self.metadata['num_images_to_read']
		streamlines_image_coords = self.convertStreamlinesToImageCooords()
  
		fascicle_label_filelist = glob.glob(self.fascicleSegmentationsPath + '\\*.png')

		self.actThread = ACTClass(num_images_to_process, streamlines_image_coords, fascicle_label_filelist)
  
		self.actThread.progressSignal.connect(self.progressUpdate)
		self.actThread.progressMinimumSignal.connect(self.progressMinimum)
		self.actThread.progressMaximumSignal.connect(self.progressMaximum)
		self.actThread.completeSignal.connect(self.actComplete)
		self.actThread.statusBarSignal.connect(self.statusBarMessage)
		self.computeTracksLKButton.setEnabled(False)
		self.computeTracksSTButton.setEnabled(False)
		self.anatomicallyConstrainStreamlinesButton.setEnabled(False)
		self.actThread.start()

	# Anatomically constrain tractography (ACT) thread finished slot
	def actComplete(self, value):
	 
		if value == 1:
			streamlines_image_coords = self.actThread.get_streamlines_image_coords()
	
			# Convert from image coordinates back to physical coordinates, save to list of lists
			self.convertStreamlinesToPhysCoords(streamlines_image_coords)

			self.SceneManager.removeStreamlinesActor()
			self.SceneManager.updateStreamlinesAndColors(self.streamlines, self.color)  
			self.SceneManager.createStreamlinesActor()
			self.SceneManager.visualizeStreamlines(self.streamlinesVisibilityCheckbox.isChecked())
   
			self.SceneManager.clipStreamlines(self.clipStreamlinesCheckbox.isChecked(), self.clipStreamlinesSlider.value(), self.xySlider.value(), self.ieTabSelected)

			self.actThread.terminate_thread()	
   	
			self.computeTracksLKButton.setEnabled(True)
			self.computeTracksSTButton.setEnabled(True)
			self.anatomicallyConstrainStreamlinesButton.setEnabled(True)  

	# Convert streamlines from physical space to image space
	def convertStreamlinesToImageCooords(self):

		self.statusBar().showMessage('Converting streamlines to image coordinates.')
  
		# Convert streamlines in physical coordinates to image space
		lin_T = self.affine[:3, :3].T.copy()
		offset = self.affine[:3, 3].copy()
  
		streamlines_phys = [None] * len(self.streamlines)
 
		for k in tqdm(np.arange(len(self.streamlines))):
			current_streamline = self.streamlines[k]
			current_streamline = np.array(current_streamline)
   
			current_streamline[:,0] = (current_streamline[:,0] - offset[0]) / self.metadata['pixel_size_xy']
			current_streamline[:,1] = self.metadata['y_size_pixels'] - ((current_streamline[:,1] - offset[1]) / self.metadata['pixel_size_xy'])
			current_streamline[:,2] = (current_streamline[:,2] - offset[2]) / self.metadata['section_thickness']

			current_streamline = current_streamline.astype(int)
   
			streamlines_phys[k] = current_streamline.tolist()
   
		return streamlines_phys

	# Convert streamlines from image space to physical space
	def convertStreamlinesToPhysCoords(self, streamlines_image_coords):
		
		self.statusBar().showMessage('Converting streamlines to physical coordinates.')
  		
		lin_T = self.affine[:3, :3].T.copy()
		offset = self.affine[:3, 3].copy()
  
		for k in tqdm(np.arange(len(streamlines_image_coords))):
			current_streamline = streamlines_image_coords[k]   
			current_streamline = np.array(current_streamline)
			current_streamline[:,1] = self.metadata['y_size_pixels'] - current_streamline[:,1]

			current_streamline = np.dot(current_streamline, lin_T) + offset
   
			self.streamlines[k] = current_streamline 

		self.statusBar().showMessage('')
   
	# For tractogram comparison only, no visualization/rendering update
	def getClusterStreamlinesAndColors(self, streamlines, colors):

		# Do not consider streamlines that do not cover full length of the stack, else short streamlines will get their own clusters
		# Computational modeling considers streamlines over the full length of the stack
		streamline_lengths = [streamline.shape[0] for streamline in streamlines]

		# Length of the full stack is the most commonly occuring streamline length
		full_length = scipy.stats.mode(streamline_lengths, keepdims=False)[0]

		# Create a new streamlines variable that only contains streamlines across the full length
		streamlines_for_clustering = []  
		colors_for_clustering = []
  
		for k in range(len(streamlines)):
			if streamlines[k].shape[0] == full_length:
				streamlines_for_clustering.append(streamlines[k])
				colors_for_clustering.append(colors[k])
	 
		colors_for_clustering = np.array(colors_for_clustering)
		qb = QuickBundles(threshold = float(self.clusteringThresholdEdit.text()))
		clusters = qb.cluster(streamlines_for_clustering)

		clusterColors = np.empty((len(clusters.centroids), 3))

		for k in np.arange(len(clusters.centroids)):
			clusterColors[k,:] = self.SceneManager.mode_rows(colors_for_clustering[clusters[k].indices, :])

		return clusters.centroids, clusterColors
  
	# Compare two tractograms using MCN distance 
	def compareTractograms(self):
		title = "Select first streamline file for comparison, color file should be in same folder"
		streamlineFileOnePath = QFileDialog.getOpenFileName(self,
										title,
										os.path.expanduser("."),
										"Streamline File (*.pkl)")
  
		if streamlineFileOnePath[0] == '':
			streamlineFileOnePath = None
			return

		title = "Select second streamline file for comparison, color file should be in same folder"
		streamlineFileTwoPath = QFileDialog.getOpenFileName(self,
										title,
										os.path.expanduser("."),
										"Streamline File (*.pkl)")
  
		if streamlineFileTwoPath[0] == '':
			streamlineFileTwoPath = None
			return

		colorsOneFileName = 'colors_' + os.path.basename(streamlineFileOnePath[0])[12:]
		colorsTwoFileName = 'colors_' + os.path.basename(streamlineFileTwoPath[0])[12:]
		colorsOneFilePath = os.path.dirname(streamlineFileOnePath[0]) + '\\' + colorsOneFileName
		colorsTwoFilePath = os.path.dirname(streamlineFileTwoPath[0]) + '\\' + colorsTwoFileName
  
		if not os.path.exists(colorsOneFilePath):
			msgBox = QMessageBox()
			msgBox.setText("Cannot find color file in the same path as the first tractogram file. Please verify.")
			msgBox.exec()
			return None	

		if not os.path.exists(colorsTwoFilePath):
			msgBox = QMessageBox()
			msgBox.setText("Cannot find color file in the same path as the second tractogram file. Please verify.")
			msgBox.exec()
			return None	

		msg = QMessageBox()
		msg.setWindowTitle("Tractogram comparison options")
		check_box = QCheckBox("Compare within the same colors?")
		msg.setCheckBox(check_box)
		reply = msg.exec()
  
		compareWithinColors = False
  
		if reply == QMessageBox.Ok:
			if check_box.checkState() == Qt.Checked:
				compareWithinColors = True

		# Cluster each tractogram  
		with open(streamlineFileOnePath[0], 'rb') as f:
			streamlinesOne = pickle.load(f)

		with open(streamlineFileTwoPath[0], 'rb') as f:
			streamlinesTwo = pickle.load(f) 

		with open(colorsOneFilePath, 'rb') as f:
			colorsOne = pickle.load(f)

		with open(colorsTwoFilePath, 'rb') as f:
			colorsTwo = pickle.load(f) 
		
		clusterStreamlinesOne, clusterColorsOne = self.getClusterStreamlinesAndColors(streamlinesOne, colorsOne)   
		clusterStreamlinesTwo, clusterColorsTwo = self.getClusterStreamlinesAndColors(streamlinesTwo, colorsTwo) 
  
		metric = compare_tractogram_clusters(clusterStreamlinesOne, clusterStreamlinesTwo, clusterColorsOne, clusterColorsTwo,compareWithinColors)
  
		msgBox = QMessageBox()
		msgBox.setWindowTitle('Similarity metric')
		msgBox.setText("Closest-neighbor based distance metric between provided tractograms: " + str(np.round(metric, 2)))
		msgBox.exec()
	
	# Window level adjustment from UI (only for visualization, not used in analysis)
	def setWindowLevel(self, value):
	 
		if value:
			self.SceneManager.windowLevelAdjustments(value)
		else:
			reply = QMessageBox()
			reply.setWindowTitle('Confirmation')
			reply.setText("Do you want to keep current window/level settings? \nIf not, window/level will revert to original.")
			reply.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
			x = reply.exec()

			if x == QMessageBox.StandardButton.Yes:
				self.SceneManager.windowLevelAdjustments(value, True)
			else:
				self.SceneManager.windowLevelAdjustments(value, False)
	
	def FileExit(self):
		app.quit()

	def ShowAboutDialog(self):
		msgBox = QMessageBox()
		msgBox.setText("NerveTracker - track nerve fibers in blockface microscopy images.")
		msgBox.exec()

	def SetViewXY(self):

		self.radioButtonXY.setChecked(True)
		self.SceneManager.SetViewXY()

	def SetViewXZ(self):
		self.SceneManager.SetViewXZ()

	def SetViewYZ(self):
		self.SceneManager.SetViewYZ()

	def Snapshot(self):
		self.SceneManager.Snapshot()

	def ToggleVisualizeAxis(self, visible):
	 
		self.actionVisualize_Axis.setChecked(visible)
		self.checkVisualizeAxis.setChecked(visible)
		self.SceneManager.ToggleVisualizeAxis(visible)

	def xySliderUpdate(self, value):
		self.XYSliceEdit.setText(str(value))
		self.XYSliceEdit2.setText(str(value))

		# Update both slider positions
		self.xySlider.setSliderPosition(value)
		self.xySlider2.setSliderPosition(value)  

		self.SceneManager.visualizeXYSlice(value, self.checkXYSlice.isChecked())		
		self.SceneManager.clipStreamlines(self.clipStreamlinesCheckbox.isChecked(), self.clipStreamlinesSlider.value(), value, self.ieTabSelected)

	def xySliceEdit_changed(self, value = None):
	 
		if value is None:
			value = self.sender().text()
  
		try:
			self.xySlider.setValue(int(value))
		except:			
			msgBox = QMessageBox()
			msgBox.setText("Incorrect value for XY Slice, expect integer")
			msgBox.exec()
			return None

		try:
			self.xySlider2.setValue(int(value))
		except:			
			msgBox = QMessageBox()
			msgBox.setText("Incorrect value for XY Slice, expect integer")
			msgBox.exec()
			return None

		self.SceneManager.visualizeXYSlice(int(self.XYSliceEdit.text()), self.checkXYSlice.isChecked())
		self.SceneManager.clipStreamlines(self.clipStreamlinesCheckbox.isChecked(), self.clipStreamlinesSlider.value(), int(self.XYSliceEdit.text()), self.ieTabSelected)

	def xySliceCheckboxSelect(self, visible):
		self.SceneManager.visualizeXYSlice(self.xySlider.value(), self.checkXYSlice.isChecked())

	def xySliceOpacity(self, value):
		self.SceneManager.opacityXYSlice(value)

	def parallelPerspectiveViewButton(self):
		self.SceneManager.toggleCameraParallelPerspective()       

	def validateTractograms(self):
		title = "Select the streamlines file that should be validated"
		streamlinesFilePath = QFileDialog.getOpenFileName(self,
										title,
										os.path.expanduser("."),
										"Streamline File (*.pkl)")

		if streamlinesFilePath[0] == '':
			streamlinesFilePath = None
			return

		colorsFile = os.path.dirname(streamlinesFilePath[0]) + '\\colors_' + os.path.basename(streamlinesFilePath[0])[12:]
  
		if not os.path.exists(colorsFile):
			msgBox = QMessageBox()
			msgBox.setText("Did not find the colors pickle file in the same folder, exiting.")
			msgBox.exec()
			return 			
  
		title = "Select the folder containing the ground truth segmentation masks (each sub-folder should containing masks tracking one ROI)"
		flags = QFileDialog.ShowDirsOnly
		self.validation_masks = QFileDialog.getExistingDirectory(self,
															title,
															os.path.expanduser("."),
															flags)

		if self.validation_masks == '':
			self.validation_masks = None
			return

		
		dialog = ValidationMetadataDialogBox()
		if dialog.exec_() == QDialog.Accepted:
			validationMetadata = dialog.get_metadata()

		dice_color_averaged, normalized_dice, false_positives, true_positives, false_negatives, true_negatives = validate(streamlinesFilePath[0], colorsFile, self.validation_masks, validationMetadata)
	
		msgBox = QMessageBox()
		msgBox.setWindowTitle('Validate tractograms result')
		msgBox.setText("Normalized Dice score at intermediate slices: " + np.array2string(normalized_dice) + "\nMean: " + str(np.round(np.mean(normalized_dice), 2)))
		msgBox.exec()
  
		msgBox = QMessageBox()
		msgBox.setWindowTitle('Validate tractograms result')
		msgBox.setText("False positives: " + np.array2string(np.sum(false_positives)) + "\n" +
                 		"True positives: " + np.array2string(np.sum(true_positives)) + "\n" +
                   "False negatives: " + np.array2string(np.sum(false_negatives)) + "\n" +
                   "True negatives: " + np.array2string(np.sum(true_negatives)) + "\n")
		msgBox.exec()
  
		return 	

	def changeSlice(self, value):
		self.xySliceEdit_changed(value)
	
	def flythroughComplete(self, value):
		self.movieThread.terminate_thread()
		
	def stack_flythrough(self):
		
		# Record screen with OBS studio before running this flythrough function
		self.movieThread = MovieClass(self.metadata['num_images_to_read'])
		self.movieThread.sliceSignal.connect(self.changeSlice)
		self.movieThread.completeSignal.connect(self.flythroughComplete)
		self.movieThread.start()
  
if __name__ == '__main__':

	app = QApplication(sys.argv)
	app.setWindowIcon(QIcon('docs\icon.jpg'))

	myappid = 'NerveTracker.v01' # arbitrary string
	ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
 
	mw = MainWindow()
	mw.show()
	sys.exit(app.exec_())