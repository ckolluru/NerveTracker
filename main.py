import sys

from PyQt5.QtWidgets import QApplication, QWidget,QMainWindow,QFileDialog,QMessageBox,QSlider,QLineEdit,QCheckBox,QListWidget,QListWidgetItem,QAbstractItemView, QInputDialog, QTabWidget, QStyledItemDelegate, QPushButton, QRadioButton,QProgressBar
from PyQt5.QtWidgets import QApplication, QDialog, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QFormLayout
from PyQt5.QtGui import QColor, QIcon, QPen
from PyQt5.QtWidgets import QStyle
from PyQt5.uic import loadUiType
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from PyQt5.QtCore import Qt

# VTK
from scenemanager import SceneManager

# We'll need to access home directory, file path read, xml read
from os.path import expanduser
import glob
import untangle

# numpy imports
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from tqdm import tqdm

from optical_flow_module import OpticFlowClass
from structure_tensor_module import StructureTensorClass
from act_module import ACTClass
import pickle
from dipy.segment.clustering import QuickBundles
import scipy

import distinctipy
from PIL import Image
import os
from compare_tractograms import compare_tractogram_clusters

# load GUI
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
			painter.setPen(QPen(QColor(255, 0, 0), 1))
			painter.drawRect(color_rect)

		painter.restore()

# Dialog box for inputting image metadata
class DialogBox(QDialog):
	def __init__(self, parent=None):
		super(DialogBox, self).__init__(parent)
  
		self.metadata = self.get_defaults()  
		self.createDialogBoxValues()
		
		# Create the OK button and connect it to the accept method
		self.ok_button = QPushButton('OK', self)
		self.ok_button.clicked.connect(self.accept)
		
		# Create a load from XML button and connect it to the load method
		self.load_button = QPushButton('Load from XML', self)
		self.load_button.clicked.connect(self.load)
		
		# Create a save to XML button and connect it to the save method
		self.save_button = QPushButton('Save to XML', self)
		self.save_button.clicked.connect(self.save)
  
		# Create the layout and add the widgets to it
		layout = QFormLayout()
		layout.addRow(self.label1, self.pixel_size_xy)
		layout.addRow(self.label2, self.section_thickness)
		layout.addRow(self.label3, self.image_type)
		layout.addRow(self.label4, self.num_images_to_read)
		layout.addRow(self.label5, self.chunk_size_z)

		buttons_layout = QHBoxLayout()
		buttons_layout.addWidget(self.load_button)
		buttons_layout.addWidget(self.save_button)
		buttons_layout.addStretch()
		buttons_layout.addWidget(self.ok_button)

		main_layout = QVBoxLayout(self)
		main_layout.addLayout(layout)
		main_layout.addLayout(buttons_layout)
  
		self.setLayout(main_layout)
		self.setWindowTitle ('Image metadata information')
		self.setWhatsThis("Provide metadata information for the image stack.")

	def get_defaults(self):
    
		metadata = dict()
		metadata['pixel_size_xy'] = 0.74
		metadata['section_thickness'] = 3.0
		metadata['image_type'] = '.png'
		metadata['num_images_to_read'] = 903
		metadata['step_size'] = 64

		return metadata
		
	# Create the labels and input fields
	def createDialogBoxValues(self):

		self.label1 = QLabel('Pixel size (XY) in microns:', self)
		self.pixel_size_xy = QLineEdit(str(self.metadata['pixel_size_xy']), self)
		self.label2 = QLabel('Section thickness in microns:', self)
		self.section_thickness = QLineEdit(str(self.metadata['section_thickness']), self)
		self.label3 = QLabel('Image Type:', self)
		self.image_type = QLineEdit(str(self.metadata['image_type']), self)
		self.label4 = QLabel('Number of images to read:', self)
		self.num_images_to_read = QLineEdit(str(self.metadata['num_images_to_read']), self)
		self.label5 = QLabel('Chunk size in Z (used for ST analysis):', self)
		self.chunk_size_z = QLineEdit(str(self.metadata['step_size']), self)  

	# Update dialog box from XML file
	def updateDialogBox(self):
    
		self.pixel_size_xy.setText(str(self.metadata['pixel_size_xy']))
		self.section_thickness.setText(str(self.metadata['section_thickness']))
		self.image_type.setText(str(self.metadata['image_type']))
		self.num_images_to_read.setText(str(self.metadata['num_images_to_read']))
		self.chunk_size_z.setText(str(self.metadata['step_size']))
  
	# Define a method to return the user input when the dialog is accepted
	def get_metadata(self):
		self.metadata['pixel_size_xy'] = float(self.pixel_size_xy.text())
		self.metadata['section_thickness'] = float(self.section_thickness.text())
		self.metadata['image_type'] = self.image_type.text()
		self.metadata['num_images_to_read'] = int(self.num_images_to_read.text())
		self.metadata['step_size'] = int(self.chunk_size_z.text())
  		
		return self.metadata
  
	# Load metadata from XML
	def load(self):
    
		title = "Open Image metadata XML file"
		self.imageMetadataXMLFileName = QFileDialog.getOpenFileName(self,
										title,
										expanduser("."),
										"Image Files (*.xml *.XML)")
  
		if self.imageMetadataXMLFileName[0] == '':
			self.imageMetadataXMLFileName = None
			return
 
		doc = untangle.parse(self.imageMetadataXMLFileName[0])

		self.metadata['pixel_size_xy'] = float(doc.root.pixel_size_xy['name'])
		self.metadata['section_thickness'] = float(doc.root.section_thickness['name'])
		self.metadata['image_type'] = doc.root.image_type['name']
		self.metadata['num_images_to_read'] = int(doc.root.num_images_to_read['name'])
		self.metadata['step_size'] = int(doc.root.step_size['name'])
  
		self.updateDialogBox()
  
	# Save metadata to XML
	def save(self):
     
		# Update current copy of metadata
		self.get_metadata()
  
		# Create a file dialog instance
		dialog = QFileDialog()

		# Set the dialog options
		dialog.setFileMode(QFileDialog.AnyFile)
		dialog.setNameFilter("All files (*.*)")

		# Set the dialog to save mode
		dialog.setAcceptMode(QFileDialog.AcceptSave)

		# Show the dialog and wait for the user to enter a file name
		if dialog.exec_():
			# Get the selected file path
			file_path = dialog.selectedFiles()[0]   
   
			if os.path.exists(file_path):
				os.remove(file_path)

			with open(file_path, 'w') as file:
				file.write('<?xml version="1.0"?>\n')  
				file.write('<root>\n')
				file.write('\t<pixel_size_xy name="{}"/>\n'.format(self.metadata['pixel_size_xy']))
				file.write('\t<section_thickness name="{}"/>\n'.format(self.metadata['section_thickness']))
				file.write('\t<image_type name="{}"/>\n'.format(self.metadata['image_type']))
				file.write('\t<num_images_to_read name="{}"/>\n'.format(self.metadata['num_images_to_read']))
				file.write('\t<step_size name="{}"/>\n'.format(self.metadata['step_size']))
				file.write('</root>')

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
		self.volume = None
		self.affine = None
		self.streamlines = None
		self.color = None
		self.mask_image = None
		self.trackingAlgoLK = 0
		self.clusters = None
		self.userSelectedColor = None
		self.startSliceIndex = None

		# Initialize widgets (visualization tab page)
		self.xySlider = self.findChild(QSlider, 'xySlider')
		self.xySliceEdit = self.findChild(QLineEdit, 'XYSliceEdit')
		self.xySliceCheckBox = self.findChild(QCheckBox, 'checkXYSlice')
		self.windowSize = self.findChild(QLineEdit, 'windowSizeEdit')
		self.maxLevel = self.findChild(QLineEdit, 'maxLevelEdit')
		self.neighborhoodScale = self.findChild(QLineEdit, 'neighborhoodScaleLineEdit')
		self.noiseScale = self.findChild(QLineEdit, 'noiseScaleLineEdit')
		self.seedsPerPixel = self.findChild(QLineEdit, 'seedsPerPixelEdit')
		self.clipStreamlinesLineEdit = self.findChild(QLineEdit, 'clipStreamlinesLineEdit')
		self.clipStreamlinesSlider = self.findChild(QSlider, 'clipStreamlinesSlider')
		self.clipStreamlinescheckbox = self.findChild(QCheckBox, 'clipStreamlinescheckbox')
		self.selectTracksListWidget = self.findChild(QListWidget, 'selectTracksListWidget')
		self.streamlinesVisibilityCheckbox = self.findChild(QCheckBox, 'streamlinesVisibilityCheckbox')
		self.selectTracksByColorCheckbox = self.findChild(QCheckBox, 'selectTracksByColorCheckBox')
		self.clustersCheckBox = self.findChild(QCheckBox, 'clusterCheckBox')
		self.blur = self.findChild(QLineEdit, 'blurLineEdit')
		self.clusteringThresholdLineEdit = self.findChild(QLineEdit, 'clusteringThresholdLineEdit')
		self.boundingBoxCheckBox = self.findChild(QCheckBox, 'boundingBoxCheckBox')
		
		self.tabWidget = self.findChild(QTabWidget, 'tabWidget')
		self.visualizationTab = self.tabWidget.findChild(QWidget, 'visualizationTab')
		self.tabWidget.setCurrentWidget(self.visualizationTab)
  
		# Initialize widgets (interactive editing tab page)
		self.xySlider2 = self.findChild(QSlider, 'xySlider2')
		self.xySliceEdit2 = self.findChild(QLineEdit, 'XYSliceEdit2')
		self.pickColorListWidget = self.findChild(QListWidget, 'pickColorsListWidget')
		self.interactiveEditingTab = self.tabWidget.findChild(QWidget, 'interactiveEditingTab')

		# Disable interactive editing tab until streamlines are created/loaded
		self.tabWidget.setTabEnabled(1, False)
  
		# Threads for optic flow analysis and structure tensor analysis
		self.opticFlowThread = None
		self.structureTensorThread = None
  
		# Push button controls, disable when computation is running
		self.computeTracksLKButton = self.findChild(QPushButton, 'computeTracksLKButton')
		self.trackROI_LKButton = self.findChild(QPushButton, 'trackROI_LKButton')  
		self.computeTracksSTButton = self.findChild(QPushButton, 'computeTracksSTButton')
		self.trackROI_STButton = self.findChild(QPushButton, 'trackROI_STButton')
		self.anatomicallyConstrainStreamlinesButton = self.findChild(QPushButton, 'anatomicallyConstrainStreamlinesButton')
  
		# Slice index from which tracks should be started
		self.tracksStartingSliceIndex = self.findChild(QLineEdit, 'tracksStartingSliceIndex')  
		self.forwardTrackingButton = self.findChild(QRadioButton, 'forwardTrackingButton')
		self.backwardTrackingButton = self.findChild(QRadioButton, 'backwardTrackingButton')
		self.forwardTrackingButton2 = self.findChild(QRadioButton, 'forwardTrackingButton2')
		self.backwardTrackingButton2 = self.findChild(QRadioButton, 'backwardTrackingButton2')
  
		# Progress Bar
		self.progressBar = self.findChild(QProgressBar, 'progressBar')
		self.progressBar2 = self.findChild(QProgressBar, 'progressBar2')
  
		# Validate inputs
		winSizeValidator = QIntValidator(3, 1000, self.windowSize)
		self.windowSize.setValidator(winSizeValidator)
  
	def initVTK(self):
		self.show()		# We need to call QVTKWidget's show function before initializing the interactor
		self.SceneManager = SceneManager(self.vtkContext)

	def OpenFolderImages(self):
		title = "Open Folder with PNG images"
		flags = QFileDialog.ShowDirsOnly
		self.imagesPath = QFileDialog.getExistingDirectory(self,
															title,
															expanduser("."),
															flags)

		if self.imagesPath == '':
			self.imagesPath = None
			return
		
		self.statusBar().showMessage('Image folder location saved', 2000)

	def OpenMaskImageForSeeds(self):
		title = "Open Mask Image with Regions For Seeds (corresponds to an arbitrary slice in image stack)"
		self.maskImageForSeedsFilePath = QFileDialog.getOpenFileName(self,
										title,
										expanduser("."),
										"Mask File (*.png *.PNG)")
  
		if self.maskImageForSeedsFilePath[0] == '':
			self.maskImageForSeedsFilePath = None
			return

		self.LoadMaskImage()

		self.statusBar().showMessage('Mask image location saved', 2000)

	# Read metadata from a dialog box
	def OpenImageMetadataXML(self):

		dialog = DialogBox()
		if dialog.exec_() == QDialog.Accepted:
			self.metadata = dialog.get_metadata()

		self.readMetadata()

	def OpenFascicleSegmentationsFolder(self):
		title = "Open Folder with Fascicle Segmentation Images"
		flags = QFileDialog.ShowDirsOnly
		self.fascicleSegmentationsPath = QFileDialog.getExistingDirectory(self,
																		title,
																		expanduser("."),
																		flags)
		if self.fascicleSegmentationsPath == '':
			self.fascicleSegmentationsPath = None
			return
		
		self.statusBar().showMessage('Fascicle segmentation folder location saved', 2000)		
	
	# def createMaps(self):
	# 	if self.fascicleSegmentationsPath is None:
	# 		msgBox = QMessageBox()
	# 		msgBox.setText("Fascicle segmentations folder path is not set, please set it first.")
	# 		msgBox.exec()
	# 		return None

	# 	text, ok = QInputDialog.getText(self, 'Export maps', 'Input spacing (in number of slices) between maps:')
	# 	if ok:
	# 		try:
	# 			spacing = int(text)
	# 		except:
	# 			msgBox = QMessageBox()
	# 			msgBox.setText("Could not convert value entered into integer, please try again.")
	# 			msgBox.exec()
	# 			return None

	# 	if self.streamlines is None:	  
	# 		msgBox = QMessageBox()
	# 		msgBox.setText("Streamlines need to be computed/loaded, please try again.")
	# 		msgBox.exec()
	# 		return None

	# 	if self.color is None:	  
	# 		msgBox = QMessageBox()
	# 		msgBox.setText("Streamlines need to be computed/loaded or colors need to be loaded, please try again.")
	# 		msgBox.exec()
	# 		return None

	# 	exportMaps(self.fascicleSegmentationsPath, self.streamlines, self.color, spacing, self.metadata['image_type'], self.metadata['num_images_to_read'], self.metadata['section_thickness'], self.metadata['pixel_size_xy'], self.metadata['y_size_pixels'])

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

		self.mask_image = (plt.imread(self.maskImageForSeedsFilePath[0])* 255).astype('uint8')
		self.mask_image = self.mask_image.astype('uint8')
		self.mask_image[self.mask_image > 0] = 255
		
		self.statusBar().showMessage('Finished reading mask image for seeds into memory.', 2000)

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
 
		if self.trackingAlgoLK:
			self.SceneManager.exportStreamlinesAndColorsLK(self.windowSize.text(), self.maxLevel.text(), self.seedsPerPixel.text(), self.blur.text(), sample_name)
		else:
			self.SceneManager.exportStreamlinesAndColorsST(self.neighborhoodScale.text(), self.noiseScale.text(), self.seedsPerPixel.text(), self.downsampleFactor, sample_name)

		self.statusBar().showMessage('Saved streamlines and colors data into pickle files folder.', 2000)

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
 
		if self.trackingAlgoLK:
			self.SceneManager.exportStreamlinesClustersLK(self.windowSize.text(), self.maxLevel.text(), self.seedsPerPixel.text(), self.blur.text(), sample_name)
		else:
			self.SceneManager.exportStreamlinesClustersST(self.neighborhoodScale.text(), self.noiseScale.text(), self.seedsPerPixel.text(), self.downsampleFactor, sample_name)

		self.statusBar().showMessage('Saved streamline clusters into pickle files folder.', 2000)
	  
	def loadStreamlines(self):

		title = "Open Streamlines Pickle file, needs colors pickle file in the same folder."
		self.streamlinesPickleFile = QFileDialog.getOpenFileName(self,
										title,
										expanduser("."),
										"Pickle Files (*.pkl)")

		if self.streamlinesPickleFile[0] == '':
			self.streamlinesPickleFile = None
			return

		# Find the colors pickle file in the same folder
		colorsFileName = 'colors_' + os.path.basename(self.streamlinesPickleFile[0])[12:]
		self.colorsPickleFile = os.path.dirname(self.streamlinesPickleFile[0]) + '\\' + colorsFileName
  
		if not os.path.exists(self.colorsPickleFile):
			msgBox = QMessageBox()
			msgBox.setText("Did not find the colors pickle file in the same folder, please verify.")
			msgBox.exec()
			return None

		# Read the streamline pickle file and update variables
		with open(self.streamlinesPickleFile[0], 'rb') as f:
			self.streamlines = pickle.load(f)   
			self.SceneManager.updateStreamlines(self.streamlines)
   
		self.statusBar().showMessage('Loading streamlines done.', 2000)  
  
		# Read the colors pickle file and update variables
		with open(self.colorsPickleFile, 'rb') as f:
			self.color = pickle.load(f)
			self.SceneManager.updateColors(self.color)

		# Update the colors list widget
		self.unique_colors = np.unique(self.color, axis = 0)
		self.selectTracksListWidget.clear()
  
		for i in np.arange(self.unique_colors.shape[0]):
			listItem = QListWidgetItem('')
			listItem.setBackground(QColor(int(self.unique_colors[i][0] * 255), int(self.unique_colors[i][1] * 255),
										  int(self.unique_colors[i][2] * 255), 200))
			self.selectTracksListWidget.addItem(listItem)

		self.selectTracksListWidget.setItemDelegate(ColorDelegate(self.selectTracksListWidget))
		self.selectTracksListWidget.setSelectionMode(QAbstractItemView.MultiSelection)
  
		self.statusBar().showMessage('Loading colors done, can display streamlines now.', 2000) 
 
 		# Enable the interactive editing tab  
		if self.color is not None and self.streamlines is not None:

			self.tabWidget.setTabEnabled(1, True)   
			self.SceneManager.updateStreamlinesAndColors(self.streamlines, self.color)
  
			self.SceneManager.showAllTracks(self.streamlinesVisibilityCheckbox.isChecked())	
			self.SceneManager.clipStreamlines(self.clipStreamlinescheckbox.isChecked(), self.clipStreamlinesSlider.value(), self.startSliceIndex)

		if '_st_' in self.streamlinesPickleFile[0]:
			self.trackingAlgoLK = 0
		else:
			self.trackingAlgoLK = 1 

	def visualizeBoundingBox(self, value):
		
		self.SceneManager.visualizeBoundingBox(self.boundingBoxCheckBox.isChecked())
  
	# Connect the forward/backward radio button group's buttonReleased() signal to a function that checks if at least one box is checked
	def trackingDirection(self):
		if not self.forwardTrackingButton.isChecked() and not self.backwardTrackingButton.isChecked():
			self.sender().setChecked(True)
   
		if not self.forwardTrackingButton2.isChecked() and not self.backwardTrackingButton2.isChecked():
			self.sender().setChecked(True)
   
	def clusterStreamlines(self, isChecked):
	 	
		if isChecked: 
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
			qb = QuickBundles(threshold = float(self.clusteringThresholdLineEdit.text()))
			self.clusters = qb.cluster(streamlines_for_clustering)
   
		self.SceneManager.visualizeClusters(self.clusters, colors_for_clustering, isChecked, self.streamlinesVisibilityCheckbox.isChecked())

		self.statusBar().showMessage('Visualizing cluster bundles complete.', 2000)

	def FileExit(self):
		app.quit()

	def ShowAboutDialog(self):
		msgBox = QMessageBox()
		msgBox.setText("NerveTracker - track nerve fibers in blockface microscopy images.")
		msgBox.exec()

	def SetViewXY(self):
		# Ensure UI is sync (if view was selected from menu)
		self.radioButtonXY.setChecked(True)
		self.SceneManager.SetViewXY()

	def SetViewXZ(self):
		# Ensure UI is sync (if view was selected from menu)
		self.radioButtonXZ.setChecked(True)
		self.SceneManager.SetViewXZ()

	def SetViewYZ(self):
		# Ensure UI is sync (if view was selected from menu)
		self.radioButtonYZ.setChecked(True)
		self.SceneManager.SetViewYZ()

	def Snapshot(self):
		self.SceneManager.Snapshot()

	def ToggleVisualizeAxis(self, visible):
		# Ensure UI is sync
		self.actionVisualize_Axis.setChecked(visible)
		self.checkVisualizeAxis.setChecked(visible)
		self.SceneManager.ToggleVisualizeAxis(visible)

	def readMetadata(self):

		if self.imagesPath is None:
			msgBox = QMessageBox()
			msgBox.setText("Folder path for images is not set, please set it first.")
			msgBox.exec()
			return None

		# Read the first image in the directory and infer the image shape
		filelist = glob.glob(self.imagesPath + '\\*' + self.metadata['image_type'])
		pil_image = Image.open(filelist[0])
		image = np.asarray(pil_image)
  
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

		self.SceneManager.addImageDataActor(self.imagesPath, self.metadata['pixel_size_xy'],
											self.metadata['section_thickness'], self.metadata['x_size_pixels'],
											self.metadata['y_size_pixels'], self.metadata['num_images_to_read'])
  
		self.SceneManager.addBoundingBox(self.metadata)

		validator = QIntValidator(0, self.metadata['num_images_to_read'] - 1, self.tracksStartingSliceIndex)
		self.tracksStartingSliceIndex.setValidator(validator)
  
		self.downsampleFactor = str(int(round(self.metadata['section_thickness'] / self.metadata['pixel_size_xy'])))
  
		self.statusBar().showMessage('Metadata reading complete', 2000)

	def xySliderUpdate(self, value):
		self.xySliceEdit.setText(str(value))
		self.xySliceEdit2.setText(str(value))

		# Update both slider positions
		self.xySlider.setSliderPosition(value)
		self.xySlider2.setSliderPosition(value)  

		self.SceneManager.visualizeXYSlice(value, self.xySliceCheckBox.isChecked())		
		self.SceneManager.clipStreamlines(self.clipStreamlinescheckbox.isChecked(), self.clipStreamlinesSlider.value(), value)

	def xySliceEdit_changed(self):
		try:
			self.xySlider.setValue(int(self.xySliceEdit.text()))
		except:			
			msgBox = QMessageBox()
			msgBox.setText("Incorrect value for XY Slice, expect integer")
			msgBox.exec()
			return None

		try:
			self.xySlider2.setValue(int(self.xySliceEdit.text()))
		except:			
			msgBox = QMessageBox()
			msgBox.setText("Incorrect value for XY Slice, expect integer")
			msgBox.exec()
			return None

		self.SceneManager.visualizeXYSlice(int(self.xySliceEdit.text()), self.xySliceCheckBox.isChecked())
		self.SceneManager.clipStreamlines(self.clipStreamlinescheckbox.isChecked(), self.clipStreamlinesSlider.value(), int(self.xySliceEdit.text()))

	def xySliceCheckboxSelect(self, visible):
		self.SceneManager.visualizeXYSlice(self.xySlider.value(), self.xySliceCheckBox.isChecked())

	def xySliceOpacity(self, value):
		self.SceneManager.opacityXYSlice(value)

	def parallelPerspectiveViewButton(self):
		self.SceneManager.toggleCameraParallelPerspective()
  
	def progressUpdate(self, value):
		self.progressBar.setValue(value)  
		self.progressBar2.setValue(value)  
  
	def progressMinimum(self, value):
		self.progressBar.setMinimum(value)
		self.progressBar2.setMinimum(value)
  
	def progressMaximum(self, value):
		self.progressBar.setMaximum(value)
		self.progressBar2.setMaximum(value)
  
	def statusBarMessage(self, string):
		self.statusBar().showMessage(string)

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
			self.SceneManager.addStreamlinesActor(self.streamlines, self.color)

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
				self.trackingAlgoLK = 1

			if value == 2:
				self.trackingAlgoLK = 0
	
			if self.color is not None and self.streamlines is not None:
				# Enable the interactive editing tab 
				self.tabWidget.setTabEnabled(1, True)

			# Enable the push buttons back   
			self.computeTracksLKButton.setEnabled(True)
			self.computeTracksSTButton.setEnabled(True)
			self.anatomicallyConstrainStreamlinesButton.setEnabled(True)

	def computeTracksLK(self):
		try:
			windowSize = int(self.windowSize.text())
			maxLevel = int(self.maxLevel.text())
			seedsPerPixel = float(self.seedsPerPixel.text())
			blur = float(self.blurLineEdit.text())
		except:	
			msgBox = QMessageBox()
			msgBox.setText("Incorrect value for window size and max. level and seeds per pixel, expect integers/floats.")
			msgBox.exec()
			return None

		if self.mask_image is None:
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

		self.opticFlowThread = OpticFlowClass(self.imagesPath, self.mask_image,
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
  
	def computeTracksST(self):
		try:
			neighborhoodScale = int(self.neighborhoodScale.text())
			noiseScale = float(self.noiseScale.text())
			seedsPerPixel = float(self.seedsPerPixel.text())
			downsampleFactor = int(self.downsampleFactor)
		except:	
			msgBox = QMessageBox()
			msgBox.setText("Incorrect value for window size and max. level and seeds per pixel, expect integers/floats.")
			msgBox.exec()
			return None

		if self.mask_image is None:
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
 
		self.structureTensorThread = StructureTensorClass(self.imagesPath, self.mask_image,
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
  
	def visualizeStreamlines(self, value):

		if self.SceneManager.streamlinesActor is None:
			msgBox = QMessageBox()
			msgBox.setText("Streamlines not created, click compute tracks button or load existing streamlines file.")
			msgBox.exec()
			return None

		self.SceneManager.visualizeStreamlines(value)
	
	def streamlinesOpacity(self, value):

		if self.SceneManager.streamlinesActor is None:
			msgBox = QMessageBox()
			msgBox.setText("Streamlines not created, click compute tracks button or load existing streamlines file.")
			msgBox.exec()
			return None

		self.SceneManager.opacityStreamlines(value)
		self.SceneManager.opacityClusters(value)

	def clipStreamlines(self, isChecked):

		if self.SceneManager.streamlinesActor is None:
			msgBox = QMessageBox()
			msgBox.setText("Streamlines not created, click compute tracks button or load existing streamlines file.")
			msgBox.exec()
			return None

		self.SceneManager.clipStreamlines(isChecked, self.clipStreamlinesSlider.value(), self.xySlider.value())

	def clipStreamlinesSliderUpdate(self, value):
		self.clipStreamlinesLineEdit.setText(str(value))

		self.SceneManager.clipStreamlines(self.clipStreamlinescheckbox.isChecked(), value, self.xySlider.value())

	def clipStreamlinesEdit_changed(self):
		try:
			self.clipStreamlinesSlider.setValue(int(self.clipStreamlinesLineEdit.text()))
		except:			
			msgBox = QMessageBox()
			msgBox.setText("Incorrect value for clip streamlines, expect integer")
			msgBox.exec()
			return None

		self.SceneManager.clipStreamlines(self.clipStreamlinescheckbox.isChecked(), int(self.clipStreamlinesLineEdit.text()), self.xySlider.value())

	def showAllTracks(self, value):

		if value:
			self.SceneManager.showAllTracks(self.streamlinesVisibilityCheckbox.isChecked())
		else:
			self.visualizeTracksByColor(self.selectTracksByColorCheckbox.isChecked())

	def visualizeTracksByColor(self, isChecked):

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
			
		else:
			selected_colors = None

		self.SceneManager.visualizeTracksByColor(selected_colors, isChecked, self.streamlinesVisibilityCheckbox.isChecked())
		self.SceneManager.visualizeClustersByColor(selected_colors, isChecked, self.clustersCheckBox.isChecked())
  
	def interactiveEditingTabSelect(self, tabIndex):
	 
		# If the user selects interactive editing tab
		if tabIndex == 1:
			self.SceneManager.interactiveEditingTabSelected()	
   
			self.pickColorsListWidget.clear()
	
			for i in np.arange(self.unique_colors.shape[0]):
				listItem = QListWidgetItem('')
				listItem.setBackground(QColor(int(self.unique_colors[i][0] * 255), int(self.unique_colors[i][1] * 255),
											int(self.unique_colors[i][2] * 255), 200))
				self.pickColorsListWidget.addItem(listItem)

			self.pickColorsListWidget.setSelectionMode(QAbstractItemView.SingleSelection)

			current_slice_num = int(self.xySliceEdit2.text())
			self.SceneManager.visualizeXYSlice(current_slice_num, True)	
			self.SceneManager.showAllTracks(self.streamlinesVisibilityCheckbox.isChecked())	
   
			# Always clip streamlines by 5 slices in interactive editing tab
			self.SceneManager.clipStreamlines(True, 5, current_slice_num)
   
		if tabIndex == 0:
			self.SceneManager.visualizationTabSelected()
			current_slice_num = int(self.xySliceEdit.text())
			self.SceneManager.clipStreamlines(self.clipStreamlinescheckbox.isChecked(), self.clipStreamlinesSlider.value(), current_slice_num)

  
	def drawROI(self):
		self.startSliceIndex = int(self.xySliceEdit2.text())
		self.SceneManager.tracerWidget()
		self.statusBar().showMessage('Draw a closed contour.', 3000)	

	def addNewColorROI(self):
		
		# Get a new color that is different from all existing colors
		unique_colors_transpose = self.unique_colors.T
		current_colors_as_list_of_tuples = list(zip(unique_colors_transpose[0], unique_colors_transpose[1], unique_colors_transpose[2]))
		current_colors_as_list_of_tuples.append((0, 0, 0))
		current_colors_as_list_of_tuples.append((1, 1, 1))
  
		self.userSelectedColor = distinctipy.get_colors(1, current_colors_as_list_of_tuples)	
		self.statusBar().showMessage('New color created, track with LK or ST!', 3000)	
  
		# TODO: remove selection rect around pickColorsListWidget if possible.

	def pickColorROI(self, listWidgetItem):
		 
		selected_indices = self.pickColorListWidget.selectionModel().selectedIndexes()

		if not len(selected_indices) == 1:	
			msgBox = QMessageBox()
			msgBox.setText("More than one color picked from list, please select only one.")
			msgBox.exec()
			return None

		self.userSelectedColor = np.expand_dims(self.unique_colors[int(selected_indices[0].row()), :], axis = 1).T
		self.statusBar().showMessage('Color pick complete, track with LK or ST!', 3000)	
  
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
	
			self.SceneManager.removeStreamlinesActor()
			self.SceneManager.addStreamlinesActor(self.streamlines, self.color)

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
				self.trackingAlgoLK = 1
			if value == 2:
				self.trackingAlgoLK = 0

			# Remove contour
			self.SceneManager.removeContour()

			# Visualize streamlines
			self.SceneManager.visualizeStreamlines(self.streamlinesVisibilityCheckbox.isChecked())
			
			# Enable push buttons back again
			self.trackROI_LKButton.setEnabled(True)
			self.trackROI_STButton.setEnabled(True)
   
	def trackROI_LK(self):

		try:
			windowSize = int(self.windowSize.text())
			maxLevel = int(self.maxLevel.text())
			seedsPerPixel = float(self.seedsPerPixel.text())
			blur = float(self.blurLineEdit.text())
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

	def trackROI_ST(self):
		try:
			neighborhoodScale = int(self.neighborhoodScale.text())
			noiseScale = float(self.noiseScale.text())
			seedsPerPixel = float(self.seedsPerPixel.text())			
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

		# Re display the actors, update color lists.
		self.SceneManager.removeStreamlinesActor()
		self.SceneManager.addStreamlinesActor(self.streamlines, self.color)

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
  
		self.SceneManager.updateStreamlinesAndColors(self.streamlines, self.color)
  
		self.SceneManager.showAllTracks(self.streamlinesVisibilityCheckbox.isChecked())	
		self.SceneManager.clipStreamlines(self.clipStreamlinescheckbox.isChecked(), self.clipStreamlinesSlider.value(), self.startSliceIndex)
	  
		self.statusBar().showMessage('Deleted tracks passing through selected ROI', 3000)
  
  		# Remove contour
		self.SceneManager.removeContour()
	
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

		# Re display the streamlines actor
		self.SceneManager.removeStreamlinesActor()
		self.SceneManager.addStreamlinesActor(self.streamlines, self.color)
  
		self.SceneManager.updateStreamlinesAndColors(self.streamlines, self.color)
  
		self.SceneManager.showAllTracks(self.streamlinesVisibilityCheckbox.isChecked())	
		self.SceneManager.clipStreamlines(self.clipStreamlinescheckbox.isChecked(), self.clipStreamlinesSlider.value(), self.startSliceIndex)
   
		self.statusBar().showMessage('Deleted tracks arising from selected ROI', 3000)       
  
  		# Remove contour
		self.SceneManager.removeContour()          

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

	def actComplete(self, value):
     
		if value == 1:
			streamlines_image_coords = self.actThread.get_streamlines_image_coords()
	
			# Convert from image coordinates back to physical coordinates, save to list of lists
			self.convertStreamlinesToPhysCoords(streamlines_image_coords)

			self.actThread.terminate_thread()
  

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
   
		self.SceneManager.updateStreamlinesAndColors(self.streamlines, self.color)  
  
		self.SceneManager.showAllTracks(self.streamlinesVisibilityCheckbox.isChecked())	
		self.SceneManager.clipStreamlines(self.clipStreamlinescheckbox.isChecked(), self.clipStreamlinesSlider.value(), self.startSliceIndex)

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
		qb = QuickBundles(threshold = float(self.clusteringThresholdLineEdit.text()))
		clusters = qb.cluster(streamlines_for_clustering)

		clusterColors = np.empty((len(clusters.centroids), 3))

		for k in np.arange(len(clusters.centroids)):
			clusterColors[k,:] = self.SceneManager.mode_rows(colors_for_clustering[clusters[k].indices, :])

		streamlineClusters = clusters.centroids
		streamlineClustersColors = clusterColors 
  
		return streamlineClusters, streamlineClustersColors
  
	def compareTractograms(self):
		title = "Select first streamline file for comparison, color file should be in same folder"
		streamlineFileOnePath = QFileDialog.getOpenFileName(self,
										title,
										expanduser("."),
										"Streamline File (*.pkl)")
  
		if streamlineFileOnePath[0] == '':
			streamlineFileOnePath = None
			return

		title = "Select second streamline file for comparison, color file should be in same folder"
		streamlineFileTwoPath = QFileDialog.getOpenFileName(self,
										title,
										expanduser("."),
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
	
if __name__ == '__main__':

	app = QApplication(sys.argv)
	app.setWindowIcon(QIcon('icon.jpg'))
 
	# If you want a dark theme, uncomment one of lines below
 	# setup stylesheet
	#app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
	# or in new API
	#app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))
 
	mw = MainWindow()
	mw.show()
	sys.exit(app.exec_())