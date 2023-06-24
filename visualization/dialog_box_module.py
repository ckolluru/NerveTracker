from PyQt5.QtWidgets import QPushButton, QFileDialog
from PyQt5.QtWidgets import QDialog, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QFormLayout, QRadioButton

# We'll need to access home directory, file path read, xml read
import untangle
import os

# Dialog box for inputting image metadata
class MetadataDialogBox(QDialog):
	def __init__(self, parent=None):
		super(MetadataDialogBox, self).__init__(parent)
  
		self.metadata = self.get_defaults()  
		self.createDialogBoxValues()
  
		# Create radio buttons for image type
		self.image_type_label = QLabel("Image type", self)
		self.png_radio_button = QRadioButton("PNG", self)
		self.png_radio_button.setChecked(True)
		self.zarr_radio_button = QRadioButton("Zarr", self)
		self.zarr_radio_button.setChecked(False)  
		
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
		layout.addRow(self.label3, self.num_images_to_read)
		layout.addRow(self.label4, self.chunk_size_z)
  
		image_type_layout = QHBoxLayout()
		image_type_layout.addWidget(self.image_type_label)
		image_type_layout.addStretch()
		image_type_layout.addWidget(self.png_radio_button)
		image_type_layout.addStretch()
		image_type_layout.addWidget(self.zarr_radio_button)

		buttons_layout = QHBoxLayout()
		buttons_layout.addWidget(self.load_button)
		buttons_layout.addWidget(self.save_button)
		buttons_layout.addStretch()
		buttons_layout.addWidget(self.ok_button)

		main_layout = QVBoxLayout(self)
		main_layout.addLayout(layout)
		main_layout.addLayout(image_type_layout)
		main_layout.addLayout(buttons_layout)
  
		self.setLayout(main_layout)
		self.setWindowTitle ('Image metadata information')
		self.setWhatsThis("Provide metadata information for the image stack.")

    # Set default values for fields in the dialog box
	def get_defaults(self):
    
		metadata = dict()
		metadata['pixel_size_xy'] = 0.9
		metadata['section_thickness'] = 3.0
		metadata['image_type'] = '.png'
		metadata['num_images_to_read'] = 1500
		metadata['step_size'] = 64

		return metadata
		
	# Create the labels and input fields
	def createDialogBoxValues(self):

		self.label1 = QLabel('Pixel size (XY) in microns:', self)
		self.pixel_size_xy = QLineEdit(str(self.metadata['pixel_size_xy']), self)
		self.label2 = QLabel('Section thickness in microns:', self)
		self.section_thickness = QLineEdit(str(self.metadata['section_thickness']), self)
		self.label3 = QLabel('Number of images to read:', self)
		self.num_images_to_read = QLineEdit(str(self.metadata['num_images_to_read']), self)
		self.label4 = QLabel('Chunk size in Z (used for ST analysis):', self)
		self.chunk_size_z = QLineEdit(str(self.metadata['step_size']), self)  

	# Update dialog box from XML file
	def updateDialogBox(self):
    
		self.pixel_size_xy.setText(str(self.metadata['pixel_size_xy']))
		self.section_thickness.setText(str(self.metadata['section_thickness']))
		self.num_images_to_read.setText(str(self.metadata['num_images_to_read']))
		self.chunk_size_z.setText(str(self.metadata['step_size']))
  
		if str(self.metadata['image_type']) == '.png':
			self.png_radio_button.setChecked(True)
		if str(self.metadata['image_type']) == '.zarr':
			self.zarr_radio_button.setChecked(True)
  
	# Define a method to return the user input when the dialog is accepted
	def get_metadata(self):
		self.metadata['pixel_size_xy'] = float(self.pixel_size_xy.text())
		self.metadata['section_thickness'] = float(self.section_thickness.text())
		self.metadata['num_images_to_read'] = int(self.num_images_to_read.text())
		self.metadata['step_size'] = int(self.chunk_size_z.text())
  
		if self.png_radio_button.isChecked():
			self.metadata['image_type'] = '.png'
		else:
			self.metadata['image_type'] = '.zarr'
  		
		return self.metadata
  
	# Load metadata from XML
	def load(self):
    
		title = "Open Image metadata XML file"
		self.imageMetadataXMLFileName = QFileDialog.getOpenFileName(self,
										title,
										os.path.expanduser("."),
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

# Dialog box for inputting validation metadata
class ValidationMetadataDialogBox(QDialog):
	def __init__(self, parent=None):
		super(ValidationMetadataDialogBox, self).__init__(parent)
  
		self.metadata = self.get_defaults()  
		self.createDialogBoxValues()
		
		# Create the OK button and connect it to the accept method
		self.ok_button = QPushButton('OK', self)
		self.ok_button.clicked.connect(self.accept)	
  
		# Create the layout and add the widgets to it
		layout = QFormLayout()
		layout.addRow(self.label1, self.ds_factor)
		layout.addRow(self.label2, self.section_thickness)
		layout.addRow(self.label3, self.slice_nums_to_evaluate)
		layout.addRow(self.label4, self.normalize_index)
		layout.addRow(self.label5, self.image_height)
		layout.addRow(self.label6, self.image_width)
		layout.addRow(self.label7, self.pixel_size)

		buttons_layout = QHBoxLayout()
		buttons_layout.addStretch()
		buttons_layout.addWidget(self.ok_button)
		buttons_layout.addStretch()

		main_layout = QVBoxLayout(self)
		main_layout.addLayout(layout)
		main_layout.addLayout(buttons_layout)
  
		self.setLayout(main_layout)
		self.setWindowTitle ('Validation method metadata information')
		self.setWhatsThis("Provide metadata information for the validation stack.")

    # Set default values for fields in the dialog box
	def get_defaults(self):
    
		# sample 1
		metadata = dict()
		metadata['ds_factor'] = 1
		metadata['section_thickness'] = 3.0
		metadata['slice_numbers_to_evaluate'] = "0, 499, 999, 1499"
		metadata['normalize_wrt_slice_physical_distance_microns_array_index'] = 0
		metadata['image_height'] = 3000
		metadata['image_width'] = 4000
		metadata['pixel_size'] = 0.9
  
		# sample 2
		# metadata['ds_factor'] = 1
		# metadata['section_thickness'] = 3.0
		# metadata['slice_numbers_to_evaluate'] = "0, 300, 600, 902"
		# metadata['normalize_wrt_slice_physical_distance_microns_array_index'] = 0
		# metadata['image_height'] = 2380
		# metadata['image_width'] = 2196
		# metadata['pixel_size'] = 0.74
  
		return metadata
		
	# Create the labels and input fields
	def createDialogBoxValues(self):

		self.label1 = QLabel('Downsampling factor:', self)
		self.ds_factor = QLineEdit(str(self.metadata['ds_factor']), self)
		self.label2 = QLabel('Section thickness in microns:', self)
		self.section_thickness = QLineEdit(str(self.metadata['section_thickness']), self)
		self.label3 = QLabel('Slice numbers to evaluate at (seperated by commas):', self)
		self.slice_nums_to_evaluate = QLineEdit(str(self.metadata['slice_numbers_to_evaluate']), self)
		self.label4 = QLabel('Normalize w.r.t slice array index:', self)
		self.normalize_index = QLineEdit(str(self.metadata['normalize_wrt_slice_physical_distance_microns_array_index']), self)
		self.label5 = QLabel('Image height:', self)
		self.image_height = QLineEdit(str(self.metadata['image_height']), self)  
		self.label6 = QLabel('Image width:', self)
		self.image_width = QLineEdit(str(self.metadata['image_width']), self)  
		self.label7 = QLabel('Original pixel size:', self)
		self.pixel_size = QLineEdit(str(self.metadata['pixel_size']), self)  
  
	# Define a method to return the user input when the dialog is accepted
	def get_metadata(self):
		self.metadata['ds_factor'] = float(self.ds_factor.text())
		self.metadata['section_thickness'] = float(self.section_thickness.text())
		slice_nums_string = self.slice_nums_to_evaluate.text()
		self.metadata['slice_numbers_to_evaluate'] =  [int(i) for i in slice_nums_string.split(',')]  		
		self.metadata['normalize_wrt_slice_physical_distance_microns_array_index'] = int(self.normalize_index.text())
		self.metadata['image_height'] = int(self.image_height.text())
		self.metadata['image_width'] = int(self.image_width.text())
		self.metadata['pixel_size'] = float(self.pixel_size.text())
  		
		return self.metadata