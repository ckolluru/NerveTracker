from PyQt5 import QtCore
import time

class MovieClass(QtCore.QThread):
	sliceSignal = QtCore.pyqtSignal(int)
	completeSignal = QtCore.pyqtSignal(int)
 
	def __init__(self, sliceCount):        
		super(MovieClass, self).__init__(None)
  
		self.sliceCount = sliceCount

	def run(self):
		for i in range(0, self.sliceCount, 5):
			self.sliceSignal.emit(i)
			time.sleep(1/5)
   
		self.completeSignal.emit(1)
  
	def terminate_thread(self):
		self.quit()
		self.wait()
 