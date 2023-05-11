import sys
import vtk

from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

class QVTKWidget(QVTKRenderWindowInteractor):

    def __init__(self, parent = None):
        QVTKRenderWindowInteractor.__init__(self, parent)
        self.GetRenderWindow().GetInteractor().SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
