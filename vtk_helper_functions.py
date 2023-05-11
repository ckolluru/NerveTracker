from vtkmodules.util import numpy_support
import vtk
import numpy as np
from scipy.ndimage import map_coordinates

def numpy_to_vtk_points(points):
	"""Convert Numpy points array to a vtk points array.

	Parameters
	----------
	points : ndarray

	Returns
	-------
	vtk_points : vtkPoints()

	"""
	vtk_points = vtk.vtkPoints()
	vtk_points.SetData(numpy_support.numpy_to_vtk(np.asarray(points),
												  deep=True))
	return vtk_points

def numpy_to_vtk_cells(data, is_coords=True):
	"""Convert numpy array to a vtk cell array.

	Parameters
	----------
	data : ndarray
		points coordinate or connectivity array (e.g triangles).
	is_coords : ndarray
		Select the type of array. default: True.

	Returns
	-------
	vtk_cell : vtkCellArray
		connectivity + offset information

	"""
	data = np.array(data, dtype=object)
	nb_cells = len(data)

	# Get lines_array in vtk input format
	connectivity = data.flatten() if not is_coords else []
	offset = [0, ]
	current_position = 0

	cell_array = vtk.vtkCellArray()

	for i in range(nb_cells):
		current_len = len(data[i])
		offset.append(offset[-1] + current_len)

		if is_coords:
			end_position = current_position + current_len
			connectivity += list(range(current_position, end_position))
			current_position = end_position

	connectivity = np.array(connectivity, np.intp)
	offset = np.array(offset, dtype=connectivity.dtype)

	vtk_array_type = numpy_support.get_vtk_array_type(connectivity.dtype)
	cell_array.SetData(
		numpy_support.numpy_to_vtk(offset, deep=True,
								   array_type=vtk_array_type),
		numpy_support.numpy_to_vtk(connectivity, deep=True,
								   array_type=vtk_array_type))

	cell_array.SetNumberOfCells(nb_cells)
	return cell_array

def numpy_to_vtk_colors(colors):
	"""Convert Numpy color array to a vtk color array.

	Parameters
	----------
	colors: ndarray

	Returns
	-------
	vtk_colors : vtkDataArray

	Notes
	-----
	If colors are not already in UNSIGNED_CHAR you may need to multiply by 255.

	Examples
	--------
	>>> import numpy as np
	>>> from fury.utils import numpy_to_vtk_colors
	>>> rgb_array = np.random.rand(100, 3)
	>>> vtk_colors = numpy_to_vtk_colors(255 * rgb_array)

	"""
	vtk_colors = numpy_support.numpy_to_vtk(np.asarray(colors), deep=True,
											array_type=vtk.VTK_UNSIGNED_CHAR)
	return vtk_colors

def cc(na, nd):
    return na * np.cos(nd * np.pi / 180.0)


def ss(na, nd):
    return na * np.sin(nd * np.pi / 180.0)

def boys2rgb(v):
	""" boys 2 rgb cool colormap

	Maps a given field of undirected lines (line field) to rgb
	colors using Boy's Surface immersion of the real projective
	plane.
	Boy's Surface is one of the three possible surfaces
	obtained by gluing a Mobius strip to the edge of a disk.
	The other two are the crosscap and Roman surface,
	Steiner surfaces that are homeomorphic to the real
	projective plane (Pinkall 1986). The Boy's surface
	is the only 3D immersion of the projective plane without
	singularities.
	Visit http://www.cs.brown.edu/~cad/rp2coloring for further details.
	Cagatay Demiralp, 9/7/2008.

	Code was initially in matlab and was rewritten in Python for fury by
	the FURY Team. Thank you Cagatay for putting this online.

	Parameters
	------------
	v : array, shape (N, 3) of unit vectors (e.g., principal eigenvectors of
	   tensor data) representing one of the two directions of the
	   undirected lines in a line field.

	Returns
	---------
	c : array, shape (N, 3) matrix of rgb colors corresponding to the vectors
		   given in V.

	Examples
	----------

	>>> from fury import colormap
	>>> v = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
	>>> c = colormap.boys2rgb(v)
	"""

	if v.ndim == 1:
		x = v[0]
		y = v[1]
		z = v[2]

	if v.ndim == 2:
		x = v[:, 0]
		y = v[:, 1]
		z = v[:, 2]

	x2 = x ** 2
	y2 = y ** 2
	z2 = z ** 2

	x3 = x * x2
	y3 = y * y2
	z3 = z * z2

	z4 = z * z2

	xy = x * y
	xz = x * z
	yz = y * z

	hh1 = .5 * (3 * z2 - 1) / 1.58
	hh2 = 3 * xz / 2.745
	hh3 = 3 * yz / 2.745
	hh4 = 1.5 * (x2 - y2) / 2.745
	hh5 = 6 * xy / 5.5
	hh6 = (1 / 1.176) * .125 * (35 * z4 - 30 * z2 + 3)
	hh7 = 2.5 * x * (7 * z3 - 3 * z) / 3.737
	hh8 = 2.5 * y * (7 * z3 - 3 * z) / 3.737
	hh9 = ((x2 - y2) * 7.5 * (7 * z2 - 1)) / 15.85
	hh10 = ((2 * xy) * (7.5 * (7 * z2 - 1))) / 15.85
	hh11 = 105 * (4 * x3 * z - 3 * xz * (1 - z2)) / 59.32
	hh12 = 105 * (-4 * y3 * z + 3 * yz * (1 - z2)) / 59.32

	s0 = -23.0
	s1 = 227.9
	s2 = 251.0
	s3 = 125.0

	ss23 = ss(2.71, s0)
	cc23 = cc(2.71, s0)
	ss45 = ss(2.12, s1)
	cc45 = cc(2.12, s1)
	ss67 = ss(.972, s2)
	cc67 = cc(.972, s2)
	ss89 = ss(.868, s3)
	cc89 = cc(.868, s3)

	X = 0.0

	X = X + hh2 * cc23
	X = X + hh3 * ss23

	X = X + hh5 * cc45
	X = X + hh4 * ss45

	X = X + hh7 * cc67
	X = X + hh8 * ss67

	X = X + hh10 * cc89
	X = X + hh9 * ss89

	Y = 0.0

	Y = Y + hh2 * -ss23
	Y = Y + hh3 * cc23

	Y = Y + hh5 * -ss45
	Y = Y + hh4 * cc45

	Y = Y + hh7 * -ss67
	Y = Y + hh8 * cc67

	Y = Y + hh10 * -ss89
	Y = Y + hh9 * cc89

	Z = 0.0

	Z = Z + hh1 * -2.8
	Z = Z + hh6 * -0.5
	Z = Z + hh11 * 0.3
	Z = Z + hh12 * -2.5

	# scale and normalize to fit
	# in the rgb space

	w_x = 4.1925
	trl_x = -2.0425
	w_y = 4.0217
	trl_y = -1.8541
	w_z = 4.0694
	trl_z = -2.1899

	if v.ndim == 2:

		N = len(x)
		C = np.zeros((N, 3))

		C[:, 0] = 0.9 * np.abs(((X - trl_x) / w_x)) + 0.05
		C[:, 1] = 0.9 * np.abs(((Y - trl_y) / w_y)) + 0.05
		C[:, 2] = 0.9 * np.abs(((Z - trl_z) / w_z)) + 0.05

	if v.ndim == 1:

		C = np.zeros((3,))
		C[0] = 0.9 * np.abs(((X - trl_x) / w_x)) + 0.05
		C[1] = 0.9 * np.abs(((Y - trl_y) / w_y)) + 0.05
		C[2] = 0.9 * np.abs(((Z - trl_z) / w_z)) + 0.05

	return C


def orient2rgb(v):
	"""Get Standard orientation 2 rgb colormap.

	v : array, shape (N, 3) of vectors not necessarily normalized

	Returns
	-------
	c : array, shape (N, 3) matrix of rgb colors corresponding to the vectors
		   given in V.

	Examples
	--------
	>>> from fury import colormap
	>>> v = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
	>>> c = colormap.orient2rgb(v)

	"""
	if v.ndim == 1:
		r = np.linalg.norm(v)
		orient = np.abs(np.divide(v, r, where=r != 0))

	elif v.ndim == 2:
		orientn = np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2 + v[:, 2] ** 2)
		orientn.shape = orientn.shape + (1,)
		orient = np.abs(np.divide(v, orientn, where=orientn != 0))
	else:
		raise IOError("Wrong vector dimension, It should be an array"
					  " with a shape (N, 3)")

	return orient

def line_colors(streamlines, cmap='rgb_standard'):
	"""Create colors for streamlines to be used in actor.line.

	Parameters
	----------
	streamlines : sequence of ndarrays
	cmap : ('rgb_standard', 'boys_standard')

	Returns
	-------
	colors : ndarray

	"""
	if cmap == 'rgb_standard':
		col_list = [orient2rgb(streamline[-1] - streamline[0])
					for streamline in streamlines]

	if cmap == 'boys_standard':
		col_list = [boys2rgb(streamline[-1] - streamline[0])
					for streamline in streamlines]

	return np.vstack(col_list)

def map_coordinates_3d_4d(input_array, indices):
    """Evaluate input_array at the given indices using trilinear interpolation.

    Parameters
    ----------
    input_array : ndarray,
        3D or 4D array
    indices : ndarray

    Returns
    -------
    output : ndarray
        1D or 2D array

    """
    if input_array.ndim <= 2 or input_array.ndim >= 5:
        raise ValueError("Input array can only be 3d or 4d")

    if input_array.ndim == 3:
        return map_coordinates(input_array, indices.T, order=1)

    if input_array.ndim == 4:
        values_4d = []
        for i in range(input_array.shape[-1]):
            values_tmp = map_coordinates(input_array[..., i],
                                         indices.T, order=1)
            values_4d.append(values_tmp)
        return np.ascontiguousarray(np.array(values_4d).T)

def lines_to_vtk_polydata(lines, colors=None):
	"""Create a vtkPolyData with lines and colors.

	Parameters
	----------
	lines : list
		list of N curves represented as 2D ndarrays
	colors : array (N, 3), list of arrays, tuple (3,), array (K,)
		If None or False, a standard orientation colormap is used for every
		line.
		If one tuple of color is used. Then all streamlines will have the same
		colour.
		If an array (N, 3) is given, where N is equal to the number of lines.
		Then every line is coloured with a different RGB color.
		If a list of RGB arrays is given then every point of every line takes
		a different color.
		If an array (K, 3) is given, where K is the number of points of all
		lines then every point is colored with a different RGB color.
		If an array (K,) is given, where K is the number of points of all
		lines then these are considered as the values to be used by the
		colormap.
		If an array (L,) is given, where L is the number of streamlines then
		these are considered as the values to be used by the colormap per
		streamline.
		If an array (X, Y, Z) or (X, Y, Z, 3) is given then the values for the
		colormap are interpolated automatically using trilinear interpolation.

	Returns
	-------
	poly_data : vtkPolyData
	color_is_scalar : bool, true if the color array is a single scalar
		Scalar array could be used with a colormap lut
		None if no color was used

	"""
	# Get the 3d points_array
	points_array = np.vstack(lines)

	# Set Points to vtk array format
	vtk_points = numpy_to_vtk_points(points_array)

	# Set Lines to vtk array format
	vtk_cell_array = numpy_to_vtk_cells(lines)

	# Create the poly_data
	poly_data = vtk.vtkPolyData()
	poly_data.SetPoints(vtk_points)
	poly_data.SetLines(vtk_cell_array)

	# Get colors_array (reformat to have colors for each points)
	#           - if/else tested and work in normal simple case
	nb_points = len(points_array)
	nb_lines = len(lines)
	lines_range = range(nb_lines)
	points_per_line = [len(lines[i]) for i in lines_range]
	points_per_line = np.array(points_per_line, np.intp)
	color_is_scalar = False
	if colors is None or colors is False:
		# set automatic rgb colors
		cols_arr = line_colors(lines)
		colors_mapper = np.repeat(lines_range, points_per_line, axis=0)
		vtk_colors = numpy_to_vtk_colors(255 * cols_arr[colors_mapper])
	else:
		cols_arr = np.asarray(colors)
		if cols_arr.dtype == object:  # colors is a list of colors
			vtk_colors = numpy_to_vtk_colors(255 * np.vstack(colors))
		else:
			if len(cols_arr) == nb_points:
				if cols_arr.ndim == 1:  # values for every point
					vtk_colors = numpy_support.numpy_to_vtk(cols_arr,
															deep=True)
					color_is_scalar = True
				elif cols_arr.ndim == 2:  # map color to each point
					vtk_colors = numpy_to_vtk_colors(255 * cols_arr)

			elif cols_arr.ndim == 1:
				if len(cols_arr) == nb_lines:  # values for every streamline
					cols_arrx = []
					for (i, value) in enumerate(colors):
						cols_arrx += lines[i].shape[0]*[value]
					cols_arrx = np.array(cols_arrx)
					vtk_colors = numpy_support.numpy_to_vtk(cols_arrx,
															deep=True)
					color_is_scalar = True
				else:  # the same colors for all points
					vtk_colors = numpy_to_vtk_colors(
						np.tile(255 * cols_arr, (nb_points, 1)))

			elif cols_arr.ndim == 2:  # map color to each line
				colors_mapper = np.repeat(lines_range, points_per_line, axis=0)
				vtk_colors = numpy_to_vtk_colors(255 * cols_arr[colors_mapper])
			else:  # colormap
				#  get colors for each vertex
				cols_arr = map_coordinates_3d_4d(cols_arr, points_array)
				vtk_colors = numpy_support.numpy_to_vtk(cols_arr, deep=True)
				color_is_scalar = True

	vtk_colors.SetName("colors")
	poly_data.GetPointData().SetScalars(vtk_colors)
	return poly_data, color_is_scalar