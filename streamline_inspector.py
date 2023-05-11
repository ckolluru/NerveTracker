import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

folder  = r'D:\MUSE processing\PyFibers\pickle files\comparison-fast-stain-thin-3mm\\LK 4x DS\\'
streamlinesFile = glob.glob(folder + 'streamlines*.pkl')[0]
colorsFile = os.path.dirname(streamlinesFile) + '\\colors_' + os.path.basename(streamlinesFile)[12:]
    
total_slice_count = 900

with open(streamlinesFile, 'rb') as f:
    streamlines = pickle.load(f)

with open(colorsFile, 'rb') as f:
    colors = pickle.load(f)
    
slices_in_each_streamline = np.zeros((len(streamlines)))

for k in np.arange(len(streamlines)):
    current_streamline = streamlines[k]    
    num_slices_in_current_streamline = current_streamline.shape[0]    
    slices_in_each_streamline[k] = num_slices_in_current_streamline

plt.title('Length of each streamline (in slices)')
plt.plot(slices_in_each_streamline, 'bo', markersize=2)
plt.xlabel('Streamline #')
plt.show()

streamlines_at_each_slice = np.zeros(total_slice_count)

for k in np.arange(total_slice_count):
    
    streamlines_stopped_earlier = np.sum(slices_in_each_streamline < k)
    streamlines_at_each_slice[k] = len(streamlines) - streamlines_stopped_earlier
    
plt.title('Number of streamlines at each slice')    
plt.plot(streamlines_at_each_slice, 'bo', markersize=2)
plt.xlabel('Slice number')
plt.show()