import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from dipy.segment.metric import mdf

from dipy.segment.featurespeed import ResampleFeature
from dipy.segment.metricspeed import AveragePointwiseEuclideanMetric
from dipy.segment.metric import dist

def find_closest_line_distances(source, target, sourceColors, targetColors, compareWithinColors):
    distances = []
    
    feature = ResampleFeature(nb_points=100)
    metric = AveragePointwiseEuclideanMetric(feature)
    
    for s_index, s_line in enumerate(tqdm(source)):
        closest_distance = np.inf
        sourceColor = sourceColors[s_index,:]
        
        for t_index, t_line in enumerate(target):
            
            if compareWithinColors:
                if not np.array_equiv(targetColors[t_index, :], sourceColor):
                    continue
            
            distance = dist(metric, s_line, t_line)
            if distance < closest_distance:
                closest_distance = distance
        distances.append(closest_distance)
        
    return distances
 
# Assumes that these are tractograms after clustering, else this may take a long time to run. 
def compare_tractogram_clusters(streamlinesOne, streamlinesTwo, colorsOne, colorsTwo, compareWithinColors = False): 
        
    print('streamlinesOne length: ', len(streamlinesOne))
    print('streamlinesOne one streamline shape: ', streamlinesOne[0].shape)
    
    print('streamlinesTwo length: ', len(streamlinesTwo))
    print('streamlinesTwo one streamline shape: ', streamlinesTwo[0].shape)   
    
    distances_one_two = find_closest_line_distances(streamlinesOne, streamlinesTwo, colorsOne, colorsTwo, compareWithinColors)
    distances_two_one = find_closest_line_distances(streamlinesTwo, streamlinesOne, colorsTwo, colorsOne, compareWithinColors)
    
    average_minimum_distances_one_two = round(np.mean(distances_one_two), 2)
    average_minimum_distances_two_one = round(np.mean(distances_two_one), 2)
    
    print('Average minimum distance, streamlines one to two, in microns: ', average_minimum_distances_one_two)
    print('Average minimum distance, streamlines two to one, in microns: ', average_minimum_distances_two_one)
    
    mean_average_minimum_distance = (average_minimum_distances_one_two + average_minimum_distances_two_one)/2
    
    print('Mean average minimum distance (microns): ', mean_average_minimum_distance)    
    return mean_average_minimum_distance
    
    # # Create the plot
    # fig, ax = plt.subplots()
    # ax.plot(distances_one_two, color='blue', label='$s_{1}  to  s_{2}$')
    # ax.plot(distances_two_one, color='red', label='$s_{2}  to  s_{1}$')

    # # Add axis labels and legend
    # ax.set_xlabel('Streamline #')
    # ax.set_ylabel('Minimum distances between streamlines (microns)')
    # ax.legend()

    # # Show the plot
    # plt.show()